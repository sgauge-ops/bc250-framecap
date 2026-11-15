use crate::layer::diagnostics;
use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Paired GPU and CPU clock readings from calibrated timestamps, plus a CPU-side reference point and its uncertainty.
#[derive(Clone, Copy)]
pub(crate) struct Calibration {
    pub gpu_ticks: u64,
    pub cpu_ns: u64,
    pub max_deviation_ns: u64,
    pub reference_cpu_ns: u64,
    pub reference_instant: Instant,
    pub reference_uncertainty_ns: u64,
}

/// Latest outcome of the calibration worker: no sample yet, a ready sample, a failure, or the value being locked.
#[derive(Clone, Copy)]
pub(crate) enum CalibrationState {
    Waiting,
    Ready(Calibration),
    Error,
    Busy,
}

/// Background thread that periodically samples GPU/CPU clock calibration while a capture is active.
pub(crate) struct CalibrationWorker {
    latest: Arc<RwLock<CalibrationState>>,
    stop: mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl CalibrationWorker {
    /// Spawns the worker, which samples every 100 ms during captures and logs failure and recovery transitions.
    pub(crate) fn start(
        device: vk::Device,
        calibrate: vk::PFN_vkGetCalibratedTimestampsKHR,
    ) -> std::io::Result<Self> {
        let latest = Arc::new(RwLock::new(CalibrationState::Waiting));
        let worker_latest = Arc::clone(&latest);
        let (stop, receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("framecap-gpu-calibration".to_owned())
            .spawn(move || {
                let mut failed = false;
                loop {
                    if crate::layer::capture::is_active() {
                        match unsafe { calibration_sample(device, calibrate) } {
                            Ok(sample) => {
                                *worker_latest.write() = CalibrationState::Ready(sample);
                                if failed {
                                    diagnostics::info("GPU timestamp calibration recovered");
                                    failed = false;
                                }
                            }
                            Err(error) => {
                                *worker_latest.write() = CalibrationState::Error;
                                if !failed {
                                    diagnostics::error(format!(
                                        "GPU timestamp calibration failed: {error}"
                                    ));
                                    failed = true;
                                }
                            }
                        }
                    } else {
                        *worker_latest.write() = CalibrationState::Waiting;
                    }
                    match receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                }
            })?;
        Ok(Self {
            latest,
            stop,
            thread: Some(thread),
        })
    }

    /// Returns the latest state without blocking, reporting `Busy` if the worker holds the lock.
    pub(crate) fn current(&self) -> CalibrationState {
        self.latest
            .try_read()
            .map(|latest| *latest)
            .unwrap_or(CalibrationState::Busy)
    }

    /// Signals the worker thread to exit and joins it.
    pub(crate) fn stop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                diagnostics::error("GPU timestamp calibration worker panicked");
            }
        }
    }
}

/// Reads device and CLOCK_MONOTONIC timestamps in one call and brackets a direct clock read to build a `Calibration`.
unsafe fn calibration_sample(
    device: vk::Device,
    calibrate: vk::PFN_vkGetCalibratedTimestampsKHR,
) -> Result<Calibration, String> {
    let infos = [
        vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE),
        vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::CLOCK_MONOTONIC),
    ];
    let mut timestamps = [0_u64; 2];
    let mut max_deviation_ns = 0;
    let result = calibrate(
        device,
        infos.len() as u32,
        infos.as_ptr(),
        timestamps.as_mut_ptr(),
        &mut max_deviation_ns,
    );
    if result != vk::Result::SUCCESS {
        return Err(format!("{result:?}"));
    }
    if max_deviation_ns == 0 {
        return Err("calibration returned zero maximum deviation".to_owned());
    }
    let before = Instant::now();
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let after = Instant::now();
    let elapsed = after.duration_since(before);
    if !(0..1_000_000_000).contains(&time.tv_nsec) {
        return Err("invalid CLOCK_MONOTONIC nanoseconds".to_owned());
    }
    let reference_cpu_ns = u64::try_from(time.tv_sec)
        .ok()
        .and_then(|seconds| seconds.checked_mul(1_000_000_000))
        .and_then(|seconds| {
            u64::try_from(time.tv_nsec)
                .ok()
                .and_then(|nanos| seconds.checked_add(nanos))
        })
        .ok_or_else(|| "invalid CLOCK_MONOTONIC timestamp".to_owned())?;
    Ok(Calibration {
        gpu_ticks: timestamps[0],
        cpu_ns: timestamps[1],
        max_deviation_ns,
        reference_cpu_ns,
        reference_instant: before + elapsed / 2,
        reference_uncertainty_ns: (elapsed.as_nanos() / 2)
            .saturating_add(1)
            .min(u128::from(u64::MAX)) as u64,
    })
}

/// Raw GPU timestamp query result with the tick period and valid bit count of its queue family.
#[derive(Clone, Copy)]
pub(crate) struct Timestamp {
    pub ticks: u64,
    pub period_ns: f32,
    pub valid_bits: u32,
}

/// State of a GPU timestamp marker: awaiting the GPU, resolved, or failed.
#[derive(Clone, Copy)]
pub(crate) enum SampleResult {
    Pending,
    Ready(Timestamp),
    Error,
}

/// Shared handle through which a marker's result is filled in once the GPU finishes.
pub(crate) type Sample = Arc<Mutex<SampleResult>>;

/// Outcome of submitting a marker: the semaphore to wait on and its sample, or the reason it was skipped.
pub(crate) enum Submission {
    Submitted(vk::Semaphore, Sample),
    Skipped(&'static str),
}

/// Recorded timestamp command buffer, with its fence and sample, for the caller to submit itself.
pub(crate) struct PreparedMarker {
    pub command: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub sample: Sample,
}

/// Outcome of preparing an overlay marker: the recorded marker, or the reason it was skipped.
pub(crate) enum Preparation {
    Prepared(PreparedMarker),
    Skipped(&'static str),
}

/// Vulkan resources for one timestamp marker (command pool/buffer, query, optional relay semaphore, fence) per swapchain image.
struct Slot {
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    query: vk::QueryPool,
    signal: vk::Semaphore,
    fence: vk::Fence,
    family: u32,
    pending: Option<Sample>,
    poisoned: bool,
}

impl Slot {
    /// Creates the slot's Vulkan objects for `family`, destroying any partial set on failure; the semaphore exists only if `relay_signal`.
    unsafe fn new(
        device: &ash::Device,
        family: u32,
        relay_signal: bool,
    ) -> Result<Self, vk::Result> {
        let mut slot = Self {
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            query: vk::QueryPool::null(),
            signal: vk::Semaphore::null(),
            fence: vk::Fence::null(),
            family,
            pending: None,
            poisoned: false,
        };
        let result = (|| {
            slot.pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            slot.command = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(slot.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            slot.query = device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(1),
                None,
            )?;
            if relay_signal {
                slot.signal = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            }
            slot.fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            Ok(())
        })();
        if let Err(error) = result {
            slot.destroy(device);
            return Err(error);
        }
        Ok(slot)
    }

    /// Resolves the pending sample if the fence has signalled; returns whether the slot is free for reuse.
    unsafe fn poll(
        &mut self,
        device: &ash::Device,
        period_ns: f32,
        valid_bits: u32,
    ) -> Result<bool, vk::Result> {
        if self.pending.is_none() {
            return Ok(true);
        }
        if !device.get_fence_status(self.fence)? {
            return Ok(false);
        }
        let mut ticks = [0_u64];
        device.get_query_pool_results(self.query, 0, &mut ticks, vk::QueryResultFlags::TYPE_64)?;
        if let Some(sample) = self.pending.take() {
            *sample.lock() = SampleResult::Ready(Timestamp {
                ticks: ticks[0],
                period_ns,
                valid_bits,
            });
        }
        Ok(true)
    }

    /// Resets the fence and re-records the command buffer to write a bottom-of-pipe timestamp.
    unsafe fn prepare(&mut self, device: &ash::Device) -> Result<(), vk::Result> {
        device.reset_fences(&[self.fence])?;
        device.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            self.command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        device.cmd_reset_query_pool(self.command, self.query, 0, 1);
        device.cmd_write_timestamp(
            self.command,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.query,
            0,
        );
        device.end_command_buffer(self.command)?;
        Ok(())
    }

    /// Submits the marker on `queue` after `waits`, signalling the slot's relay semaphore, and returns its pending sample.
    unsafe fn submit(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        waits: &[vk::Semaphore],
    ) -> Result<Sample, vk::Result> {
        if self.signal == vk::Semaphore::null() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        self.prepare(device)?;
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let commands = [self.command];
        let signals = [self.signal];
        let submits = [vk::SubmitInfo::default()
            .wait_semaphores(waits)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&commands)
            .signal_semaphores(&signals)];
        device.queue_submit(queue, &submits, self.fence)?;
        let sample = Arc::new(Mutex::new(SampleResult::Pending));
        self.pending = Some(Arc::clone(&sample));
        Ok(sample)
    }

    /// Permanently disables the slot and resolves any pending sample as an error.
    fn fail(&mut self) {
        self.poisoned = true;
        if let Some(sample) = self.pending.take() {
            *sample.lock() = SampleResult::Error;
        }
    }

    /// Destroys whichever Vulkan objects the slot has created.
    unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.fence != vk::Fence::null() {
            device.destroy_fence(self.fence, None);
        }
        if self.signal != vk::Semaphore::null() {
            device.destroy_semaphore(self.signal, None);
        }
        if self.query != vk::QueryPool::null() {
            device.destroy_query_pool(self.query, None);
        }
        if self.pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Per-device pool of timestamp marker slots keyed by swapchain and image index, with slots of destroyed swapchains awaiting cleanup.
pub(crate) struct Markers {
    families: Vec<vk::QueueFamilyProperties>,
    period_ns: f32,
    relay_signal: bool,
    slots: HashMap<(u64, u32), Slot>,
    retired: Vec<Slot>,
}

/// Result of slot lookup: a slot ready for a new marker, or the reason none can be used.
enum AvailableSlot<'a> {
    Ready(&'a mut Slot),
    Skipped(&'static str),
}

impl Markers {
    /// Creates markers whose submissions signal a relay semaphore for the present to wait on.
    pub(crate) fn new(families: Vec<vk::QueueFamilyProperties>, period_ns: f32) -> Self {
        Self {
            families,
            period_ns,
            relay_signal: true,
            slots: HashMap::new(),
            retired: Vec::new(),
        }
    }

    /// Creates markers for overlay use, where the caller submits the command buffer itself, so no relay semaphore is created.
    pub(crate) fn new_overlay(families: Vec<vk::QueueFamilyProperties>, period_ns: f32) -> Self {
        Self {
            relay_signal: false,
            ..Self::new(families, period_ns)
        }
    }

    /// Resolves finished markers across all slots, failing any slot whose query read errors.
    pub(crate) unsafe fn poll(&mut self, device: &ash::Device) {
        for slot in self.slots.values_mut().chain(self.retired.iter_mut()) {
            let Some(family) = self.families.get(slot.family as usize) else {
                continue;
            };
            if let Err(error) = slot.poll(device, self.period_ns, family.timestamp_valid_bits) {
                slot.fail();
                diagnostics::error(format!("GPU capture query failed: {error:?}"));
            }
        }
    }

    /// Finds or creates the slot for a swapchain image, skipping unsupported queues, resource limits, disabled slots and pending queries.
    unsafe fn available_slot(
        &mut self,
        device: &ash::Device,
        family: u32,
        swapchain: u64,
        image_index: u32,
    ) -> Result<AvailableSlot<'_>, vk::Result> {
        let Some(properties) = self.families.get(family as usize) else {
            return Ok(AvailableSlot::Skipped("unknown_queue_family"));
        };
        if properties.timestamp_valid_bits == 0
            || !properties
                .queue_flags
                .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            || !self.period_ns.is_finite()
            || self.period_ns <= 0.0
        {
            return Ok(AvailableSlot::Skipped("unsupported_queue"));
        }
        let key = (swapchain, image_index);
        if !self.slots.contains_key(&key) {
            if self.slots.len() + self.retired.len() >= 1024 {
                return Ok(AvailableSlot::Skipped("resource_limit"));
            }
            self.slots
                .insert(key, Slot::new(device, family, self.relay_signal)?);
        }
        let slot = self.slots.get_mut(&key).expect("GPU capture slot inserted");
        if slot.family != family {
            return Ok(AvailableSlot::Skipped("different_queue_family"));
        }
        if slot.poisoned {
            return Ok(AvailableSlot::Skipped("slot_disabled"));
        }
        match slot.poll(device, self.period_ns, properties.timestamp_valid_bits) {
            Ok(true) => {}
            Ok(false) => return Ok(AvailableSlot::Skipped("query_pending")),
            Err(error) => {
                slot.fail();
                return Err(error);
            }
        }
        Ok(AvailableSlot::Ready(slot))
    }

    /// Submits a timestamp marker on `queue` for a swapchain image, or reports why it was skipped.
    pub(crate) unsafe fn submit(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        family: u32,
        swapchain: u64,
        image_index: u32,
        waits: &[vk::Semaphore],
    ) -> Result<Submission, vk::Result> {
        let slot = match self.available_slot(device, family, swapchain, image_index)? {
            AvailableSlot::Ready(slot) => slot,
            AvailableSlot::Skipped(reason) => return Ok(Submission::Skipped(reason)),
        };
        match slot.submit(device, queue, waits) {
            Ok(sample) => Ok(Submission::Submitted(slot.signal, sample)),
            Err(error) => {
                slot.fail();
                Err(error)
            }
        }
    }

    /// Records an overlay timestamp command buffer for a swapchain image without submitting it, or reports why it was skipped.
    pub(crate) unsafe fn prepare_overlay(
        &mut self,
        device: &ash::Device,
        family: u32,
        swapchain: u64,
        image_index: u32,
    ) -> Result<Preparation, vk::Result> {
        let slot = match self.available_slot(device, family, swapchain, image_index)? {
            AvailableSlot::Ready(slot) => slot,
            AvailableSlot::Skipped(reason) => return Ok(Preparation::Skipped(reason)),
        };
        if let Err(error) = slot.prepare(device) {
            slot.fail();
            return Err(error);
        }
        let sample = Arc::new(Mutex::new(SampleResult::Pending));
        slot.pending = Some(Arc::clone(&sample));
        Ok(Preparation::Prepared(PreparedMarker {
            command: slot.command,
            fence: slot.fence,
            sample,
        }))
    }

    /// Fails the image's slot, disabling it and resolving its pending sample as an error.
    pub(crate) fn fail(&mut self, swapchain: u64, image_index: u32) {
        if let Some(slot) = self.slots.get_mut(&(swapchain, image_index)) {
            slot.fail();
        }
    }

    /// Disables the image's slot so it is skipped from now on, leaving any pending sample untouched.
    pub(crate) fn poison(&mut self, swapchain: u64, image_index: u32) {
        if let Some(slot) = self.slots.get_mut(&(swapchain, image_index)) {
            slot.poisoned = true;
        }
    }

    /// Moves a destroyed swapchain's slots to the retired list; they are destroyed with the device.
    pub(crate) fn retire(&mut self, swapchain: u64) {
        let keys: Vec<_> = self
            .slots
            .keys()
            .filter(|(handle, _)| *handle == swapchain)
            .copied()
            .collect();
        for key in keys {
            if let Some(slot) = self.slots.remove(&key) {
                self.retired.push(slot);
            }
        }
    }

    /// Polls a final time, then destroys the Vulkan objects of all active and retired slots.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        self.poll(device);
        for slot in self.slots.values_mut().chain(self.retired.iter_mut()) {
            slot.destroy(device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "system" fn calibrate(
        _device: vk::Device,
        count: u32,
        infos: *const vk::CalibratedTimestampInfoKHR<'_>,
        timestamps: *mut u64,
        deviation: *mut u64,
    ) -> vk::Result {
        assert_eq!(count, 2);
        assert_eq!((*infos).time_domain, vk::TimeDomainKHR::DEVICE);
        assert_eq!(
            (*infos.add(1)).time_domain,
            vk::TimeDomainKHR::CLOCK_MONOTONIC
        );
        *timestamps = 123;
        *timestamps.add(1) = 456;
        *deviation = 789;
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn fail(
        _device: vk::Device,
        _count: u32,
        _infos: *const vk::CalibratedTimestampInfoKHR<'_>,
        _timestamps: *mut u64,
        _deviation: *mut u64,
    ) -> vk::Result {
        vk::Result::ERROR_DEVICE_LOST
    }

    unsafe extern "system" fn zero_deviation(
        _device: vk::Device,
        _count: u32,
        _infos: *const vk::CalibratedTimestampInfoKHR<'_>,
        _timestamps: *mut u64,
        deviation: *mut u64,
    ) -> vk::Result {
        *deviation = 0;
        vk::Result::SUCCESS
    }

    #[test]
    fn calibrated_samples_keep_clock_domains_and_report_errors() {
        let before = Instant::now();
        let sample = unsafe { calibration_sample(vk::Device::null(), calibrate) }.unwrap();
        let after = Instant::now();
        assert_eq!(
            (sample.gpu_ticks, sample.cpu_ns, sample.max_deviation_ns),
            (123, 456, 789)
        );
        assert!(sample.reference_cpu_ns > 0);
        assert!(sample.reference_instant >= before && sample.reference_instant <= after);
        assert!(unsafe { calibration_sample(vk::Device::null(), fail) }.is_err());
        assert!(unsafe { calibration_sample(vk::Device::null(), zero_deviation) }.is_err());
    }

    #[test]
    #[ignore = "requires a Vulkan device with timestamp support"]
    fn overlay_marker_uses_the_existing_submission_and_fence() {
        unsafe {
            let entry = ash::Entry::load().expect("load Vulkan");
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default(), None)
                .expect("create Vulkan instance");
            let (physical, family, families) = instance
                .enumerate_physical_devices()
                .expect("enumerate Vulkan devices")
                .into_iter()
                .find_map(|physical| {
                    let families = instance.get_physical_device_queue_family_properties(physical);
                    families
                        .iter()
                        .position(|properties| {
                            properties.queue_count > 0
                                && properties.timestamp_valid_bits > 0
                                && properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        })
                        .map(|family| (physical, family as u32, families))
                })
                .expect("graphics queue with timestamps");
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let device = instance
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                    None,
                )
                .expect("create Vulkan device");
            let queue = device.get_device_queue(family, 0);
            let period = instance
                .get_physical_device_properties(physical)
                .limits
                .timestamp_period;
            let mut before = Markers::new(families.clone(), period);
            let mut after = Markers::new_overlay(families, period);
            let (signal, before_sample) = match before
                .submit(&device, queue, family, 42, 0, &[])
                .expect("submit pre-overlay marker")
            {
                Submission::Submitted(signal, sample) => (signal, sample),
                Submission::Skipped(reason) => panic!("pre-overlay marker skipped: {reason}"),
            };
            let marker = match after
                .prepare_overlay(&device, family, 42, 0)
                .expect("prepare post-overlay marker")
            {
                Preparation::Prepared(marker) => marker,
                Preparation::Skipped(reason) => panic!("post-overlay marker skipped: {reason}"),
            };
            assert!(matches!(*marker.sample.lock(), SampleResult::Pending));
            assert!(matches!(
                after.prepare_overlay(&device, family, 42, 0).unwrap(),
                Preparation::Skipped("query_pending")
            ));

            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
                .expect("create overlay command pool");
            let command = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("allocate overlay command")[0];
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .expect("begin overlay command");
            device
                .end_command_buffer(command)
                .expect("end overlay command");
            let waits = [signal];
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let commands = [command, marker.command];
            let submits = [vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&commands)];
            device
                .queue_submit(queue, &submits, marker.fence)
                .expect("submit overlay and marker together");
            device
                .wait_for_fences(&[marker.fence], true, 5_000_000_000)
                .expect("wait for overlay marker");
            before.poll(&device);
            after.poll(&device);
            let before_timestamp = match *before_sample.lock() {
                SampleResult::Ready(timestamp) => timestamp,
                _ => panic!("pre-overlay timestamp not ready"),
            };
            let after_timestamp = match *marker.sample.lock() {
                SampleResult::Ready(timestamp) => timestamp,
                _ => panic!("post-overlay timestamp not ready"),
            };
            assert_eq!(before_timestamp.period_ns, after_timestamp.period_ns);
            assert_eq!(before_timestamp.valid_bits, after_timestamp.valid_bits);
            let bits = before_timestamp.valid_bits;
            let mask = u64::MAX >> (64 - bits);
            let elapsed_ticks = after_timestamp.ticks.wrapping_sub(before_timestamp.ticks) & mask;
            assert!(elapsed_ticks < (1_u64 << (bits - 1)));

            let failed = match after.prepare_overlay(&device, family, 42, 0).unwrap() {
                Preparation::Prepared(marker) => marker,
                Preparation::Skipped(reason) => panic!("reused marker skipped: {reason}"),
            };
            after.fail(42, 0);
            assert!(matches!(*failed.sample.lock(), SampleResult::Error));
            assert!(matches!(
                after.prepare_overlay(&device, family, 42, 0).unwrap(),
                Preparation::Skipped("slot_disabled")
            ));
            before.destroy(&device);
            after.destroy(&device);
            device.destroy_command_pool(pool, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}
