use crate::layer::auto::{AutoEvaluationSchedule, AutoRegulator};
use crate::layer::auto_metrics::AutoMetricsHistory;
use crate::layer::capture::{self, CaptureFrame};
use crate::layer::config::{Config, Hotkey};
use crate::layer::diagnostics;
use crate::layer::gpu_capture::{CalibrationWorker, Markers, Preparation, Submission};
use crate::layer::input_x11::HotkeyWorker;
use crate::layer::lib::{GpuMeasurement, HardwareMetrics, MemoryTemperature, TelemetryStatus};
use crate::layer::limiter::{wait_until, FrameCapController, FrameScheduler};
use crate::layer::renderer::{
    OverlayFeedbackEvent, OverlayFeedbackState, OverlayMode, OverlayModeState,
    OverlayRendererState, VulkanSwapchainRenderer,
};
use crate::layer::telemetry::{
    AmdGpuProvider, CpuUsageProvider, GpuTelemetryProvider, GpuUsageSmoother,
    HardwareMetricsSmoother, HwmonMetricsProvider, MemoryUsageProvider,
};
use ash::vk::{self, Handle};
use parking_lot::{Condvar, Mutex};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{c_char, c_void, CStr};
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const LOADER_INSTANCE_CREATE_INFO: i32 = 47;
const LOADER_DEVICE_CREATE_INFO: i32 = 48;
const LAYER_LINK_INFO: i32 = 0;
/// Optional `vk_layerGetPhysicalDeviceProcAddr` function pointer from the loader interface.
type PfnGetPhysicalDeviceProcAddr =
    Option<unsafe extern "system" fn(vk::Instance, *const c_char) -> vk::PFN_vkVoidFunction>;

/// Loader-provided linked-list node holding the next layer's instance entry points.
#[repr(C)]
struct LayerInstanceLink {
    next: *mut LayerInstanceLink,
    next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    next_get_physical_device_proc_addr: PfnGetPhysicalDeviceProcAddr,
}

/// Payload of the instance layer create-info: either the link chain or loader data callback.
#[repr(C)]
union InstanceLayerInfoUnion {
    layer_info: *mut LayerInstanceLink,
    set_loader_data: *const c_void,
}

/// Loader-defined `VkLayerInstanceCreateInfo` chained into `VkInstanceCreateInfo::pNext`.
#[repr(C)]
struct LayerInstanceCreateInfo {
    s_type: i32,
    next: *const c_void,
    function: i32,
    u: InstanceLayerInfoUnion,
}

/// Loader-provided linked-list node holding the next layer's instance and device entry points.
#[repr(C)]
struct LayerDeviceLink {
    next: *mut LayerDeviceLink,
    next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

/// Payload of the device layer create-info: either the link chain or loader data callback.
#[repr(C)]
union DeviceLayerInfoUnion {
    layer_info: *mut LayerDeviceLink,
    set_loader_data: *const c_void,
}

/// Loader-defined `VkLayerDeviceCreateInfo` chained into `VkDeviceCreateInfo::pNext`.
#[repr(C)]
struct LayerDeviceCreateInfo {
    s_type: i32,
    next: *const c_void,
    function: i32,
    u: DeviceLayerInfoUnion,
}

/// Loader/layer interface negotiation structure (`VkNegotiateLayerInterface`).
#[repr(C)]
pub struct NegotiateLayerInterface {
    s_type: i32,
    next: *mut c_void,
    loader_layer_interface_version: u32,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    get_physical_device_proc_addr: PfnGetPhysicalDeviceProcAddr,
}

/// Per-instance dispatch data: next `vkGetInstanceProcAddr` and calibrated-timestamp support flags.
#[derive(Clone, Copy)]
struct InstanceDispatch {
    handle: vk::Instance,
    next_gipa: vk::PFN_vkGetInstanceProcAddr,
    calibration_core: bool,
    calibration_properties2: bool,
}

/// Downstream submit and wait entry points that the layer intercepts for tracing.
#[derive(Clone, Copy)]
struct TraceDispatch {
    queue_submit: Option<vk::PFN_vkQueueSubmit>,
    queue_submit2: Option<vk::PFN_vkQueueSubmit2>,
    queue_submit2_khr: Option<vk::PFN_vkQueueSubmit2>,
    wait_fences: Option<vk::PFN_vkWaitForFences>,
    wait_semaphores: Option<vk::PFN_vkWaitSemaphores>,
    wait_semaphores_khr: Option<vk::PFN_vkWaitSemaphores>,
    queue_wait_idle: Option<vk::PFN_vkQueueWaitIdle>,
    device_wait_idle: Option<vk::PFN_vkDeviceWaitIdle>,
}

impl TraceDispatch {
    /// Resolves the downstream trace entry points for a device; missing ones stay `None`.
    unsafe fn load(device: vk::Device, next: vk::PFN_vkGetDeviceProcAddr) -> Self {
        Self {
            queue_submit: next(device, c"vkQueueSubmit".as_ptr()).map(|p| mem::transmute(p)),
            queue_submit2: next(device, c"vkQueueSubmit2".as_ptr()).map(|p| mem::transmute(p)),
            queue_submit2_khr: next(device, c"vkQueueSubmit2KHR".as_ptr())
                .map(|p| mem::transmute(p)),
            wait_fences: next(device, c"vkWaitForFences".as_ptr()).map(|p| mem::transmute(p)),
            wait_semaphores: next(device, c"vkWaitSemaphores".as_ptr()).map(|p| mem::transmute(p)),
            wait_semaphores_khr: next(device, c"vkWaitSemaphoresKHR".as_ptr())
                .map(|p| mem::transmute(p)),
            queue_wait_idle: next(device, c"vkQueueWaitIdle".as_ptr()).map(|p| mem::transmute(p)),
            device_wait_idle: next(device, c"vkDeviceWaitIdle".as_ptr()).map(|p| mem::transmute(p)),
        }
    }
}

/// Per-device layer state: downstream dispatch, telemetry, overlay, pacing, auto regulation and swapchains.
struct DeviceState {
    next_gdpa: vk::PFN_vkGetDeviceProcAddr,
    next_present: Option<vk::PFN_vkQueuePresentKHR>,
    next_acquire: Option<vk::PFN_vkAcquireNextImageKHR>,
    next_acquire2: Option<vk::PFN_vkAcquireNextImage2KHR>,
    trace_dispatch: TraceDispatch,
    ash_device: ash::Device,
    telemetry: TelemetryWorker,
    telemetry_expiry: Duration,
    fps_smoothing_window: Duration,
    minimum_metrics_collection_interval: Duration,
    overlay_refresh_rate: u64,
    scale_2x_min_width_px: u32,
    overlay_background_opacity: f32,
    overlay_renderer: Arc<OverlayRendererState>,
    overlay_fps_color: [u8; 3],
    queue_families: HashMap<usize, u32>,
    protected_queues: HashSet<usize>,
    multi_device: bool,
    gpu_markers: Markers,
    gpu_overlay_markers: Markers,
    gpu_calibration: Option<CalibrationWorker>,
    frame_cap: Arc<FrameCapController>,
    overlay_mode: Arc<OverlayModeState>,
    active_auto_algorithm: Arc<AtomicI32>,
    auto_hysteresis: Arc<Vec<AtomicU32>>,
    hotkey_config: [Hotkey; 4],
    manual_step_fps: f32,
    overlay_feedback: Arc<OverlayFeedbackState>,
    auto_regulator: AutoRegulator,
    auto_evaluation_schedule: AutoEvaluationSchedule,
    auto_metrics: HashMap<u64, Vec<AutoMetricsHistory>>,
    auto_evaluation_swapchain: Option<u64>,
    schedulers: HashMap<usize, FrameScheduler>,
    present_modes: HashMap<u64, vk::PresentModeKHR>,
    hotkeys: Option<HotkeyWorker>,
    swapchains: HashMap<u64, VulkanSwapchainRenderer>,
}

/// Background GPU/hardware telemetry poller sharing its latest sample with presenting threads.
struct TelemetryWorker {
    latest: Arc<parking_lot::RwLock<TelemetrySample>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    memory: Option<MemoryTemperatureWorker>,
}

/// Background poller of the GDDR6 hotspot temperature; stops permanently after a read error.
struct MemoryTemperatureWorker {
    latest: Arc<parking_lot::RwLock<Option<(MemoryTemperature, Instant)>>>,
    stop: Arc<(Mutex<bool>, Condvar)>,
    worker: Option<JoinHandle<()>>,
    interval: Duration,
    stale_reported: AtomicBool,
}

/// Adds a sample and returns the mean over the window; the result is a lower bound if any sample was.
fn smooth_memory_temperature(
    samples: &mut VecDeque<(Instant, MemoryTemperature)>,
    timestamp: Instant,
    window: Duration,
    sample: MemoryTemperature,
) -> MemoryTemperature {
    while samples
        .front()
        .is_some_and(|(previous, _)| timestamp.saturating_duration_since(*previous) > window)
    {
        samples.pop_front();
    }
    samples.push_back((timestamp, sample));
    MemoryTemperature {
        celsius: samples.iter().map(|(_, value)| value.celsius).sum::<f32>() / samples.len() as f32,
        lower_bound: samples.iter().any(|(_, value)| value.lower_bound),
    }
}

impl MemoryTemperatureWorker {
    /// Starts polling the real hwmon hotspot sensor.
    fn start(interval: Duration, smoothing_window: Duration) -> Self {
        Self::start_with_reader(interval, smoothing_window, || {
            crate::layer::memory_temperature::read_hotspot()
        })
    }

    /// Starts the polling thread with a custom reader, smoothing readings over the given window.
    fn start_with_reader(
        interval: Duration,
        smoothing_window: Duration,
        mut reader: impl FnMut() -> anyhow::Result<crate::layer::memory_temperature::Hotspot>
            + Send
            + 'static,
    ) -> Self {
        let latest = Arc::new(parking_lot::RwLock::new(None));
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_latest = Arc::clone(&latest);
        let worker_stop = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let mut samples: VecDeque<(Instant, MemoryTemperature)> = VecDeque::new();
            loop {
                if *worker_stop.0.lock() {
                    return;
                }
                match reader() {
                    Ok(hotspot) => {
                        let now = Instant::now();
                        let smoothed = smooth_memory_temperature(
                            &mut samples,
                            now,
                            smoothing_window,
                            MemoryTemperature {
                                celsius: hotspot.temperature_c as f32,
                                lower_bound: hotspot.saturated,
                            },
                        );
                        *worker_latest.write() = Some((smoothed, now));
                    }
                    Err(error) => {
                        *worker_latest.write() = None;
                        diagnostics::warn(format!(
                            "GDDR6 temperature polling stopped after error: {error:#}; restart the application to retry"
                        ));
                        return;
                    }
                }
                let (lock, wake) = &*worker_stop;
                let mut stopped = lock.lock();
                let started = Instant::now();
                while !*stopped && started.elapsed() < interval {
                    let remaining = interval.saturating_sub(started.elapsed());
                    wake.wait_for(&mut stopped, remaining.min(Duration::from_secs(1)));
                }
                if *stopped {
                    return;
                }
            }
        });
        Self {
            latest,
            stop,
            worker: Some(worker),
            interval,
            stale_reported: AtomicBool::new(false),
        }
    }

    /// Returns the latest smoothed temperature, or `None` if unavailable or older than two intervals.
    fn current(&self) -> Option<MemoryTemperature> {
        let latest = *self.latest.read();
        match latest {
            Some((_, timestamp)) if timestamp.elapsed() > self.interval.saturating_mul(2) => {
                if !self.stale_reported.swap(true, Ordering::Relaxed) {
                    diagnostics::warn("GDDR6 temperature sample is stale");
                }
                None
            }
            Some((value, _)) => {
                self.stale_reported.store(false, Ordering::Relaxed);
                Some(value)
            }
            None => None,
        }
    }

    /// Signals the polling thread to exit and joins it.
    fn stop(&mut self) {
        let (lock, wake) = &*self.stop;
        *lock.lock() = true;
        wake.notify_all();
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                diagnostics::error("GDDR6 temperature polling thread panicked");
            }
        }
    }
}

/// Latest telemetry: smoothed values for display and raw values for auto regulation.
#[derive(Clone)]
struct TelemetrySample {
    gpu: GpuMeasurement,
    hardware: HardwareMetrics,
    hardware_timestamp: Instant,
    raw_gpu: GpuMeasurement,
    raw_hardware: HardwareMetrics,
}

impl TelemetryWorker {
    /// Starts polling and smoothing GPU usage on a background thread.
    fn start(
        provider: AmdGpuProvider,
        hwmon_provider: HwmonMetricsProvider,
        poll_rate: Duration,
        smoothing_window: Duration,
        memory_poll_rate_ms: u64,
        bc250: bool,
    ) -> Self {
        let initial = provider.get_metric();
        let mut smoother = GpuUsageSmoother::new(smoothing_window);
        let mut hardware_smoother = HardwareMetricsSmoother::new(smoothing_window);
        let mut cpu_usage = CpuUsageProvider::new();
        let mut memory_usage = MemoryUsageProvider::new(&provider);
        let mut initial_hardware = hwmon_provider.get_metrics();
        initial_hardware.cpu_load_percent = cpu_usage.get_load_percent();
        memory_usage.update(&mut initial_hardware);
        let initial_hardware_timestamp = Instant::now();
        let smoothed_hardware =
            hardware_smoother.add(initial_hardware.clone(), initial_hardware_timestamp);
        let latest = Arc::new(parking_lot::RwLock::new(TelemetrySample {
            gpu: smoother.add(initial.clone()),
            hardware: smoothed_hardware,
            hardware_timestamp: initial_hardware_timestamp,
            raw_gpu: initial,
            raw_hardware: initial_hardware,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_latest = Arc::clone(&latest);
        let worker_stop = Arc::clone(&stop);
        let poll_rate = poll_rate.clamp(Duration::from_millis(10), Duration::from_secs(1));
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                thread::sleep(poll_rate);
                if worker_stop.load(Ordering::Relaxed) {
                    break;
                }
                let mut hardware = hwmon_provider.get_metrics();
                hardware.cpu_load_percent = cpu_usage.get_load_percent();
                memory_usage.update(&mut hardware);
                let hardware_timestamp = Instant::now();
                let smoothed_hardware = hardware_smoother.add(hardware.clone(), hardware_timestamp);
                let gpu = provider.get_metric();
                *worker_latest.write() = TelemetrySample {
                    gpu: smoother.add(gpu.clone()),
                    hardware: smoothed_hardware,
                    hardware_timestamp,
                    raw_gpu: gpu,
                    raw_hardware: hardware,
                };
            }
        });
        Self {
            latest,
            stop,
            worker: Some(worker),
            memory: if memory_poll_rate_ms >= 2000 && bc250 {
                Some(MemoryTemperatureWorker::start(
                    Duration::from_millis(memory_poll_rate_ms),
                    smoothing_window,
                ))
            } else {
                if memory_poll_rate_ms >= 2000 && !bc250 {
                    diagnostics::info(
                        "GDDR6 temperature polling unavailable: selected GPU is not a BC-250",
                    );
                }
                None
            },
        }
    }

    /// Returns the latest GPU sample, marking it expired after the configured age.
    fn current(&self, expiry: Duration) -> TelemetrySample {
        let mut sample = self.latest.read().clone();
        if sample.gpu.status == TelemetryStatus::Available
            && sample.gpu.timestamp.elapsed() > expiry
        {
            sample.gpu.status = TelemetryStatus::Expired;
            sample.gpu.value = 0.0;
        }
        sample.hardware.memory_temp = self
            .memory
            .as_ref()
            .and_then(MemoryTemperatureWorker::current);
        sample
    }

    /// Stops and joins the GPU polling thread.
    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if let Some(memory) = self.memory.as_mut() {
            memory.stop();
        }
    }
}

/// Returns the global map of Vulkan instance dispatch data.
fn instances() -> &'static Mutex<HashMap<usize, InstanceDispatch>> {
    static INSTANCES: OnceLock<Mutex<HashMap<usize, InstanceDispatch>>> = OnceLock::new();
    INSTANCES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Returns the map from enumerated physical devices to their parent instances.
fn physical_device_instances() -> &'static Mutex<HashMap<usize, usize>> {
    static PHYSICAL_DEVICE_INSTANCES: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    PHYSICAL_DEVICE_INSTANCES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Returns the global map of tracked Vulkan devices.
fn devices() -> &'static Mutex<HashMap<usize, Arc<Mutex<DeviceState>>>> {
    static DEVICES: OnceLock<Mutex<HashMap<usize, Arc<Mutex<DeviceState>>>>> = OnceLock::new();
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Converts a Vulkan handle to its raw integer lookup key.
fn key<T: Handle>(handle: T) -> usize {
    handle.as_raw() as usize
}

/// Converts a queue handle to its device-state lookup key.
fn queue_key(queue: vk::Queue) -> usize {
    queue.as_raw() as usize
}

/// Finds the tracked device state that owns a queue.
fn device_for_queue(queue: vk::Queue) -> Option<Arc<Mutex<DeviceState>>> {
    let queue_dispatch_key = queue_key(queue);
    devices().lock().values().find_map(|state| {
        let contains_queue = state
            .lock()
            .queue_families
            .contains_key(&queue_dispatch_key);
        contains_queue.then(|| state.clone())
    })
}

static TRACE_DISPATCH_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Thread-local cache of trace dispatch tables, invalidated when the global epoch changes.
#[derive(Default)]
struct TraceDispatchCache {
    epoch: u64,
    devices: HashMap<usize, TraceDispatch>,
    queues: HashMap<usize, TraceDispatch>,
}

thread_local! {
    static TRACE_DISPATCH_CACHE: RefCell<TraceDispatchCache> = RefCell::new(TraceDispatchCache::default());
}

/// Looks up the trace dispatch under the global locks, keyed by device or by queue.
fn lookup_trace_dispatch(key: usize, queue: bool) -> Option<TraceDispatch> {
    if queue {
        device_for_queue(vk::Queue::from_raw(key as u64)).map(|state| state.lock().trace_dispatch)
    } else {
        devices()
            .lock()
            .get(&key)
            .map(|state| state.lock().trace_dispatch)
    }
}

/// Returns the trace dispatch from the thread-local cache, falling back to a locked lookup on a miss.
fn cached_trace_dispatch(key: usize, queue: bool) -> Option<TraceDispatch> {
    TRACE_DISPATCH_CACHE
        .try_with(|cache| {
            let mut cache = cache.borrow_mut();
            let epoch = TRACE_DISPATCH_EPOCH.load(Ordering::Acquire);
            if cache.epoch != epoch {
                cache.epoch = epoch;
                cache.devices.clear();
                cache.queues.clear();
            }
            let entries = if queue {
                &mut cache.queues
            } else {
                &mut cache.devices
            };
            if let Some(dispatch) = entries.get(&key) {
                return Some(*dispatch);
            }
            let dispatch = lookup_trace_dispatch(key, queue)?;
            entries.insert(key, dispatch);
            Some(dispatch)
        })
        .unwrap_or_else(|_| lookup_trace_dispatch(key, queue))
}

/// Returns the dispatch-table key for a Vulkan handle.
unsafe fn dispatch_key<T: Handle>(handle: T) -> usize {
    key(handle)
}

/// Converts a non-null Vulkan procedure name to a C string reference.
unsafe fn proc_name(name: *const c_char) -> Option<&'static CStr> {
    (!name.is_null()).then(|| CStr::from_ptr(name))
}

/// Returns whether a C array of extension names contains the given extension.
unsafe fn has_extension(count: u32, names: *const *const c_char, extension: &CStr) -> bool {
    if count == 0 || names.is_null() {
        return false;
    }
    std::slice::from_raw_parts(names, count as usize)
        .iter()
        .any(|name| !name.is_null() && CStr::from_ptr(*name) == extension)
}

/// Checks that the device exposes the calibrated-timestamps extension with device and monotonic-clock domains.
unsafe fn calibration_supported(
    instance: InstanceDispatch,
    physical_device: vk::PhysicalDevice,
    extension: &CStr,
    domain_command: &CStr,
) -> bool {
    let Some(enumerate) = (instance.next_gipa)(
        instance.handle,
        c"vkEnumerateDeviceExtensionProperties".as_ptr(),
    ) else {
        diagnostics::warn("GPU calibration: device extension enumeration unavailable");
        return false;
    };
    let enumerate: vk::PFN_vkEnumerateDeviceExtensionProperties = mem::transmute(enumerate);
    let mut count = 0;
    let result = enumerate(
        physical_device,
        std::ptr::null(),
        &mut count,
        std::ptr::null_mut(),
    );
    if result != vk::Result::SUCCESS {
        diagnostics::warn(format!(
            "GPU calibration: extension enumeration failed: {result:?}"
        ));
        return false;
    }
    let mut properties = vec![vk::ExtensionProperties::default(); count as usize];
    let result = enumerate(
        physical_device,
        std::ptr::null(),
        &mut count,
        properties.as_mut_ptr(),
    );
    if result != vk::Result::SUCCESS {
        diagnostics::warn(format!(
            "GPU calibration: extension enumeration failed: {result:?}"
        ));
        return false;
    }
    properties.truncate(count as usize);
    if !properties
        .iter()
        .any(|property| CStr::from_ptr(property.extension_name.as_ptr()) == extension)
    {
        return false;
    }
    let Some(domains) = (instance.next_gipa)(instance.handle, domain_command.as_ptr()) else {
        diagnostics::warn("GPU calibration: time-domain query unavailable");
        return false;
    };
    let domains: vk::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = mem::transmute(domains);
    let mut count = 0;
    let result = domains(physical_device, &mut count, std::ptr::null_mut());
    if result != vk::Result::SUCCESS || count == 0 {
        diagnostics::warn(format!(
            "GPU calibration: time-domain query failed: {result:?}"
        ));
        return false;
    }
    let mut values = vec![vk::TimeDomainKHR::default(); count as usize];
    let result = domains(physical_device, &mut count, values.as_mut_ptr());
    if result != vk::Result::SUCCESS {
        diagnostics::warn(format!(
            "GPU calibration: time-domain query failed: {result:?}"
        ));
        return false;
    }
    values.truncate(count as usize);
    values.contains(&vk::TimeDomainKHR::DEVICE)
        && values.contains(&vk::TimeDomainKHR::CLOCK_MONOTONIC)
}

/// Wraps a layer function pointer in Vulkan's optional function-pointer type.
unsafe fn proc<T: Copy>(function: T) -> vk::PFN_vkVoidFunction {
    Some(mem::transmute_copy(&function))
}

/// Returns the layer's tracing hook for a submit/wait entry point name, if intercepted.
unsafe fn trace_proc(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkQueueSubmit" => proc(vkQueueSubmit as vk::PFN_vkQueueSubmit),
        b"vkQueueSubmit2" => proc(vkQueueSubmit2 as vk::PFN_vkQueueSubmit2),
        b"vkQueueSubmit2KHR" => proc(vkQueueSubmit2KHR as vk::PFN_vkQueueSubmit2),
        b"vkWaitForFences" => proc(vkWaitForFences as vk::PFN_vkWaitForFences),
        b"vkWaitSemaphores" => proc(vkWaitSemaphores as vk::PFN_vkWaitSemaphores),
        b"vkWaitSemaphoresKHR" => proc(vkWaitSemaphoresKHR as vk::PFN_vkWaitSemaphores),
        b"vkQueueWaitIdle" => proc(vkQueueWaitIdle as vk::PFN_vkQueueWaitIdle),
        b"vkDeviceWaitIdle" => proc(vkDeviceWaitIdle as vk::PFN_vkDeviceWaitIdle),
        _ => None,
    }
}

/// Resolves the next layer's instance proc-address function.
unsafe fn next_instance_gipa(instance: vk::Instance) -> Option<vk::PFN_vkGetInstanceProcAddr> {
    if instance.is_null() {
        return instances()
            .lock()
            .values()
            .next()
            .map(|state| state.next_gipa);
    }
    instances()
        .lock()
        .get(&key(instance))
        .map(|state| state.next_gipa)
}

/// Finds the instance dispatch data associated with a physical device.
fn instance_dispatch_for_physical_device(
    physical_device: vk::PhysicalDevice,
) -> Option<InstanceDispatch> {
    let physical_device_instances = physical_device_instances().lock();
    let tracked_instances = instances().lock();
    find_instance_dispatch(
        key(physical_device),
        &physical_device_instances,
        &tracked_instances,
    )
}

/// Selects a mapped parent instance or the safe single-instance fallback.
fn find_instance_dispatch(
    physical_device_key: usize,
    physical_device_instances: &HashMap<usize, usize>,
    instances: &HashMap<usize, InstanceDispatch>,
) -> Option<InstanceDispatch> {
    if let Some(instance_key) = physical_device_instances.get(&physical_device_key) {
        return instances.get(instance_key).copied();
    }
    if instances.len() == 1 {
        return instances.values().next().copied();
    }
    None
}

/// Records the parent instance for an enumerated physical device.
fn track_physical_device(instance: vk::Instance, physical_device: vk::PhysicalDevice) {
    physical_device_instances()
        .lock()
        .insert(key(physical_device), key(instance));
}

/// Calls downstream enumeration and records returned physical-device handles.
unsafe fn enumerate_physical_devices(
    instance: vk::Instance,
    count: *mut u32,
    physical_devices: *mut vk::PhysicalDevice,
    enumerate: vk::PFN_vkEnumeratePhysicalDevices,
) -> vk::Result {
    let result = enumerate(instance, count, physical_devices);
    if matches!(result, vk::Result::SUCCESS | vk::Result::INCOMPLETE)
        && !count.is_null()
        && !physical_devices.is_null()
    {
        let devices = std::slice::from_raw_parts(physical_devices, *count as usize);
        for &physical_device in devices {
            track_physical_device(instance, physical_device);
        }
    }
    result
}

/// Calls downstream group enumeration and records every returned device handle.
unsafe fn enumerate_physical_device_groups(
    instance: vk::Instance,
    count: *mut u32,
    groups: *mut vk::PhysicalDeviceGroupProperties<'_>,
    enumerate: vk::PFN_vkEnumeratePhysicalDeviceGroups,
) -> vk::Result {
    let result = enumerate(instance, count, groups);
    if matches!(result, vk::Result::SUCCESS | vk::Result::INCOMPLETE)
        && !count.is_null()
        && !groups.is_null()
    {
        let groups = std::slice::from_raw_parts(groups, *count as usize);
        for group in groups {
            for &physical_device in group
                .physical_devices
                .iter()
                .take(group.physical_device_count as usize)
            {
                track_physical_device(instance, physical_device);
            }
        }
    }
    result
}

#[no_mangle]
/// Negotiates the Vulkan loader interface version and layer entry points.
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    interface: *mut NegotiateLayerInterface,
) -> vk::Result {
    diagnostics::initialize();
    if interface.is_null() {
        diagnostics::error("vkNegotiateLoaderLayerInterfaceVersion received a null interface");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let requested_version = (*interface).loader_layer_interface_version;
    (*interface).loader_layer_interface_version =
        (*interface).loader_layer_interface_version.min(2);
    (*interface).get_instance_proc_addr = vkGetInstanceProcAddr;
    (*interface).get_device_proc_addr = vkGetDeviceProcAddr;
    (*interface).get_physical_device_proc_addr = None;
    diagnostics::info(format!(
        "loader interface negotiated: requested_version={requested_version} selected_version={}",
        (*interface).loader_layer_interface_version
    ));
    vk::Result::SUCCESS
}

#[no_mangle]
/// Resolves Vulkan instance-level functions, intercepting Framecap entry points.
pub unsafe extern "system" fn vkGetInstanceProcAddr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let Some(name) = proc_name(name) else {
        return None;
    };
    if let Some(interceptor) = trace_proc(name) {
        return next_instance_gipa(instance)
            .and_then(|next| next(instance, name.as_ptr()))
            .map(|_| interceptor);
    }
    let device_interceptor = match name.to_bytes() {
        b"vkGetDeviceQueue" => proc(vkGetDeviceQueue as vk::PFN_vkGetDeviceQueue),
        b"vkGetDeviceQueue2" => proc(vkGetDeviceQueue2 as vk::PFN_vkGetDeviceQueue2),
        b"vkQueuePresentKHR" => proc(vkQueuePresentKHR as vk::PFN_vkQueuePresentKHR),
        b"vkAcquireNextImageKHR" => proc(vkAcquireNextImageKHR as vk::PFN_vkAcquireNextImageKHR),
        b"vkAcquireNextImage2KHR" => proc(vkAcquireNextImage2KHR as vk::PFN_vkAcquireNextImage2KHR),
        _ => None,
    };
    if let Some(interceptor) = device_interceptor {
        return next_instance_gipa(instance)
            .and_then(|next| next(instance, name.as_ptr()))
            .map(|_| interceptor);
    }
    match name.to_bytes() {
        b"vkGetInstanceProcAddr" => proc(vkGetInstanceProcAddr as vk::PFN_vkGetInstanceProcAddr),
        b"vkGetDeviceProcAddr" => proc(vkGetDeviceProcAddr as vk::PFN_vkGetDeviceProcAddr),
        b"vkCreateInstance" => proc(vkCreateInstance as vk::PFN_vkCreateInstance),
        b"vkDestroyInstance" => proc(vkDestroyInstance as vk::PFN_vkDestroyInstance),
        b"vkEnumeratePhysicalDevices" => {
            proc(vkEnumeratePhysicalDevices as vk::PFN_vkEnumeratePhysicalDevices)
        }
        b"vkEnumeratePhysicalDeviceGroups" => {
            proc(vkEnumeratePhysicalDeviceGroups as vk::PFN_vkEnumeratePhysicalDeviceGroups)
        }
        b"vkEnumeratePhysicalDeviceGroupsKHR" => {
            proc(vkEnumeratePhysicalDeviceGroupsKHR as vk::PFN_vkEnumeratePhysicalDeviceGroups)
        }
        b"vkCreateDevice" => proc(vkCreateDevice as vk::PFN_vkCreateDevice),
        _ => next_instance_gipa(instance).and_then(|next| next(instance, name.as_ptr())),
    }
}

#[no_mangle]
/// Enumerates physical devices and records their owning instance.
pub unsafe extern "system" fn vkEnumeratePhysicalDevices(
    instance: vk::Instance,
    count: *mut u32,
    physical_devices: *mut vk::PhysicalDevice,
) -> vk::Result {
    let Some(next_gipa) = next_instance_gipa(instance) else {
        diagnostics::error("vkEnumeratePhysicalDevices: instance is not tracked by Framecap");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(enumerate) = next_gipa(instance, c"vkEnumeratePhysicalDevices".as_ptr()) else {
        diagnostics::error("vkEnumeratePhysicalDevices: downstream function was not found");
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let enumerate: vk::PFN_vkEnumeratePhysicalDevices = mem::transmute(enumerate);
    let result = enumerate_physical_devices(instance, count, physical_devices, enumerate);
    match (result, count.is_null()) {
        (vk::Result::SUCCESS | vk::Result::INCOMPLETE, false) => diagnostics::info(format!(
            "vkEnumeratePhysicalDevices completed: instance=0x{:x} result={result:?} count={} output_provided={}",
            key(instance),
            *count,
            !physical_devices.is_null()
        )),
        _ => diagnostics::warn(format!(
            "vkEnumeratePhysicalDevices returned: instance=0x{:x} result={result:?}",
            key(instance)
        )),
    }
    result
}

#[no_mangle]
/// Enumerates physical-device groups and records each group's device ownership.
pub unsafe extern "system" fn vkEnumeratePhysicalDeviceGroups(
    instance: vk::Instance,
    count: *mut u32,
    groups: *mut vk::PhysicalDeviceGroupProperties<'_>,
) -> vk::Result {
    let Some(next_gipa) = next_instance_gipa(instance) else {
        diagnostics::error("vkEnumeratePhysicalDeviceGroups: instance is not tracked by Framecap");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(enumerate) = next_gipa(instance, c"vkEnumeratePhysicalDeviceGroups".as_ptr()) else {
        diagnostics::error("vkEnumeratePhysicalDeviceGroups: downstream function was not found");
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let enumerate: vk::PFN_vkEnumeratePhysicalDeviceGroups = mem::transmute(enumerate);
    let result = enumerate_physical_device_groups(instance, count, groups, enumerate);
    match (result, count.is_null()) {
        (vk::Result::SUCCESS | vk::Result::INCOMPLETE, false) => diagnostics::info(format!(
            "vkEnumeratePhysicalDeviceGroups completed: instance=0x{:x} result={result:?} groups={} output_provided={}",
            key(instance),
            *count,
            !groups.is_null()
        )),
        _ => diagnostics::warn(format!(
            "vkEnumeratePhysicalDeviceGroups returned: instance=0x{:x} result={result:?}",
            key(instance)
        )),
    }
    result
}

#[no_mangle]
/// KHR alias for physical-device-group enumeration.
pub unsafe extern "system" fn vkEnumeratePhysicalDeviceGroupsKHR(
    instance: vk::Instance,
    count: *mut u32,
    groups: *mut vk::PhysicalDeviceGroupProperties<'_>,
) -> vk::Result {
    let Some(next_gipa) = next_instance_gipa(instance) else {
        diagnostics::error(
            "vkEnumeratePhysicalDeviceGroupsKHR: instance is not tracked by Framecap",
        );
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(enumerate) = next_gipa(instance, c"vkEnumeratePhysicalDeviceGroupsKHR".as_ptr())
    else {
        diagnostics::error("vkEnumeratePhysicalDeviceGroupsKHR: downstream function was not found");
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let enumerate: vk::PFN_vkEnumeratePhysicalDeviceGroups = mem::transmute(enumerate);
    let result = enumerate_physical_device_groups(instance, count, groups, enumerate);
    match (result, count.is_null()) {
        (vk::Result::SUCCESS | vk::Result::INCOMPLETE, false) => diagnostics::info(format!(
            "vkEnumeratePhysicalDeviceGroupsKHR completed: instance=0x{:x} result={result:?} groups={} output_provided={}",
            key(instance),
            *count,
            !groups.is_null()
        )),
        _ => diagnostics::warn(format!(
            "vkEnumeratePhysicalDeviceGroupsKHR returned: instance=0x{:x} result={result:?}",
            key(instance)
        )),
    }
    result
}

#[no_mangle]
/// Resolves Vulkan device-level functions, intercepting Framecap entry points.
pub unsafe extern "system" fn vkGetDeviceProcAddr(
    device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let Some(name) = proc_name(name) else {
        return None;
    };
    if let Some(interceptor) = trace_proc(name) {
        return devices()
            .lock()
            .get(&dispatch_key(device))
            .and_then(|state| (state.lock().next_gdpa)(device, name.as_ptr()))
            .map(|_| interceptor);
    }
    match name.to_bytes() {
        b"vkGetDeviceProcAddr" => proc(vkGetDeviceProcAddr as vk::PFN_vkGetDeviceProcAddr),
        b"vkDestroyDevice" => proc(vkDestroyDevice as vk::PFN_vkDestroyDevice),
        b"vkQueuePresentKHR" => proc(vkQueuePresentKHR as vk::PFN_vkQueuePresentKHR),
        b"vkAcquireNextImageKHR" => devices()
            .lock()
            .get(&dispatch_key(device))
            .and_then(|state| state.lock().next_acquire)
            .and_then(|_| proc(vkAcquireNextImageKHR as vk::PFN_vkAcquireNextImageKHR)),
        b"vkAcquireNextImage2KHR" => devices()
            .lock()
            .get(&dispatch_key(device))
            .and_then(|state| state.lock().next_acquire2)
            .and_then(|_| proc(vkAcquireNextImage2KHR as vk::PFN_vkAcquireNextImage2KHR)),
        b"vkCreateSwapchainKHR" => proc(vkCreateSwapchainKHR as vk::PFN_vkCreateSwapchainKHR),
        b"vkDestroySwapchainKHR" => proc(vkDestroySwapchainKHR as vk::PFN_vkDestroySwapchainKHR),
        b"vkGetDeviceQueue" => proc(vkGetDeviceQueue as vk::PFN_vkGetDeviceQueue),
        b"vkGetDeviceQueue2" => proc(vkGetDeviceQueue2 as vk::PFN_vkGetDeviceQueue2),
        _ => devices()
            .lock()
            .get(&dispatch_key(device))
            .and_then(|state| {
                let next_gdpa = state.lock().next_gdpa;
                next_gdpa(device, name.as_ptr())
            }),
    }
}

#[no_mangle]
/// Creates and tracks a Vulkan instance while forwarding to the next layer.
pub unsafe extern "system" fn vkCreateInstance(
    create_info: *const vk::InstanceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    instance: *mut vk::Instance,
) -> vk::Result {
    if create_info.is_null() || instance.is_null() {
        diagnostics::error("vkCreateInstance received a null create_info or instance pointer");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let mut chain = (*create_info).p_next;
    let mut next_gipa = None;
    let mut link_found = false;
    while !chain.is_null() {
        let header = chain.cast::<LayerInstanceCreateInfo>();
        if (*header).s_type == LOADER_INSTANCE_CREATE_INFO && (*header).function == LAYER_LINK_INFO
        {
            link_found = true;
            let link = (*header).u.layer_info;
            if link.is_null() {
                diagnostics::error("vkCreateInstance: loader instance link is null");
                return vk::Result::ERROR_INITIALIZATION_FAILED;
            }
            next_gipa = Some((*link).next_get_instance_proc_addr);
            (*header.cast_mut()).u.layer_info = (*link).next;
            break;
        }
        chain = (*(chain.cast::<vk::BaseInStructure<'_>>())).p_next.cast();
    }
    let Some(next_gipa) = next_gipa else {
        diagnostics::error(format!(
            "vkCreateInstance: loader instance link not found (found={link_found})"
        ));
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(create_instance) = next_gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr())
    else {
        diagnostics::error("vkCreateInstance: downstream vkCreateInstance was not found");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let application = (*create_info).p_application_info;
    let application_name = if application.is_null() || (*application).p_application_name.is_null() {
        "<unspecified>".to_owned()
    } else {
        CStr::from_ptr((*application).p_application_name)
            .to_string_lossy()
            .into_owned()
    };
    let engine_name = if application.is_null() || (*application).p_engine_name.is_null() {
        "<unspecified>".to_owned()
    } else {
        CStr::from_ptr((*application).p_engine_name)
            .to_string_lossy()
            .into_owned()
    };
    diagnostics::info(format!(
        "vkCreateInstance requested: application={application_name:?} engine={engine_name:?} api_version={} enabled_extensions={} enabled_layers={}",
        if application.is_null() {
            "unspecified".to_owned()
        } else {
            format!(
                "{}.{}.{}",
                vk::api_version_major((*application).api_version),
                vk::api_version_minor((*application).api_version),
                vk::api_version_patch((*application).api_version)
            )
        },
        (*create_info).enabled_extension_count,
        (*create_info).enabled_layer_count,
    ));
    let create_instance: vk::PFN_vkCreateInstance = mem::transmute(create_instance);
    let start = Instant::now();
    let result = create_instance(create_info, allocator, instance);
    let elapsed = start.elapsed();
    if result != vk::Result::SUCCESS {
        diagnostics::error(format!(
            "downstream vkCreateInstance failed: result={result:?} elapsed_ms={:.3}",
            elapsed.as_secs_f64() * 1000.0
        ));
    }
    if result == vk::Result::SUCCESS {
        let api_version = if application.is_null() || (*application).api_version == 0 {
            vk::API_VERSION_1_0
        } else {
            (*application).api_version
        };
        let calibration_core = vk::api_version_major(api_version) > 1
            || (vk::api_version_major(api_version) == 1 && vk::api_version_minor(api_version) >= 1);
        let calibration_properties2 = has_extension(
            (*create_info).enabled_extension_count,
            (*create_info).pp_enabled_extension_names,
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME,
        );
        instances().lock().insert(
            key(*instance),
            InstanceDispatch {
                handle: *instance,
                next_gipa,
                calibration_core,
                calibration_properties2,
            },
        );
        diagnostics::info(format!(
            "vkCreateInstance succeeded: elapsed_ms={:.3}; Framecap Vulkan layer loaded",
            elapsed.as_secs_f64() * 1000.0
        ));
    }
    result
}

#[no_mangle]
/// Removes instance and physical-device tracking before forwarding destruction.
pub unsafe extern "system" fn vkDestroyInstance(
    instance: vk::Instance,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    let instance_key = key(instance);
    physical_device_instances()
        .lock()
        .retain(|_, parent_instance| *parent_instance != instance_key);
    if let Some(state) = instances().lock().remove(&instance_key) {
        if let Some(destroy) = (state.next_gipa)(instance, c"vkDestroyInstance".as_ptr()) {
            let destroy: vk::PFN_vkDestroyInstance = mem::transmute(destroy);
            destroy(instance, allocator);
        }
    }
}

#[no_mangle]
/// Creates a Vulkan device using the instance associated with its physical device.
pub unsafe extern "system" fn vkCreateDevice(
    physical_device: vk::PhysicalDevice,
    create_info: *const vk::DeviceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    device: *mut vk::Device,
) -> vk::Result {
    if create_info.is_null() || device.is_null() {
        diagnostics::error("vkCreateDevice received a null create_info or device pointer");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let mut device_chain = (*create_info).p_next;
    let mut multi_device = false;
    while !device_chain.is_null() {
        let header = &*device_chain.cast::<vk::BaseInStructure<'_>>();
        if header.s_type == vk::StructureType::DEVICE_GROUP_DEVICE_CREATE_INFO {
            let group = &*device_chain.cast::<vk::DeviceGroupDeviceCreateInfo<'_>>();
            multi_device = group.physical_device_count > 1;
        }
        device_chain = header.p_next.cast();
    }
    let mut chain = (*create_info).p_next;
    let mut next_gdpa = None;
    while !chain.is_null() {
        let header = chain.cast::<LayerDeviceCreateInfo>();
        if (*header).s_type == LOADER_DEVICE_CREATE_INFO && (*header).function == LAYER_LINK_INFO {
            let link = (*header).u.layer_info;
            if link.is_null() {
                diagnostics::error("vkCreateDevice: loader device link is null");
                return vk::Result::ERROR_INITIALIZATION_FAILED;
            }
            next_gdpa = Some((*link).next_get_device_proc_addr);
            (*header.cast_mut()).u.layer_info = (*link).next;
            break;
        }
        chain = (*(chain.cast::<vk::BaseInStructure<'_>>())).p_next.cast();
    }
    let Some(next_gdpa) = next_gdpa else {
        diagnostics::error("vkCreateDevice: loader device link was not found");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(instance_dispatch) = instance_dispatch_for_physical_device(physical_device) else {
        diagnostics::error(format!(
            "vkCreateDevice: no tracked instance found for physical device 0x{:x}",
            key(physical_device)
        ));
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(create_device) =
        (instance_dispatch.next_gipa)(instance_dispatch.handle, c"vkCreateDevice".as_ptr())
    else {
        diagnostics::error("vkCreateDevice: downstream vkCreateDevice was not found");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_device: vk::PFN_vkCreateDevice = mem::transmute(create_device);
    let mut physical_properties = vk::PhysicalDeviceProperties::default();
    if let Some(get_properties) = (instance_dispatch.next_gipa)(
        instance_dispatch.handle,
        c"vkGetPhysicalDeviceProperties".as_ptr(),
    ) {
        let get_properties: vk::PFN_vkGetPhysicalDeviceProperties = mem::transmute(get_properties);
        get_properties(physical_device, &mut physical_properties);
    }
    let physical_api_version = physical_properties.api_version;
    let physical_core = vk::api_version_major(physical_api_version) > 1
        || (vk::api_version_major(physical_api_version) == 1
            && vk::api_version_minor(physical_api_version) >= 1);
    let names = (*create_info).pp_enabled_extension_names;
    let count = (*create_info).enabled_extension_count;
    let khr_enabled = has_extension(count, names, vk::KHR_CALIBRATED_TIMESTAMPS_NAME);
    let ext_enabled = has_extension(count, names, vk::EXT_CALIBRATED_TIMESTAMPS_NAME);
    let khr = (
        vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
        c"vkGetPhysicalDeviceCalibrateableTimeDomainsKHR",
        c"vkGetCalibratedTimestampsKHR",
    );
    let ext = (
        vk::EXT_CALIBRATED_TIMESTAMPS_NAME,
        c"vkGetPhysicalDeviceCalibrateableTimeDomainsEXT",
        c"vkGetCalibratedTimestampsEXT",
    );
    let both = [ext, khr];
    let candidates: &[_] = if khr_enabled {
        std::slice::from_ref(&khr)
    } else if ext_enabled {
        std::slice::from_ref(&ext)
    } else {
        &both
    };
    let calibration = if !multi_device
        && (instance_dispatch.calibration_properties2
            || (instance_dispatch.calibration_core && physical_core))
        && (count == 0 || !names.is_null())
    {
        candidates.iter().copied().find(|(name, domains, _)| {
            calibration_supported(instance_dispatch, physical_device, name, domains)
        })
    } else {
        None
    };
    let mut extension_names = Vec::new();
    let mut augmented_info = *create_info;
    let add_calibration = calibration.is_some() && !khr_enabled && !ext_enabled;
    if let Some((calibration_name, _, _)) = calibration.filter(|_| add_calibration) {
        if count > 0 {
            extension_names.extend_from_slice(std::slice::from_raw_parts(names, count as usize));
        }
        extension_names.push(calibration_name.as_ptr());
        augmented_info.enabled_extension_count = extension_names.len() as u32;
        augmented_info.pp_enabled_extension_names = extension_names.as_ptr();
    }
    let start = Instant::now();
    let result = create_device(
        physical_device,
        if add_calibration {
            &augmented_info
        } else {
            create_info
        },
        allocator,
        device,
    );
    let elapsed = start.elapsed();
    if result != vk::Result::SUCCESS {
        diagnostics::error(format!(
            "downstream vkCreateDevice failed: result={result:?} elapsed_ms={:.3}",
            elapsed.as_secs_f64() * 1000.0
        ));
    }
    if result == vk::Result::SUCCESS {
        let mut queue_properties = Vec::new();
        if let Some(get_queues) = (instance_dispatch.next_gipa)(
            instance_dispatch.handle,
            c"vkGetPhysicalDeviceQueueFamilyProperties".as_ptr(),
        ) {
            let get_queues: vk::PFN_vkGetPhysicalDeviceQueueFamilyProperties =
                mem::transmute(get_queues);
            let mut count = 0;
            get_queues(physical_device, &mut count, std::ptr::null_mut());
            queue_properties.resize(count as usize, vk::QueueFamilyProperties::default());
            get_queues(physical_device, &mut count, queue_properties.as_mut_ptr());
            queue_properties.truncate(count as usize);
        }
        let config = Config::load();
        let device_name = CStr::from_ptr(physical_properties.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        diagnostics::info(format!(
            "vkCreateDevice succeeded: elapsed_ms={:.3} instance=0x{:x} physical_device=0x{:x} device={device_name:?} vendor_id=0x{:04x} device_id=0x{:04x} api_version={}.{}.{}",
            elapsed.as_secs_f64() * 1000.0,
            key(instance_dispatch.handle),
            key(physical_device),
            physical_properties.vendor_id,
            physical_properties.device_id,
            vk::api_version_major(physical_properties.api_version),
            vk::api_version_minor(physical_properties.api_version),
            vk::api_version_patch(physical_properties.api_version),
        ));
        let auto_algorithm_summary = config
            .auto_algorithms
            .iter()
            .map(|algorithm| {
                format!(
                    "{}(metric={:?},target={:?},hysteresis={} {},window={}ms)",
                    algorithm.name,
                    algorithm.metric,
                    algorithm.target,
                    algorithm.hysteresis,
                    algorithm.metric.hysteresis_unit(),
                    algorithm.window.as_millis()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        diagnostics::info(format!(
            "configuration: startup_fps={} startup_mode={:?} input_enabled={} hotkeys=auto:{:?},down:{:?},up:{:?},toggle:{:?} overlay_startup={} overlay_renderer={:?} auto_algorithms=[{}] gpu_increase_limit={} gpu_decrease_limit={} frame_jitter_limit_ms={} manual_step_fps={} manual_bounds={}..{} gpu_poll_rate_ms={} memory_poll_rate_ms={} overlay_refresh_rate={} gpu_expiry_ms={} gpu_smoothing_window_ms={} background_opacity={} fps_color={:?} window_size={}",
            config.target_fps.map_or_else(|| "OFF".to_owned(), |fps| fps.to_string()),
            config.startup_mode,
            config.input_enabled,
            config.hotkey_auto,
            config.hotkey_down,
            config.hotkey_up,
            config.hotkey_toggle,
            config.startup,
            config.overlay_renderer,
            auto_algorithm_summary,
            config.gpu_increase_limit,
            config.gpu_decrease_limit,
            config.frame_jitter_limit_ms,
            config.manual_step_fps,
            config.manual_min_fps,
            config.manual_max_fps,
            config.gpu_poll_rate_ms,
            config.memory_poll_rate_ms,
            config.refresh_rate,
            config.gpu_expiry_ms,
            config.gpu_smoothing_window_ms,
            config.overlay_background_opacity,
            config.overlay_fps_color,
            config.window_size,
        ));
        let frame_cap = Arc::new(FrameCapController::with_manual_limits(
            config.target_fps,
            config.manual_min_fps,
            config.manual_max_fps,
        ));
        if matches!(config.startup_mode, crate::layer::config::StartupMode::Off) {
            frame_cap.set_target_fps(None);
        }
        let overlay_mode = Arc::new(OverlayModeState::new(config.startup));
        let overlay_renderer = Arc::new(OverlayRendererState::new(config.overlay_renderer));
        let active_auto_algorithm = Arc::new(AtomicI32::new(match config.startup_mode {
            crate::layer::config::StartupMode::Off => -1,
            crate::layer::config::StartupMode::Manual => 0,
            crate::layer::config::StartupMode::Automatic(index) => index as i32 + 1,
        }));
        let overlay_feedback = Arc::new(OverlayFeedbackState::new());
        let auto_hysteresis = Arc::new(
            config
                .auto_algorithms
                .iter()
                .map(|algorithm| AtomicU32::new(algorithm.hysteresis.to_bits()))
                .collect::<Vec<_>>(),
        );
        let auto_regulator = AutoRegulator::new(
            config.auto_algorithms.clone(),
            config.gpu_increase_limit,
            config.gpu_decrease_limit,
            config.frame_jitter_limit_ms,
        );
        if config.input_enabled {
            diagnostics::info("registering device with process-wide X11/XWayland hotkey listener");
        } else {
            diagnostics::info("global hotkeys disabled by configuration");
        }
        let hotkeys = HotkeyWorker::start(
            config.input_enabled,
            config.hotkey_auto,
            config.hotkey_up,
            config.hotkey_down,
            config.hotkey_toggle,
            Arc::clone(&frame_cap),
            Arc::clone(&overlay_mode),
            Arc::clone(&overlay_renderer),
            Arc::clone(&active_auto_algorithm),
            Arc::clone(&auto_hysteresis),
            auto_regulator.targets(),
            config
                .auto_algorithms
                .iter()
                .map(|algorithm| algorithm.metric)
                .collect(),
            config
                .auto_algorithms
                .iter()
                .map(|algorithm| algorithm.name.clone())
                .collect(),
            config.manual_step_fps,
            config
                .target_fps
                .unwrap_or(crate::layer::config::DEFAULT_TARGET_FPS),
            Arc::clone(&overlay_feedback),
        );
        let next_present = next_gdpa(*device, c"vkQueuePresentKHR".as_ptr())
            .map(|function| mem::transmute(function));
        let next_acquire = next_gdpa(*device, c"vkAcquireNextImageKHR".as_ptr())
            .map(|function| mem::transmute(function));
        let next_acquire2 = next_gdpa(*device, c"vkAcquireNextImage2KHR".as_ptr())
            .map(|function| mem::transmute(function));
        let gpu_calibration = if let Some((_, _, timestamp_command)) = calibration {
            match next_gdpa(*device, timestamp_command.as_ptr()) {
                Some(function) => {
                    let function: vk::PFN_vkGetCalibratedTimestampsKHR = mem::transmute(function);
                    match CalibrationWorker::start(*device, function) {
                        Ok(worker) => Some(worker),
                        Err(error) => {
                            diagnostics::error(format!(
                                "GPU calibration worker failed to start: {error}"
                            ));
                            None
                        }
                    }
                }
                None => {
                    diagnostics::warn("GPU calibration: timestamp command unavailable");
                    None
                }
            }
        } else {
            diagnostics::warn("GPU calibration unavailable: extension or time domains unsupported");
            None
        };
        let ash_device = ash::Device::load_with(
            |name: &CStr| {
                next_gdpa(*device, name.as_ptr())
                    .map_or(std::ptr::null(), |function| mem::transmute(function))
            },
            *device,
        );
        let gpu_provider = AmdGpuProvider::for_device(
            physical_properties.vendor_id,
            physical_properties.device_id,
        );
        diagnostics::info(format!(
            "GPU telemetry provider selected: sysfs_path={:?}",
            gpu_provider.card_path
        ));
        let memory_gpu_supported = physical_properties.vendor_id == 0x1002
            && physical_properties.device_id == 0x13fe
            && !gpu_provider.card_path.is_empty();
        devices().lock().insert(
            dispatch_key(*device),
            Arc::new(Mutex::new(DeviceState {
                next_gdpa,
                next_present,
                next_acquire,
                next_acquire2,
                trace_dispatch: TraceDispatch::load(*device, next_gdpa),
                ash_device,
                telemetry: TelemetryWorker::start(
                    gpu_provider,
                    HwmonMetricsProvider::with_settings(&config.sensor_settings),
                    Duration::from_millis(config.gpu_poll_rate_ms),
                    Duration::from_millis(config.gpu_smoothing_window_ms),
                    config.memory_poll_rate_ms,
                    memory_gpu_supported,
                ),
                telemetry_expiry: Duration::from_millis(config.gpu_expiry_ms),
                fps_smoothing_window: Duration::from_millis(config.gpu_smoothing_window_ms),
                minimum_metrics_collection_interval: Duration::from_millis(config.gpu_poll_rate_ms),
                overlay_refresh_rate: config.refresh_rate,
                scale_2x_min_width_px: config.scale_2x_min_width_px,
                overlay_background_opacity: config.overlay_background_opacity,
                overlay_renderer,
                overlay_fps_color: config.overlay_fps_color,
                auto_regulator,
                auto_evaluation_schedule: AutoEvaluationSchedule::new(Duration::from_millis(
                    config.gpu_poll_rate_ms,
                )),
                auto_metrics: HashMap::new(),
                auto_evaluation_swapchain: None,
                queue_families: HashMap::new(),
                protected_queues: HashSet::new(),
                multi_device,
                gpu_markers: Markers::new(
                    queue_properties.clone(),
                    physical_properties.limits.timestamp_period,
                ),
                gpu_overlay_markers: Markers::new_overlay(
                    queue_properties,
                    physical_properties.limits.timestamp_period,
                ),
                gpu_calibration,
                frame_cap,
                overlay_mode,
                active_auto_algorithm,
                auto_hysteresis,
                hotkey_config: [
                    config.hotkey_auto,
                    config.hotkey_up,
                    config.hotkey_down,
                    config.hotkey_toggle,
                ],
                manual_step_fps: config.manual_step_fps,
                overlay_feedback,
                schedulers: HashMap::new(),
                present_modes: HashMap::new(),
                hotkeys,
                swapchains: HashMap::new(),
            })),
        );
        TRACE_DISPATCH_EPOCH.fetch_add(1, Ordering::Release);
    }
    result
}

#[no_mangle]
/// Forwards `vkAcquireNextImageKHR`, recording capture and trace data when active.
pub unsafe extern "system" fn vkAcquireNextImageKHR(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    image_index: *mut u32,
) -> vk::Result {
    let acquire = devices()
        .lock()
        .get(&dispatch_key(device))
        .and_then(|state| state.lock().next_acquire);
    let Some(acquire) = acquire else {
        diagnostics::error("vkAcquireNextImageKHR: downstream function was not found");
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let trace = capture::trace(
        "acquire",
        device.as_raw(),
        swapchain.as_raw(),
        1,
        Some(timeout),
    );
    let capture = capture::acquire();
    let result = acquire(device, swapchain, timeout, semaphore, fence, image_index);
    if let Some(capture) = capture {
        let image = if matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR)
            && !image_index.is_null()
        {
            Some(*image_index)
        } else {
            None
        };
        capture.finish(device.as_raw(), swapchain.as_raw(), image, result.as_raw());
    }
    if let Some(trace) = trace {
        trace.finish(result.as_raw());
    }
    result
}

#[no_mangle]
/// Forwards `vkAcquireNextImage2KHR`, recording capture and trace data when active.
pub unsafe extern "system" fn vkAcquireNextImage2KHR(
    device: vk::Device,
    info: *const vk::AcquireNextImageInfoKHR<'_>,
    image_index: *mut u32,
) -> vk::Result {
    let acquire = devices()
        .lock()
        .get(&dispatch_key(device))
        .and_then(|state| state.lock().next_acquire2);
    let Some(acquire) = acquire else {
        diagnostics::error("vkAcquireNextImage2KHR: downstream function was not found");
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let (trace_swapchain, trace_timeout) = if capture::is_active() && !info.is_null() {
        ((*info).swapchain.as_raw(), Some((*info).timeout))
    } else {
        (0, None)
    };
    let trace = capture::trace(
        "acquire2",
        device.as_raw(),
        trace_swapchain,
        1,
        trace_timeout,
    );
    let capture = capture::acquire();
    let result = acquire(device, info, image_index);
    if let Some(capture) = capture {
        let image = if matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR)
            && !info.is_null()
            && !image_index.is_null()
        {
            Some(*image_index)
        } else {
            None
        };
        let swapchain = if info.is_null() {
            0
        } else {
            (*info).swapchain.as_raw()
        };
        capture.finish(device.as_raw(), swapchain, image, result.as_raw());
    }
    if let Some(trace) = trace {
        trace.finish(result.as_raw());
    }
    result
}

#[no_mangle]
/// Creates a swapchain and initializes Framecap resources when its usage permits.
pub unsafe extern "system" fn vkCreateSwapchainKHR(
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    if create_info.is_null() || swapchain.is_null() {
        diagnostics::error("vkCreateSwapchainKHR received a null create_info or swapchain pointer");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let device_key = dispatch_key(device);
    let Some(state) = devices().lock().get(&device_key).cloned() else {
        diagnostics::error("vkCreateSwapchainKHR: device is not tracked by Framecap");
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let (
        next_gdpa,
        ash_device,
        fps_smoothing_window,
        minimum_metrics_collection_interval,
        overlay_refresh_rate,
        scale_2x_min_width_px,
        overlay_background_opacity,
        overlay_renderer,
        overlay_fps_color,
    ) = {
        let state = state.lock();
        (
            state.next_gdpa,
            state.ash_device.clone(),
            state.fps_smoothing_window,
            state.minimum_metrics_collection_interval,
            state.overlay_refresh_rate,
            state.scale_2x_min_width_px,
            state.overlay_background_opacity,
            Arc::clone(&state.overlay_renderer),
            state.overlay_fps_color,
        )
    };
    let create_swapchain = next_gdpa(device, c"vkCreateSwapchainKHR".as_ptr())
        .map(|function| mem::transmute::<_, vk::PFN_vkCreateSwapchainKHR>(function));
    let Some(create_swapchain) = create_swapchain else {
        diagnostics::error("vkCreateSwapchainKHR: downstream function was not found");
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let start = Instant::now();
    let result = create_swapchain(device, create_info, allocator, swapchain);
    let elapsed = start.elapsed();
    if result == vk::Result::SUCCESS {
        state
            .lock()
            .present_modes
            .insert((*swapchain).as_raw(), (*create_info).present_mode);
    }
    if result != vk::Result::SUCCESS {
        diagnostics::error(format!(
            "downstream vkCreateSwapchainKHR failed: result={result:?} extent={}x{} format={:?} image_usage={:?} elapsed_ms={:.3}",
            (*create_info).image_extent.width,
            (*create_info).image_extent.height,
            (*create_info).image_format,
            (*create_info).image_usage,
            elapsed.as_secs_f64() * 1000.0
        ));
    } else {
        diagnostics::info(format!(
            "vkCreateSwapchainKHR succeeded: extent={}x{} format={:?} image_usage={:?} elapsed_ms={:.3}",
            (*create_info).image_extent.width,
            (*create_info).image_extent.height,
            (*create_info).image_format,
            (*create_info).image_usage,
            elapsed.as_secs_f64() * 1000.0
        ));
    }
    if result != vk::Result::SUCCESS
        || !(*create_info)
            .image_usage
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        if result == vk::Result::SUCCESS {
            diagnostics::warn(format!(
                "overlay renderer skipped: swapchain image usage {:?} does not include COLOR_ATTACHMENT",
                (*create_info).image_usage
            ));
        }
        return result;
    }

    let get_images = next_gdpa(device, c"vkGetSwapchainImagesKHR".as_ptr())
        .map(|function| mem::transmute::<_, vk::PFN_vkGetSwapchainImagesKHR>(function));
    let Some(get_images) = get_images else {
        diagnostics::error(
            "vkCreateSwapchainKHR: downstream vkGetSwapchainImagesKHR was not found",
        );
        return result;
    };
    let mut image_count = 0;
    let image_count_result = get_images(device, *swapchain, &mut image_count, std::ptr::null_mut());
    if image_count_result != vk::Result::SUCCESS {
        diagnostics::error(format!(
            "vkGetSwapchainImagesKHR failed while querying image count: result={image_count_result:?}"
        ));
        return result;
    }
    let mut images = vec![vk::Image::null(); image_count as usize];
    let image_result = get_images(device, *swapchain, &mut image_count, images.as_mut_ptr());
    if image_result != vk::Result::SUCCESS {
        diagnostics::error(format!(
            "vkGetSwapchainImagesKHR failed while retrieving images: result={image_result:?}"
        ));
        return result;
    }
    images.truncate(image_count as usize);
    match VulkanSwapchainRenderer::new(
        &ash_device,
        (*create_info).image_format,
        (*create_info).image_extent,
        &images,
        fps_smoothing_window,
        minimum_metrics_collection_interval,
        overlay_refresh_rate,
        overlay_background_opacity,
        overlay_renderer,
        overlay_fps_color,
        scale_2x_min_width_px,
    ) {
        Ok(renderer) => {
            state
                .lock()
                .swapchains
                .insert((*swapchain).as_raw(), renderer);
            diagnostics::info(format!(
                "overlay renderer initialized: images={} extent={}x{}",
                images.len(),
                (*create_info).image_extent.width,
                (*create_info).image_extent.height
            ));
        }
        Err(error) => diagnostics::error(format!(
            "overlay renderer initialization failed: result={error:?}; game presentation will continue without this renderer"
        )),
    }
    result
}

#[no_mangle]
/// Destroys Framecap swapchain resources before forwarding destruction.
pub unsafe extern "system" fn vkDestroySwapchainKHR(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    if let Some(state) = devices().lock().get(&dispatch_key(device)).cloned() {
        capture::retire_swapchain(device.as_raw(), swapchain.as_raw());
        let (next_gdpa, ash_device, mut renderer) = {
            let mut state = state.lock();
            state.auto_metrics.remove(&swapchain.as_raw());
            state.present_modes.remove(&swapchain.as_raw());
            state.gpu_markers.retire(swapchain.as_raw());
            state.gpu_overlay_markers.retire(swapchain.as_raw());
            if state.auto_evaluation_swapchain == Some(swapchain.as_raw()) {
                state.auto_evaluation_swapchain = None;
            }
            (
                state.next_gdpa,
                state.ash_device.clone(),
                state.swapchains.remove(&swapchain.as_raw()),
            )
        };
        if let Some(renderer) = renderer.as_mut() {
            renderer.destroy(&ash_device);
        }
        let destroy_swapchain = next_gdpa(device, c"vkDestroySwapchainKHR".as_ptr())
            .map(|function| mem::transmute::<_, vk::PFN_vkDestroySwapchainKHR>(function));
        if let Some(destroy_swapchain) = destroy_swapchain {
            destroy_swapchain(device, swapchain, allocator);
        }
    }
}

#[no_mangle]
/// Retrieves and records a queue returned by the Vulkan device.
pub unsafe extern "system" fn vkGetDeviceQueue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    queue: *mut vk::Queue,
) {
    if queue.is_null() {
        return;
    }
    let Some(state) = devices().lock().get(&dispatch_key(device)).cloned() else {
        return;
    };
    let next_gdpa = state.lock().next_gdpa;
    let get_queue = next_gdpa(device, c"vkGetDeviceQueue".as_ptr())
        .map(|function| mem::transmute::<_, vk::PFN_vkGetDeviceQueue>(function));
    if let Some(get_queue) = get_queue {
        get_queue(device, queue_family_index, queue_index, queue);
        if !(*queue).is_null() {
            state
                .lock()
                .queue_families
                .insert(dispatch_key(*queue), queue_family_index);
            TRACE_DISPATCH_EPOCH.fetch_add(1, Ordering::Release);
        }
    }
}

#[no_mangle]
/// Retrieves and records a queue using the Vulkan 1.1 queue-info structure.
pub unsafe extern "system" fn vkGetDeviceQueue2(
    device: vk::Device,
    queue_info: *const vk::DeviceQueueInfo2<'_>,
    queue: *mut vk::Queue,
) {
    if queue_info.is_null() || queue.is_null() {
        return;
    }
    let Some(state) = devices().lock().get(&dispatch_key(device)).cloned() else {
        return;
    };
    let next_gdpa = state.lock().next_gdpa;
    let get_queue = next_gdpa(device, c"vkGetDeviceQueue2".as_ptr())
        .map(|function| mem::transmute::<_, vk::PFN_vkGetDeviceQueue2>(function));
    if let Some(get_queue) = get_queue {
        get_queue(device, queue_info, queue);
        if !(*queue).is_null() {
            let mut state = state.lock();
            let key = dispatch_key(*queue);
            state
                .queue_families
                .insert(key, (*queue_info).queue_family_index);
            if (*queue_info)
                .flags
                .contains(vk::DeviceQueueCreateFlags::PROTECTED)
            {
                state.protected_queues.insert(key);
            }
            TRACE_DISPATCH_EPOCH.fetch_add(1, Ordering::Release);
        }
    }
}

macro_rules! trace_downstream {
    ($dispatch:expr, $field:ident, $name:expr, $trace:expr) => {
        match $dispatch.and_then(|dispatch| dispatch.$field) {
            Some(function) => function,
            None => {
                diagnostics::error(format!("{}: downstream function was not found", $name));
                let result = vk::Result::ERROR_INITIALIZATION_FAILED;
                if let Some(trace) = $trace {
                    trace.finish(result.as_raw());
                }
                return result;
            }
        }
    };
}

#[no_mangle]
/// Traces and forwards `vkQueueSubmit`.
pub unsafe extern "system" fn vkQueueSubmit(
    queue: vk::Queue,
    count: u32,
    submits: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> vk::Result {
    let trace = capture::trace("queue_submit", queue.as_raw(), fence.as_raw(), count, None);
    let dispatch = cached_trace_dispatch(queue_key(queue), true);
    let submit = trace_downstream!(dispatch, queue_submit, "vkQueueSubmit", trace);
    let result = submit(queue, count, submits, fence);
    if let Some(trace) = trace {
        trace.finish(result.as_raw());
    }
    result
}

macro_rules! queue_submit2 {
    ($name:ident, $field:ident, $kind:literal) => {
        #[no_mangle]
        pub unsafe extern "system" fn $name(
            queue: vk::Queue,
            count: u32,
            submits: *const vk::SubmitInfo2<'_>,
            fence: vk::Fence,
        ) -> vk::Result {
            let trace = capture::trace($kind, queue.as_raw(), fence.as_raw(), count, None);
            let dispatch = cached_trace_dispatch(queue_key(queue), true);
            let submit = trace_downstream!(dispatch, $field, stringify!($name), trace);
            let result = submit(queue, count, submits, fence);
            if let Some(trace) = trace {
                trace.finish(result.as_raw());
            }
            result
        }
    };
}

queue_submit2!(vkQueueSubmit2, queue_submit2, "queue_submit2");
queue_submit2!(vkQueueSubmit2KHR, queue_submit2_khr, "queue_submit2_khr");

#[no_mangle]
/// Traces and forwards `vkWaitForFences`.
pub unsafe extern "system" fn vkWaitForFences(
    device: vk::Device,
    count: u32,
    fences: *const vk::Fence,
    wait_all: vk::Bool32,
    timeout: u64,
) -> vk::Result {
    let first = if capture::is_active() && count > 0 && !fences.is_null() {
        (*fences).as_raw()
    } else {
        0
    };
    let trace = capture::trace("wait_fences", device.as_raw(), first, count, Some(timeout));
    let dispatch = cached_trace_dispatch(key(device), false);
    let wait = trace_downstream!(dispatch, wait_fences, "vkWaitForFences", trace);
    let result = wait(device, count, fences, wait_all, timeout);
    if let Some(trace) = trace {
        trace.finish(result.as_raw());
    }
    result
}

macro_rules! wait_semaphores {
    ($name:ident, $field:ident, $kind:literal) => {
        #[no_mangle]
        pub unsafe extern "system" fn $name(
            device: vk::Device,
            info: *const vk::SemaphoreWaitInfo<'_>,
            timeout: u64,
        ) -> vk::Result {
            let (count, first) = if capture::is_active() && !info.is_null() {
                let info = &*info;
                let first = if info.semaphore_count > 0 && !info.p_semaphores.is_null() {
                    (*info.p_semaphores).as_raw()
                } else {
                    0
                };
                (info.semaphore_count, first)
            } else {
                (0, 0)
            };
            let trace = capture::trace($kind, device.as_raw(), first, count, Some(timeout));
            let dispatch = cached_trace_dispatch(key(device), false);
            let wait = trace_downstream!(dispatch, $field, stringify!($name), trace);
            let result = wait(device, info, timeout);
            if let Some(trace) = trace {
                trace.finish(result.as_raw());
            }
            result
        }
    };
}

wait_semaphores!(vkWaitSemaphores, wait_semaphores, "wait_semaphores");
wait_semaphores!(
    vkWaitSemaphoresKHR,
    wait_semaphores_khr,
    "wait_semaphores_khr"
);

#[no_mangle]
/// Traces and forwards `vkQueueWaitIdle`.
pub unsafe extern "system" fn vkQueueWaitIdle(queue: vk::Queue) -> vk::Result {
    let trace = capture::trace("queue_wait_idle", queue.as_raw(), 0, 0, None);
    let dispatch = cached_trace_dispatch(queue_key(queue), true);
    let wait = trace_downstream!(dispatch, queue_wait_idle, "vkQueueWaitIdle", trace);
    let result = wait(queue);
    if let Some(trace) = trace {
        trace.finish(result.as_raw());
    }
    result
}

#[no_mangle]
/// Traces and forwards `vkDeviceWaitIdle`.
pub unsafe extern "system" fn vkDeviceWaitIdle(device: vk::Device) -> vk::Result {
    let trace = capture::trace("device_wait_idle", device.as_raw(), 0, 0, None);
    let dispatch = cached_trace_dispatch(key(device), false);
    let wait = trace_downstream!(dispatch, device_wait_idle, "vkDeviceWaitIdle", trace);
    let result = wait(device);
    if let Some(trace) = trace {
        trace.finish(result.as_raw());
    }
    result
}

#[no_mangle]
/// Stops device workers, destroys layer resources, and forwards device destruction.
pub unsafe extern "system" fn vkDestroyDevice(
    device: vk::Device,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    TRACE_DISPATCH_EPOCH.fetch_add(1, Ordering::Release);
    if let Some(state) = devices().lock().remove(&dispatch_key(device)) {
        let (next_gdpa, ash_device, mut swapchains, mut hotkeys) = {
            let mut state = state.lock();
            (
                state.next_gdpa,
                state.ash_device.clone(),
                mem::take(&mut state.swapchains),
                state.hotkeys.take(),
            )
        };
        if let Some(hotkeys) = hotkeys.as_mut() {
            hotkeys.stop();
        }
        state.lock().telemetry.stop();
        if let Some(mut calibration) = state.lock().gpu_calibration.take() {
            calibration.stop();
        }
        state.lock().gpu_markers.destroy(&ash_device);
        state.lock().gpu_overlay_markers.destroy(&ash_device);
        for renderer in swapchains.values_mut() {
            renderer.destroy(&ash_device);
        }
        if let Some(destroy) = next_gdpa(device, c"vkDestroyDevice".as_ptr()) {
            let destroy: vk::PFN_vkDestroyDevice = mem::transmute(destroy);
            destroy(device, allocator);
        }
    }
}

#[no_mangle]
/// Records the overlay and presents with the configured frame-pacing placement.
pub unsafe extern "system" fn vkQueuePresentKHR(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    let trace = capture::trace("present", queue.as_raw(), 0, 0, None);
    let mut capture = capture::frame(queue_key(queue));
    let result = queue_present(queue, present_info, &mut capture);
    if let Some(capture) = capture {
        capture.finish(result.as_raw());
    }
    if let Some(trace) = trace {
        trace.finish(result.as_raw());
    }
    result
}

/// Calls the downstream present, recording its start and end times in the capture frame.
unsafe fn captured_present(
    present: vk::PFN_vkQueuePresentKHR,
    queue: vk::Queue,
    info: *const vk::PresentInfoKHR<'_>,
    capture: &mut Option<CaptureFrame>,
    record_start: impl FnOnce() -> Instant,
) -> (vk::Result, Instant) {
    let started = record_start();
    if let Some(capture) = capture.as_mut() {
        capture.frame.present_start = Some(started);
    }
    let result = present(queue, info);
    if let Some(capture) = capture.as_mut() {
        capture.frame.present_end = Some(Instant::now());
    }
    (result, started)
}

/// Waits for the frame-cap deadline, presents, and drops the queue's scheduler if presentation failed.
unsafe fn paced_present(
    present: vk::PFN_vkQueuePresentKHR,
    queue: vk::Queue,
    info: *const vk::PresentInfoKHR<'_>,
    capture: &mut Option<CaptureFrame>,
    state: &Mutex<DeviceState>,
) -> (vk::Result, Instant) {
    pace_queue(state, queue, capture);
    let (result, started) = captured_present(present, queue, info, capture, || {
        let mut state = state.lock();
        let started = Instant::now();
        state
            .schedulers
            .entry(queue_key(queue))
            .or_default()
            .record_present(started);
        started
    });
    if result != vk::Result::SUCCESS && result != vk::Result::SUBOPTIMAL_KHR {
        state.lock().schedulers.remove(&queue_key(queue));
    }
    (result, started)
}

/// Presents with pacing, then feeds successful presents into the overlay graph, metrics and auto histories.
unsafe fn captured_present_with_graph(
    present: vk::PFN_vkQueuePresentKHR,
    queue: vk::Queue,
    info: *const vk::PresentInfoKHR<'_>,
    capture: &mut Option<CaptureFrame>,
    state: &Arc<Mutex<DeviceState>>,
    swapchains: &[vk::SwapchainKHR],
    graph_enabled: bool,
    metrics: Option<&TelemetrySample>,
) -> vk::Result {
    let (result, started) = paced_present(present, queue, info, capture, state);
    if graph_enabled || metrics.is_some() {
        let mut state = state.lock();
        let successful = result == vk::Result::SUCCESS || result == vk::Result::SUBOPTIMAL_KHR;
        let cap_generation = state.frame_cap.generation();
        for swapchain in swapchains {
            if let Some(renderer) = state.swapchains.get_mut(&swapchain.as_raw()) {
                if graph_enabled {
                    if successful {
                        renderer.record_graph_present(started);
                    } else {
                        renderer.reset_graph();
                    }
                }
                if let Some(metrics) = metrics.filter(|_| successful) {
                    let metrics_start = capture.as_ref().map(|_| Instant::now());
                    let (_, _, refreshed) = renderer.record_present(
                        started,
                        metrics.gpu.clone(),
                        metrics.hardware.clone(),
                        metrics.hardware_timestamp,
                        cap_generation,
                    );
                    if let (Some(capture), Some(start)) = (capture.as_mut(), metrics_start) {
                        capture.frame.metrics_ns += elapsed_ns(start);
                        capture.frame.overlay_refreshes += u32::from(refreshed);
                    }
                    let auto_start = capture.as_ref().map(|_| Instant::now());
                    let windows: Vec<_> = state.auto_regulator.windows().collect();
                    let histories =
                        state
                            .auto_metrics
                            .entry(swapchain.as_raw())
                            .or_insert_with(|| {
                                windows.into_iter().map(AutoMetricsHistory::new).collect()
                            });
                    for history in histories.iter_mut() {
                        history.record(
                            started,
                            &metrics.raw_gpu,
                            &metrics.raw_hardware,
                            metrics.hardware_timestamp,
                        );
                    }
                    if let (Some(capture), Some(start)) = (capture.as_mut(), auto_start) {
                        capture.frame.auto_ns += elapsed_ns(start);
                    }
                }
            }
        }
    }
    result
}

/// Presents and poisons the GPU marker for the image if presentation failed.
unsafe fn captured_present_with_marker(
    present: vk::PFN_vkQueuePresentKHR,
    queue: vk::Queue,
    info: *const vk::PresentInfoKHR<'_>,
    capture: &mut Option<CaptureFrame>,
    state: &Arc<Mutex<DeviceState>>,
    swapchains: &[vk::SwapchainKHR],
    graph_enabled: bool,
    marker: Option<(u64, u32)>,
    metrics: Option<&TelemetrySample>,
) -> vk::Result {
    let result = captured_present_with_graph(
        present,
        queue,
        info,
        capture,
        state,
        swapchains,
        graph_enabled,
        metrics,
    );
    if result != vk::Result::SUCCESS && result != vk::Result::SUBOPTIMAL_KHR {
        if let Some((swapchain, image_index)) = marker {
            state.lock().gpu_markers.poison(swapchain, image_index);
            diagnostics::error(format!(
                "GPU capture presentation failed: result={result:?}; marker disabled for image"
            ));
        }
    }
    result
}

/// Sleeps until the queue's next frame-cap deadline, recording scheduling details when capturing.
fn pace_queue(state: &Mutex<DeviceState>, queue: vk::Queue, capture: &mut Option<CaptureFrame>) {
    let deadline = {
        let mut state = state.lock();
        let target_fps = state.frame_cap.target_fps();
        let scheduler = state.schedulers.entry(queue_key(queue)).or_default();
        let now = Instant::now();
        if let Some(capture) = capture.as_mut() {
            (
                capture.frame.previous_target,
                capture.frame.previous_deadline,
            ) = scheduler.scheduled_state();
            capture.frame.schedule_at = Some(now);
            capture.frame.target = target_fps;
        }
        let deadline = scheduler.next_deadline(now, target_fps);
        if let Some(capture) = capture.as_mut() {
            capture.frame.deadline = deadline;
        }
        deadline
    };
    if let Some(deadline) = deadline {
        if let Some(capture) = capture.as_mut() {
            capture.frame.wait_start = Some(Instant::now());
        }
        if let Err(wait_error) = wait_until(deadline) {
            if let Some(capture) = capture.as_mut() {
                capture.frame.wait_failed = true;
            }
            static WAIT_ERROR_LOGGED: AtomicBool = AtomicBool::new(false);
            if !WAIT_ERROR_LOGGED.swap(true, Ordering::Relaxed) {
                diagnostics::error(format!(
                    "frame cap wait failed; continuing presentation: {wait_error}"
                ));
            }
        }
    }
    if let Some(capture) = capture.as_mut() {
        capture.frame.wake = Some(Instant::now());
    }
}

/// Copies present-mode, present-id and Google present-time info from the `pNext` chain into the capture frame.
unsafe fn capture_present_metadata(info: &vk::PresentInfoKHR<'_>, frame: &mut capture::Frame) {
    let mut next = info.p_next.cast::<vk::BaseInStructure<'_>>();
    while let Some(header) = next.as_ref() {
        match header.s_type {
            vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT => {
                let modes = &*next.cast::<vk::SwapchainPresentModeInfoEXT<'_>>();
                if modes.swapchain_count > 0 && !modes.p_present_modes.is_null() {
                    frame.present_mode_override = Some((*modes.p_present_modes).as_raw());
                }
            }
            vk::StructureType::PRESENT_ID_KHR => {
                let ids = &*next.cast::<vk::PresentIdKHR<'_>>();
                if ids.swapchain_count > 0 && !ids.p_present_ids.is_null() {
                    frame.present_id = Some(*ids.p_present_ids);
                }
            }
            vk::StructureType::PRESENT_TIMES_INFO_GOOGLE => {
                let times = &*next.cast::<vk::PresentTimesInfoGOOGLE<'_>>();
                if times.swapchain_count > 0 && !times.p_times.is_null() {
                    frame.google_present_id = Some((*times.p_times).present_id);
                }
            }
            _ => {}
        }
        next = header.p_next;
    }
}

/// Core `vkQueuePresentKHR` logic: gathers device state and telemetry, submits GPU markers and presents.
unsafe fn queue_present(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
    capture: &mut Option<CaptureFrame>,
) -> vk::Result {
    if present_info.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let Some(state) = device_for_queue(queue) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    if let Some(capture) = capture.as_mut() {
        capture.frame.lookup_done = Some(Instant::now());
    }
    let (present, ash_device, queue_family, metrics, calibration) = {
        let mut state = state.lock();
        let ash_device = state.ash_device.clone();
        state.gpu_markers.poll(&ash_device);
        state.gpu_overlay_markers.poll(&ash_device);
        (
            state.next_present,
            ash_device,
            state.queue_families.get(&queue_key(queue)).copied(),
            state.telemetry.current(state.telemetry_expiry),
            capture.as_ref().and_then(|_| {
                state
                    .gpu_calibration
                    .as_ref()
                    .map(CalibrationWorker::current)
            }),
        )
    };
    let Some(present) = present else {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    if let Some(capture) = capture.as_mut() {
        capture.frame.context_done = Some(Instant::now());
        capture.record_calibration(calibration);
        if metrics.raw_gpu.status == TelemetryStatus::Available {
            capture.frame.gpu_load = Some(metrics.raw_gpu.value);
            capture.frame.gpu_timestamp = Some(metrics.raw_gpu.timestamp);
        }
    }
    let info = &*present_info;
    if let Some(capture) = capture.as_mut() {
        capture.frame.swapchain_count = info.swapchain_count;
        capture_present_metadata(info, &mut capture.frame);
    }
    if info.swapchain_count == 0 || info.p_swapchains.is_null() || info.p_image_indices.is_null() {
        return paced_present(present, queue, present_info, capture, &state).0;
    }
    let swapchains = std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize);
    let image_indices =
        std::slice::from_raw_parts(info.p_image_indices, info.swapchain_count as usize);
    if let Some(capture) = capture.as_mut() {
        capture.associate_acquisition(
            ash_device.handle().as_raw(),
            swapchains[0].as_raw(),
            image_indices[0],
        );
    }
    let mut waits = if info.wait_semaphore_count == 0 || info.p_wait_semaphores.is_null() {
        Vec::new()
    } else {
        std::slice::from_raw_parts(info.p_wait_semaphores, info.wait_semaphore_count as usize)
            .to_vec()
    };
    let mut marker_key = None;
    if let Some(capture) = capture.as_mut() {
        let mut state = state.lock();
        capture.frame.gpu_marker_status = Some("unsupported");
        capture.frame.gpu_overlay_marker_status = Some("no_overlay");
        if let Some(family) = queue_family.filter(|_| {
            !state.multi_device
                && !state.protected_queues.contains(&queue_key(queue))
                && swapchains.len() == 1
                && !waits.is_empty()
        }) {
            capture.frame.gpu_marker_submit_start = Some(Instant::now());
            let result = state.gpu_markers.submit(
                &ash_device,
                queue,
                family,
                swapchains[0].as_raw(),
                image_indices[0],
                &waits,
            );
            capture.frame.gpu_marker_submit_end = Some(Instant::now());
            match result {
                Ok(Submission::Submitted(signal, sample)) => {
                    waits.clear();
                    waits.push(signal);
                    marker_key = Some((swapchains[0].as_raw(), image_indices[0]));
                    capture.frame.gpu_marker = Some(sample);
                    capture.frame.gpu_marker_status = Some("pending");
                }
                Ok(Submission::Skipped(reason)) => {
                    capture.frame.gpu_marker_status = Some(reason);
                }
                Err(error) => {
                    capture.frame.gpu_marker_status = Some("error");
                    diagnostics::error(format!("GPU capture marker failed: {error:?}"));
                    if error == vk::Result::ERROR_DEVICE_LOST {
                        return error;
                    }
                }
            }
        }
    }
    let mut marker_info = *info;
    if marker_key.is_some() {
        marker_info.p_wait_semaphores = waits.as_ptr();
        marker_info.wait_semaphore_count = waits.len() as u32;
    }
    let effective_present_info = if marker_key.is_some() {
        &marker_info as *const _
    } else {
        present_info
    };
    let mut commands = Vec::new();
    let mut signals = Vec::new();
    let mut graph_enabled = false;
    if let Some(queue_family) = queue_family {
        if let Some(capture) = capture.as_mut() {
            capture.frame.work_lock_start = Some(Instant::now());
        }
        let mut state = state.lock();
        if let Some(capture) = capture.as_mut() {
            capture.frame.work_lock_acquired = Some(Instant::now());
            capture.frame.swapchain = swapchains[0].as_raw();
            capture.frame.present_mode = state
                .present_modes
                .get(&capture.frame.swapchain)
                .map(|mode| mode.as_raw());
        }
        let active_mode = state.active_auto_algorithm.load(Ordering::Relaxed);
        let auto_algorithm_index = (active_mode > 0).then_some((active_mode - 1) as usize);
        let auto_algorithm_label =
            auto_algorithm_index.and_then(|index| state.auto_regulator.overlay_label(index));
        let auto_target =
            auto_algorithm_index.and_then(|index| state.auto_regulator.overlay_target(index));
        let target_fps = state.frame_cap.target_fps();
        let hysteresis = auto_algorithm_index
            .and_then(|index| state.auto_hysteresis.get(index))
            .map(|value| f32::from_bits(value.load(Ordering::Relaxed)));
        let overlay_mode = state.overlay_mode.mode();
        graph_enabled = overlay_mode == OverlayMode::Full
            && state.overlay_renderer.current() == crate::layer::settings::OverlayRenderer::Ascii;
        if let Some(capture) = capture.as_mut() {
            capture.frame.overlay_mode = Some(overlay_mode as u8);
            capture.frame.overlay_renderer = Some(match state.overlay_renderer.current() {
                crate::layer::settings::OverlayRenderer::Ascii => "ascii",
                crate::layer::settings::OverlayRenderer::Modern => "modern",
                crate::layer::settings::OverlayRenderer::Helix => "helix",
            });
        }
        for (&swapchain, &image_index) in swapchains.iter().zip(image_indices) {
            if !graph_enabled {
                if let Some(renderer) = state.swapchains.get_mut(&swapchain.as_raw()) {
                    renderer.reset_graph();
                }
            }
            if overlay_mode == OverlayMode::Off {
                continue;
            }
            let overlay_start = capture.as_ref().map(|_| Instant::now());
            let hotkey_config = state.hotkey_config;
            let manual_step_fps = state.manual_step_fps;
            let feedback = state.overlay_feedback.current(Instant::now());
            let Some(renderer) = state.swapchains.get_mut(&swapchain.as_raw()) else {
                continue;
            };
            let Some(displayed_metrics) = renderer.displayed_metrics() else {
                continue;
            };
            match renderer.record_commands(
                &ash_device,
                queue_family,
                image_index,
                &displayed_metrics,
                overlay_mode,
                target_fps,
                auto_algorithm_index,
                auto_algorithm_label,
                hysteresis,
                auto_target,
                hotkey_config,
                manual_step_fps,
                feedback,
            ) {
                Ok((command, signal)) => {
                    commands.push(command);
                    signals.push(signal);
                }
                Err(error) => {
                    static RENDER_ERROR_LOGGED: AtomicBool = AtomicBool::new(false);
                    if !RENDER_ERROR_LOGGED.swap(true, Ordering::Relaxed) {
                        diagnostics::error(format!(
                            "overlay command recording failed: result={error:?}; subsequent render errors suppressed"
                        ));
                    }
                }
            }
            if let (Some(capture), Some(start)) = (capture.as_mut(), overlay_start) {
                capture.frame.overlay_ns += elapsed_ns(start);
            }
        }
    }

    if queue_family.is_some() {
        if let Some(capture) = capture.as_mut() {
            capture.frame.metrics_lock_start = Some(Instant::now());
        }
        let mut state = state.lock();
        if let Some(capture) = capture.as_mut() {
            capture.frame.metrics_lock_acquired = Some(Instant::now());
        }
        let active_mode = state.active_auto_algorithm.load(Ordering::Relaxed);
        let auto_algorithm_index = (active_mode > 0).then_some((active_mode - 1) as usize);
        state.auto_regulator.set_active(auto_algorithm_index);
        let mut target_fps = state.frame_cap.target_fps();
        let hysteresis = auto_algorithm_index
            .and_then(|index| state.auto_hysteresis.get(index))
            .map(|value| f32::from_bits(value.load(Ordering::Relaxed)));
        let frame_cap = Arc::clone(&state.frame_cap);
        if let Some(capture) = capture.as_mut() {
            capture.frame.auto_mode = Some(active_mode);
        }
        for &swapchain in swapchains {
            if !state.swapchains.contains_key(&swapchain.as_raw()) {
                continue;
            }
            let auto_start = capture.as_ref().map(|_| Instant::now());
            let evaluation_time = Instant::now();
            if state.auto_evaluation_schedule.has_gap(evaluation_time) {
                if let Some(index) = auto_algorithm_index {
                    state.auto_regulator.reset_pending(index);
                }
            }
            let evaluation_due = state
                .auto_evaluation_schedule
                .should_evaluate(evaluation_time, auto_algorithm_index);
            if evaluation_due {
                if let Some(capture) = capture.as_mut() {
                    capture.frame.auto_evaluations += 1;
                }
            }
            if let Some(algorithm_index) = auto_algorithm_index.filter(|_| evaluation_due) {
                if state.auto_evaluation_swapchain != Some(swapchain.as_raw()) {
                    state.auto_regulator.reset_pending(algorithm_index);
                    state.auto_evaluation_swapchain = Some(swapchain.as_raw());
                }
                let metric = state.auto_regulator.metric(algorithm_index);
                let auto_metrics = metric.and_then(|metric| {
                    state
                        .auto_metrics
                        .get_mut(&swapchain.as_raw())
                        .and_then(|histories| histories.get_mut(algorithm_index))
                        .and_then(|history| history.snapshot(evaluation_time, metric))
                });
                if let (Some(auto_metrics), Some(hysteresis)) = (auto_metrics, hysteresis) {
                    if let Some((old, new)) = state.auto_regulator.evaluate(
                        algorithm_index,
                        &auto_metrics,
                        evaluation_time,
                        hysteresis,
                        &frame_cap,
                    ) {
                        state
                            .overlay_feedback
                            .record(OverlayFeedbackEvent::CapChanged);
                        diagnostics::info(format!(
                            "[AUTO] FPS limit adjusted: {old:.1} -> {new:.1} fps"
                        ));
                        target_fps = state.frame_cap.target_fps();
                    }
                } else {
                    state.auto_regulator.reset_pending(algorithm_index);
                }
            }
            if let (Some(capture), Some(start)) = (capture.as_mut(), auto_start) {
                capture.frame.auto_ns += elapsed_ns(start);
                capture.frame.target_after = target_fps;
            }
        }
    }
    if commands.is_empty() && signals.is_empty() {
        return captured_present_with_marker(
            present,
            queue,
            effective_present_info,
            capture,
            &state,
            swapchains,
            graph_enabled,
            marker_key,
            queue_family.map(|_| &metrics),
        );
    }

    let mut overlay_marker = None;
    if let Some(capture) = capture.as_mut().filter(|_| !commands.is_empty()) {
        capture.frame.gpu_overlay_marker_status = Some("unsupported");
        if let Some(family) = queue_family {
            capture.frame.gpu_overlay_marker_prepare_start = Some(Instant::now());
            let mut state = state.lock();
            if !state.multi_device
                && !state.protected_queues.contains(&queue_key(queue))
                && swapchains.len() == 1
            {
                let result = state.gpu_overlay_markers.prepare_overlay(
                    &ash_device,
                    family,
                    swapchains[0].as_raw(),
                    image_indices[0],
                );
                match result {
                    Ok(Preparation::Prepared(prepared)) => {
                        capture.frame.gpu_overlay_marker = Some(Arc::clone(&prepared.sample));
                        capture.frame.gpu_overlay_marker_status = Some("pending");
                        commands.push(prepared.command);
                        overlay_marker = Some((swapchains[0].as_raw(), image_indices[0], prepared));
                    }
                    Ok(Preparation::Skipped(reason)) => {
                        capture.frame.gpu_overlay_marker_status = Some(reason);
                    }
                    Err(error) => {
                        capture.frame.gpu_overlay_marker_status = Some("error");
                        diagnostics::error(format!("GPU overlay capture marker failed: {error:?}"));
                        if error == vk::Result::ERROR_DEVICE_LOST {
                            return error;
                        }
                    }
                }
            }
            capture.frame.gpu_overlay_marker_prepare_end = Some(Instant::now());
        }
    }

    let submit_fence = overlay_marker
        .as_ref()
        .map_or(vk::Fence::null(), |(_, _, marker)| marker.fence);

    let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
    let submit = [vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&commands)
        .signal_semaphores(&signals)];
    if let Some(capture) = capture.as_mut() {
        capture.frame.submit_start = Some(Instant::now());
    }
    let submit_result = ash_device.queue_submit(queue, &submit, submit_fence);
    if let Some(capture) = capture.as_mut() {
        capture.frame.submit_end = Some(Instant::now());
        capture.frame.submit_result = Some(match submit_result {
            Ok(()) => vk::Result::SUCCESS.as_raw(),
            Err(error) => error.as_raw(),
        });
    }
    if let Err(error) = submit_result {
        if let Some((swapchain, image_index, _)) = overlay_marker.as_ref() {
            state
                .lock()
                .gpu_overlay_markers
                .fail(*swapchain, *image_index);
        }
        static SUBMIT_ERROR_LOGGED: AtomicBool = AtomicBool::new(false);
        if !SUBMIT_ERROR_LOGGED.swap(true, Ordering::Relaxed) {
            diagnostics::error(format!(
                "overlay queue submission failed: result={error:?}; subsequent submission errors suppressed"
            ));
        }
        return captured_present_with_marker(
            present,
            queue,
            effective_present_info,
            capture,
            &state,
            swapchains,
            graph_enabled,
            marker_key,
            queue_family.map(|_| &metrics),
        );
    }
    let mut modified_info = *info;
    modified_info.p_wait_semaphores = signals.as_ptr();
    modified_info.wait_semaphore_count = signals.len() as u32;
    captured_present_with_marker(
        present,
        queue,
        &modified_info,
        capture,
        &state,
        swapchains,
        graph_enabled,
        marker_key,
        queue_family.map(|_| &metrics),
    )
}

/// Returns nanoseconds elapsed since `start`, saturating at `u64::MAX`.
fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn trace_interceptors_forward_queue_and_wait_calls() {
        static SUBMIT_OK: AtomicBool = AtomicBool::new(false);
        static WAIT_OK: AtomicBool = AtomicBool::new(false);
        static ALIAS_OK: AtomicBool = AtomicBool::new(false);
        unsafe extern "system" fn submit(
            queue: vk::Queue,
            count: u32,
            submits: *const vk::SubmitInfo<'_>,
            fence: vk::Fence,
        ) -> vk::Result {
            SUBMIT_OK.store(
                queue.as_raw() == 41 && count == 3 && submits.is_null() && fence.as_raw() == 11,
                Ordering::Relaxed,
            );
            vk::Result::SUCCESS
        }
        unsafe extern "system" fn wait(
            device: vk::Device,
            count: u32,
            fences: *const vk::Fence,
            wait_all: vk::Bool32,
            timeout: u64,
        ) -> vk::Result {
            WAIT_OK.store(
                device.as_raw() == 42
                    && count == 1
                    && !fences.is_null()
                    && (*fences).as_raw() == 11
                    && wait_all == vk::TRUE
                    && timeout == 123,
                Ordering::Relaxed,
            );
            vk::Result::TIMEOUT
        }
        unsafe extern "system" fn submit2(
            queue: vk::Queue,
            count: u32,
            submits: *const vk::SubmitInfo2<'_>,
            fence: vk::Fence,
        ) -> vk::Result {
            ALIAS_OK.store(
                queue.as_raw() == 41 && count == 2 && submits.is_null() && fence.as_raw() == 11,
                Ordering::Relaxed,
            );
            vk::Result::SUCCESS
        }
        let dispatch = TraceDispatch {
            queue_submit: Some(submit),
            queue_submit2: None,
            queue_submit2_khr: Some(submit2),
            wait_fences: Some(wait),
            wait_semaphores: None,
            wait_semaphores_khr: None,
            queue_wait_idle: None,
            device_wait_idle: None,
        };
        TRACE_DISPATCH_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.epoch = TRACE_DISPATCH_EPOCH.load(Ordering::Acquire);
            cache.queues.insert(41, dispatch);
            cache.devices.insert(42, dispatch);
        });
        let fence = vk::Fence::from_raw(11);
        assert_eq!(
            unsafe { vkQueueSubmit(vk::Queue::from_raw(41), 3, std::ptr::null(), fence) },
            vk::Result::SUCCESS
        );
        assert_eq!(
            unsafe { vkWaitForFences(vk::Device::from_raw(42), 1, &fence, vk::TRUE, 123) },
            vk::Result::TIMEOUT
        );
        assert_eq!(
            unsafe { vkQueueSubmit2KHR(vk::Queue::from_raw(41), 2, std::ptr::null(), fence) },
            vk::Result::SUCCESS
        );
        assert!(SUBMIT_OK.load(Ordering::Relaxed));
        assert!(WAIT_OK.load(Ordering::Relaxed));
        assert!(ALIAS_OK.load(Ordering::Relaxed));
    }

    #[test]
    fn capture_reads_existing_present_metadata_without_changing_it() {
        let modes = [vk::PresentModeKHR::MAILBOX];
        let ids = [42];
        let times = [vk::PresentTimeGOOGLE::default().present_id(17)];
        let mut mode = vk::SwapchainPresentModeInfoEXT::default().present_modes(&modes);
        let mut id = vk::PresentIdKHR::default().present_ids(&ids);
        let mut timing = vk::PresentTimesInfoGOOGLE::default().times(&times);
        let info = vk::PresentInfoKHR::default()
            .push_next(&mut mode)
            .push_next(&mut id)
            .push_next(&mut timing);
        let mut frame = capture::Frame::default();
        unsafe { capture_present_metadata(&info, &mut frame) };
        assert_eq!(
            frame.present_mode_override,
            Some(vk::PresentModeKHR::MAILBOX.as_raw())
        );
        assert_eq!(frame.present_id, Some(42));
        assert_eq!(frame.google_present_id, Some(17));
        assert!(!info.p_next.is_null());
    }

    #[test]
    fn captured_present_forwards_arguments_and_records_downstream_timing() {
        unsafe extern "system" fn present(
            queue: vk::Queue,
            info: *const vk::PresentInfoKHR<'_>,
        ) -> vk::Result {
            assert_eq!(queue.as_raw(), 42);
            assert!(!info.is_null());
            vk::Result::SUBOPTIMAL_KHR
        }
        let info = vk::PresentInfoKHR::default();
        let mut capture = Some(CaptureFrame::for_test());
        let (result, _) = unsafe {
            captured_present(
                present,
                vk::Queue::from_raw(42),
                &info,
                &mut capture,
                Instant::now,
            )
        };
        assert_eq!(result, vk::Result::SUBOPTIMAL_KHR);
        let frame = &capture.unwrap().frame;
        assert!(frame.present_start.unwrap() <= frame.present_end.unwrap());
    }

    #[test]
    fn memory_temperature_averages_distinct_samples_and_preserves_lower_bounds() {
        let now = Instant::now();
        let mut history = VecDeque::new();
        let window = Duration::from_secs(5);
        let sample = |celsius, lower_bound| MemoryTemperature {
            celsius,
            lower_bound,
        };
        assert_eq!(
            smooth_memory_temperature(&mut history, now, window, sample(40.0, false)),
            sample(40.0, false)
        );
        assert_eq!(
            smooth_memory_temperature(
                &mut history,
                now + Duration::from_secs(3),
                window,
                sample(120.0, true)
            ),
            sample(80.0, true)
        );
        assert_eq!(
            smooth_memory_temperature(
                &mut history,
                now + Duration::from_secs(9),
                window,
                sample(50.0, false)
            ),
            sample(50.0, false)
        );
    }

    #[test]
    fn memory_worker_polls_without_blocking_telemetry_and_stops_between_reads() {
        let (sender, receiver) = mpsc::channel();
        let mut worker = MemoryTemperatureWorker::start_with_reader(
            Duration::from_millis(100),
            Duration::from_secs(1),
            move || {
                sender.send(()).unwrap();
                Ok(crate::layer::memory_temperature::Hotspot {
                    temperature_c: 48,
                    saturated: false,
                })
            },
        );
        receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(receiver.recv_timeout(Duration::from_millis(30)).is_err());
        receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.stop();
    }

    #[test]
    fn memory_worker_waits_for_in_flight_reader_on_shutdown() {
        let (started_tx, started_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let mut worker = MemoryTemperatureWorker::start_with_reader(
            Duration::from_secs(5),
            Duration::from_secs(1),
            move || {
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
                Ok(crate::layer::memory_temperature::Hotspot {
                    temperature_c: 48,
                    saturated: false,
                })
            },
        );
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let stopper = thread::spawn(move || {
            worker.stop();
            stopped_tx.send(()).unwrap();
        });
        assert!(stopped_rx.recv_timeout(Duration::from_millis(100)).is_err());
        resume_tx.send(()).unwrap();
        stopped_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        stopper.join().unwrap();
    }

    #[test]
    fn memory_worker_rejects_stale_samples() {
        let worker = MemoryTemperatureWorker {
            latest: Arc::new(parking_lot::RwLock::new(Some((
                MemoryTemperature {
                    celsius: 48.0,
                    lower_bound: false,
                },
                Instant::now() - Duration::from_secs(5),
            )))),
            stop: Arc::new((Mutex::new(false), Condvar::new())),
            worker: None,
            interval: Duration::from_secs(2),
            stale_reported: AtomicBool::new(false),
        };
        assert_eq!(worker.current(), None);
        assert!(worker.stale_reported.load(Ordering::Relaxed));
    }

    /// Supplies a test dispatch function that reports no downstream entry points.
    unsafe extern "system" fn test_gipa(
        _instance: vk::Instance,
        _name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        None
    }

    #[test]
    /// Checks that queue lookup keys preserve the raw Vulkan handle value.
    fn queue_handle_key_uses_raw_handle_value() {
        let queue = vk::Queue::from_raw(0x1234_5678_9abc_def0);
        assert_eq!(queue_key(queue), 0x1234_5678_9abc_def0usize);
    }

    #[test]
    /// Checks that a physical device resolves to its enumerating instance.
    fn physical_device_selects_its_parent_instance() {
        let first = vk::Instance::from_raw(1);
        let second = vk::Instance::from_raw(2);
        let physical_device = vk::PhysicalDevice::from_raw(3);
        let mut instances = HashMap::new();
        instances.insert(
            key(first),
            InstanceDispatch {
                handle: first,
                next_gipa: test_gipa,
                calibration_core: true,
                calibration_properties2: false,
            },
        );
        instances.insert(
            key(second),
            InstanceDispatch {
                handle: second,
                next_gipa: test_gipa,
                calibration_core: true,
                calibration_properties2: false,
            },
        );
        let physical_device_instances = HashMap::from([(key(physical_device), key(second))]);

        assert_eq!(
            find_instance_dispatch(key(physical_device), &physical_device_instances, &instances)
                .map(|dispatch| dispatch.handle),
            Some(second)
        );
    }

    #[test]
    /// Checks that fallback routing is allowed only for a single instance.
    fn unmapped_physical_device_only_falls_back_when_one_instance_exists() {
        let instance = vk::Instance::from_raw(1);
        let physical_device = vk::PhysicalDevice::from_raw(3);
        let instances = HashMap::from([(
            key(instance),
            InstanceDispatch {
                handle: instance,
                next_gipa: test_gipa,
                calibration_core: true,
                calibration_properties2: false,
            },
        )]);
        let physical_device_instances = HashMap::new();

        assert_eq!(
            find_instance_dispatch(key(physical_device), &physical_device_instances, &instances)
                .map(|dispatch| dispatch.handle),
            Some(instance)
        );

        let multiple_instances = HashMap::from([
            (
                key(instance),
                InstanceDispatch {
                    handle: instance,
                    next_gipa: test_gipa,
                    calibration_core: true,
                    calibration_properties2: false,
                },
            ),
            (
                key(vk::Instance::from_raw(2)),
                InstanceDispatch {
                    handle: vk::Instance::from_raw(2),
                    next_gipa: test_gipa,
                    calibration_core: true,
                    calibration_properties2: false,
                },
            ),
        ]);
        assert!(find_instance_dispatch(
            key(physical_device),
            &physical_device_instances,
            &multiple_instances
        )
        .is_none());
    }
}
