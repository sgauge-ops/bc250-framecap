use crate::layer::diagnostics;
use crate::layer::gpu_capture::{Calibration, CalibrationState, Sample, SampleResult, Timestamp};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CAPTURE_DURATION: Duration = Duration::from_secs(30);
const MAX_FRAMES: usize = 65536;
const MAX_TRACE_EVENTS: usize = 262144;

thread_local! {
    static TRACE_THREAD_ID: i64 = unsafe { libc::syscall(libc::SYS_gettid) } as i64;
}

/// Process-wide capture state: the active session slot and counters for entries skipped under lock contention.
#[derive(Default)]
struct CaptureManager {
    active: AtomicBool,
    session: Mutex<Option<Arc<Session>>>,
    skipped_entries: AtomicU64,
    trace_skipped_entries: AtomicU64,
}

/// One timed capture window with its collected frames, pending acquisitions, drop counters and trace buffer.
struct Session {
    start: Instant,
    end: Instant,
    unix_start_ns: Option<u128>,
    frames: Mutex<Vec<Frame>>,
    acquisitions: Mutex<HashMap<(u64, u64, u32), Acquisition>>,
    started: AtomicU64,
    dropped: AtomicU64,
    skipped_entries: AtomicU64,
    incomplete: AtomicU64,
    acquisition_dropped: AtomicU64,
    acquisition_failed: AtomicU64,
    trace: TraceBuffer,
}

/// Bounded buffer of traced Vulkan call events plus counters for dropped and skipped ones.
#[derive(Default)]
struct TraceBuffer {
    events: Mutex<Vec<TraceEvent>>,
    started: AtomicU64,
    dropped: AtomicU64,
    skipped_entries: AtomicU64,
}

/// Completed traced call: its kind, thread, handles, timing and Vulkan result.
struct TraceEvent {
    kind: &'static str,
    thread_id: i64,
    object: u64,
    related: u64,
    count: u32,
    timeout_ns: Option<u64>,
    start: Instant,
    end: Instant,
    result: i32,
}

/// In-flight traced call, turned into a `TraceEvent` by `finish`.
pub(crate) struct TraceStart {
    session: Arc<Session>,
    kind: &'static str,
    thread_id: i64,
    object: u64,
    related: u64,
    count: u32,
    timeout_ns: Option<u64>,
    start: Instant,
}

/// Timing and result of an image acquisition, kept until the matching present claims it.
#[derive(Clone, Copy)]
struct Acquisition {
    start: Instant,
    end: Instant,
    result: i32,
}

/// In-flight image acquisition being timed for the active session.
pub(crate) struct CaptureAcquire {
    session: Arc<Session>,
    start: Instant,
}

/// Handle to a frame being recorded; `finish` commits it to the session.
pub struct CaptureFrame {
    session: Arc<Session>,
    pub frame: Frame,
}

/// Per-present record of CPU timestamps, pacing decisions, GPU markers and submit/present results.
#[derive(Default)]
pub struct Frame {
    pub queue: usize,
    pub swapchain: u64,
    pub swapchain_count: u32,
    pub present_mode: Option<i32>,
    pub present_mode_override: Option<i32>,
    pub present_id: Option<u64>,
    pub google_present_id: Option<u32>,
    pub image_index: Option<u32>,
    pub acquire_start: Option<Instant>,
    pub acquire_end: Option<Instant>,
    pub acquire_result: Option<i32>,
    pub entry: Option<Instant>,
    pub lookup_done: Option<Instant>,
    pub context_done: Option<Instant>,
    pub schedule_at: Option<Instant>,
    pub previous_deadline: Option<Instant>,
    pub previous_target: Option<f32>,
    pub target: Option<f32>,
    pub deadline: Option<Instant>,
    pub wait_start: Option<Instant>,
    pub wake: Option<Instant>,
    pub wait_failed: bool,
    pub work_lock_start: Option<Instant>,
    pub work_lock_acquired: Option<Instant>,
    pub metrics_lock_start: Option<Instant>,
    pub metrics_lock_acquired: Option<Instant>,
    pub metrics_ns: u64,
    pub auto_ns: u64,
    pub overlay_ns: u64,
    pub overlay_refreshes: u32,
    pub overlay_mode: Option<u8>,
    pub overlay_renderer: Option<&'static str>,
    pub auto_mode: Option<i32>,
    pub auto_evaluations: u32,
    pub target_after: Option<f32>,
    pub gpu_load: Option<f32>,
    pub gpu_timestamp: Option<Instant>,
    pub gpu_marker: Option<Sample>,
    pub gpu_marker_status: Option<&'static str>,
    pub gpu_marker_submit_start: Option<Instant>,
    pub gpu_marker_submit_end: Option<Instant>,
    pub gpu_overlay_marker: Option<Sample>,
    pub gpu_overlay_marker_status: Option<&'static str>,
    pub gpu_overlay_marker_prepare_start: Option<Instant>,
    pub gpu_overlay_marker_prepare_end: Option<Instant>,
    pub gpu_calibration: Option<Calibration>,
    pub gpu_calibration_status: Option<&'static str>,
    pub submit_start: Option<Instant>,
    pub submit_end: Option<Instant>,
    pub submit_result: Option<i32>,
    pub present_start: Option<Instant>,
    pub present_end: Option<Instant>,
    pub present_result: Option<i32>,
}

/// Returns the lazily initialised global capture manager.
fn manager() -> &'static CaptureManager {
    static MANAGER: OnceLock<CaptureManager> = OnceLock::new();
    MANAGER.get_or_init(CaptureManager::default)
}

/// Cheaply reports whether a capture session is currently recording.
pub(crate) fn is_active() -> bool {
    manager().active.load(Ordering::Acquire)
}

/// Starts a process-wide capture. Repeated requests are ignored until export finishes.
pub fn start() {
    if let Err(error) = start_with(manager(), CAPTURE_DURATION, save) {
        diagnostics::error(format!("Framecap capture could not start: {error}"));
    }
}

/// Opens a session lasting `duration` and spawns a thread that ends it and calls `save`; returns `false` if one is already running.
fn start_with(
    manager: &'static CaptureManager,
    duration: Duration,
    save: impl FnOnce(&Session, &[Frame]) -> io::Result<PathBuf> + Send + 'static,
) -> io::Result<bool> {
    let mut slot = manager.session.lock();
    if slot.is_some() {
        return Ok(false);
    }
    let start = Instant::now();
    manager.skipped_entries.store(0, Ordering::Relaxed);
    manager.trace_skipped_entries.store(0, Ordering::Relaxed);
    let session = Arc::new(Session {
        start,
        end: start + duration,
        unix_start_ns: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_nanos()),
        frames: Mutex::new(Vec::with_capacity(MAX_FRAMES)),
        acquisitions: Mutex::new(HashMap::new()),
        started: AtomicU64::new(0),
        dropped: AtomicU64::new(0),
        skipped_entries: AtomicU64::new(0),
        incomplete: AtomicU64::new(0),
        acquisition_dropped: AtomicU64::new(0),
        acquisition_failed: AtomicU64::new(0),
        trace: TraceBuffer {
            events: Mutex::new(Vec::with_capacity(MAX_TRACE_EVENTS)),
            ..TraceBuffer::default()
        },
    });
    *slot = Some(Arc::clone(&session));
    manager.active.store(true, Ordering::Release);
    if let Err(error) = std::thread::Builder::new()
        .name("framecap-capture".to_owned())
        .spawn(move || {
            while Instant::now() < session.end {
                std::thread::sleep(session.end.saturating_duration_since(Instant::now()));
            }
            manager.active.store(false, Ordering::Release);
            let admission = manager.session.lock();
            let frames = std::mem::take(&mut *session.frames.lock());
            session.skipped_entries.store(
                manager.skipped_entries.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            session.trace.skipped_entries.store(
                manager.trace_skipped_entries.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            session.incomplete.store(
                session
                    .started
                    .load(Ordering::Relaxed)
                    .saturating_sub(frames.len() as u64)
                    .saturating_sub(session.dropped.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
            drop(admission);
            match save(&session, &frames) {
                Ok(path) => diagnostics::info(format!(
                    "Framecap capture saved: {} ({} frames, {} dropped)",
                    path.display(),
                    frames.len(),
                    session.dropped.load(Ordering::Relaxed)
                )),
                Err(error) => {
                    diagnostics::error(format!("Framecap capture export failed: {error}"))
                }
            }
            *manager.session.lock() = None;
        })
    {
        manager.active.store(false, Ordering::Release);
        *slot = None;
        return Err(error);
    }
    Ok(true)
}

/// Starts recording a presentation without blocking on the capture buffer.
pub fn frame(queue: usize) -> Option<CaptureFrame> {
    frame_from(manager(), queue)
}

/// Starts timing an image acquisition if a session is active, without blocking on the session lock.
pub(crate) fn acquire() -> Option<CaptureAcquire> {
    let manager = manager();
    if !manager.active.load(Ordering::Acquire) {
        return None;
    }
    let Some(slot) = manager.session.try_lock() else {
        manager.skipped_entries.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let session = slot.as_ref()?.clone();
    let start = Instant::now();
    (start < session.end).then_some(CaptureAcquire { session, start })
}

/// Starts tracing a Vulkan call on the global manager, if a session is active.
pub(crate) fn trace(
    kind: &'static str,
    object: u64,
    related: u64,
    count: u32,
    timeout_ns: Option<u64>,
) -> Option<TraceStart> {
    trace_from(manager(), kind, object, related, count, timeout_ns)
}

/// Starts tracing a call against `manager`, recording the calling thread ID; skips without blocking if the session lock is busy.
fn trace_from(
    manager: &CaptureManager,
    kind: &'static str,
    object: u64,
    related: u64,
    count: u32,
    timeout_ns: Option<u64>,
) -> Option<TraceStart> {
    if !manager.active.load(Ordering::Acquire) {
        return None;
    }
    let start = Instant::now();
    let Some(slot) = manager.session.try_lock() else {
        manager
            .trace_skipped_entries
            .fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let session = slot.as_ref()?.clone();
    if start >= session.end {
        return None;
    }
    session.trace.started.fetch_add(1, Ordering::Relaxed);
    drop(slot);
    let thread_id = TRACE_THREAD_ID
        .try_with(|id| *id)
        .unwrap_or_else(|_| unsafe { libc::syscall(libc::SYS_gettid) } as i64);
    if thread_id <= 0 {
        static ERROR_LOGGED: AtomicBool = AtomicBool::new(false);
        if !ERROR_LOGGED.swap(true, Ordering::Relaxed) {
            diagnostics::error("Framecap trace could not obtain the thread ID");
        }
    }
    Some(TraceStart {
        session,
        kind,
        thread_id,
        object,
        related,
        count,
        timeout_ns,
        start,
    })
}

impl TraceStart {
    /// Records the call's end time and result, dropping the event if the session ended or the buffer is full or locked.
    pub(crate) fn finish(self, result: i32) {
        let end = Instant::now();
        if end >= self.session.end {
            return;
        }
        let Some(mut events) = self.session.trace.events.try_lock() else {
            self.session.trace.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if Instant::now() >= self.session.end {
            return;
        }
        if events.len() >= MAX_TRACE_EVENTS {
            self.session.trace.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        events.push(TraceEvent {
            kind: self.kind,
            thread_id: self.thread_id,
            object: self.object,
            related: self.related,
            count: self.count,
            timeout_ns: self.timeout_ns,
            start: self.start,
            end,
            result,
        });
    }
}

/// Discards pending acquisitions of a destroyed swapchain so they cannot match later frames.
pub(crate) fn retire_swapchain(device: u64, swapchain: u64) {
    let manager = manager();
    if !manager.active.load(Ordering::Acquire) {
        return;
    }
    if let Some(session) = manager.session.lock().as_ref() {
        session
            .acquisitions
            .lock()
            .retain(|(owner, chain, _), _| *owner != device || *chain != swapchain);
    }
}

impl CaptureAcquire {
    /// Stores the acquisition keyed by device, swapchain and image, counting failures and drops instead of blocking.
    pub(crate) fn finish(self, device: u64, swapchain: u64, image: Option<u32>, result: i32) {
        let end = Instant::now();
        if end >= self.session.end {
            return;
        }
        let Some(image) = image else {
            self.session
                .acquisition_failed
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        let Some(mut acquisitions) = self.session.acquisitions.try_lock() else {
            self.session
                .acquisition_dropped
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        let key = (device, swapchain, image);
        if acquisitions.len() >= MAX_FRAMES && !acquisitions.contains_key(&key) {
            self.session
                .acquisition_dropped
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        acquisitions.insert(
            key,
            Acquisition {
                start: self.start,
                end,
                result,
            },
        );
    }
}

/// Begins a frame record for `queue` on `manager`'s active session, or `None` if inactive, expired or contended.
fn frame_from(manager: &CaptureManager, queue: usize) -> Option<CaptureFrame> {
    if !manager.active.load(Ordering::Acquire) {
        return None;
    }
    let Some(slot) = manager.session.try_lock() else {
        manager.skipped_entries.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let session = slot.as_ref()?.clone();
    let entry = Instant::now();
    if entry >= session.end {
        return None;
    }
    session.started.fetch_add(1, Ordering::Relaxed);
    Some(CaptureFrame {
        session,
        frame: Frame {
            queue,
            entry: Some(entry),
            ..Frame::default()
        },
    })
}

impl CaptureFrame {
    /// Builds a frame attached to a standalone throwaway session for tests.
    #[cfg(test)]
    pub fn for_test() -> Self {
        let start = Instant::now();
        Self {
            session: Arc::new(Session {
                start,
                end: start + CAPTURE_DURATION,
                unix_start_ns: None,
                frames: Mutex::new(Vec::new()),
                acquisitions: Mutex::new(HashMap::new()),
                started: AtomicU64::new(1),
                dropped: AtomicU64::new(0),
                skipped_entries: AtomicU64::new(0),
                incomplete: AtomicU64::new(0),
                acquisition_dropped: AtomicU64::new(0),
                acquisition_failed: AtomicU64::new(0),
                trace: TraceBuffer::default(),
            }),
            frame: Frame::default(),
        }
    }

    /// Binds the frame to its swapchain image and copies in the matching acquisition timing, consuming it.
    pub(crate) fn associate_acquisition(&mut self, device: u64, swapchain: u64, image: u32) {
        self.frame.swapchain = swapchain;
        self.frame.image_index = Some(image);
        let Some(mut acquisitions) = self.session.acquisitions.try_lock() else {
            self.session
                .acquisition_dropped
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        if let Some(acquisition) = acquisitions.remove(&(device, swapchain, image)) {
            self.frame.acquire_start = Some(acquisition.start);
            self.frame.acquire_end = Some(acquisition.end);
            self.frame.acquire_result = Some(acquisition.result);
        }
    }

    /// Records the calibration status, attaching the sample only if it is newer than the session start and under 500 ms old.
    pub(crate) fn record_calibration(&mut self, state: Option<CalibrationState>) {
        self.frame.gpu_calibration_status = Some(match state {
            None => "unavailable",
            Some(CalibrationState::Waiting) => "waiting",
            Some(CalibrationState::Error) => "error",
            Some(CalibrationState::Busy) => "busy",
            Some(CalibrationState::Ready(sample)) => {
                if sample.reference_instant < self.session.start {
                    "waiting"
                } else if sample.reference_instant.elapsed() > Duration::from_millis(500) {
                    "stale"
                } else {
                    self.frame.gpu_calibration = Some(sample);
                    "ready"
                }
            }
        });
    }

    /// Stores the present result and pushes the frame into the session, counting it as dropped if the buffer is full or locked.
    pub fn finish(mut self, result: i32) {
        let now = Instant::now();
        self.frame.present_result = Some(result);
        if now >= self.session.end {
            return;
        }
        if let Some(mut frames) = self.session.frames.try_lock() {
            if Instant::now() >= self.session.end {
                return;
            }
            if frames.len() < MAX_FRAMES {
                frames.push(self.frame);
                return;
            }
        }
        self.session.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

/// Resolves the capture output directory: `$XDG_STATE_HOME/framecap`, `~/.local/state/framecap`, or a per-uid temp directory.
fn state_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("XDG_STATE_HOME").filter(|path| !path.is_empty()) {
        return PathBuf::from(path).join("framecap");
    }
    if let Some(home) = std::env::var_os("HOME").filter(|path| !path.is_empty()) {
        return PathBuf::from(home).join(".local/state/framecap");
    }
    std::env::temp_dir().join(format!("framecap-{}", unsafe { libc::getuid() }))
}

/// Writes the frame CSV and its companion trace CSV as new owner-only files, returning the frame CSV path.
fn save(session: &Session, frames: &[Frame]) -> io::Result<PathBuf> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let directory = state_directory();
    fs::create_dir_all(&directory)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_nanos();
    let path = directory.join(format!(
        "capture-{}-{timestamp}-{}.csv",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)?;
    let mut writer = BufWriter::new(file);
    write_capture(&mut writer, session, frames)?;
    writer.flush()?;
    let trace_path = directory.join(format!(
        "trace-{}",
        path.file_name()
            .expect("capture path has a file name")
            .to_string_lossy()
    ));
    let trace_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&trace_path)?;
    let mut trace_writer = BufWriter::new(trace_file);
    write_trace(&mut trace_writer, session)?;
    trace_writer.flush()?;
    diagnostics::info(format!("Framecap trace saved: {}", trace_path.display()));
    Ok(path)
}

/// Writes the trace CSV: a summary comment line, a header, then one row per traced call.
fn write_trace(writer: &mut impl Write, session: &Session) -> io::Result<()> {
    let events = session.trace.events.lock();
    let dropped = session.trace.dropped.load(Ordering::Relaxed);
    let incomplete = session
        .trace
        .started
        .load(Ordering::Relaxed)
        .saturating_sub(events.len() as u64)
        .saturating_sub(dropped);
    writeln!(
        writer,
        "# duration_ns={},start_unix_ns={},events={},dropped={},skipped_entries={},incomplete={}",
        session.end.duration_since(session.start).as_nanos(),
        value(session.unix_start_ns),
        events.len(),
        dropped,
        session.trace.skipped_entries.load(Ordering::Relaxed),
        incomplete
    )?;
    writeln!(
        writer,
        "kind,thread_id,object,related,count,timeout_ns,start_ns,end_ns,result"
    )?;
    for event in events.iter() {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{}",
            event.kind,
            event.thread_id,
            event.object,
            event.related,
            event.count,
            value(event.timeout_ns),
            offset(session.start, Some(event.start)),
            offset(session.start, Some(event.end)),
            event.result
        )?;
    }
    Ok(())
}

/// Formats a timestamp as signed nanoseconds relative to `start`, or an empty string if absent.
fn offset(start: Instant, timestamp: Option<Instant>) -> String {
    timestamp.map_or_else(String::new, |time| {
        if time >= start {
            time.duration_since(start).as_nanos().to_string()
        } else {
            format!("-{}", start.duration_since(time).as_nanos())
        }
    })
}

/// Formats an optional value for CSV, using an empty string for `None`.
fn value<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

/// Converts a calibration's CPU clock reading to nanoseconds relative to the session start.
fn calibration_cpu_offset(start: Instant, calibration: Calibration) -> i128 {
    let reference = if calibration.reference_instant >= start {
        calibration
            .reference_instant
            .duration_since(start)
            .as_nanos() as i128
    } else {
        -(start
            .duration_since(calibration.reference_instant)
            .as_nanos() as i128)
    };
    reference + i128::from(calibration.cpu_ns) - i128::from(calibration.reference_cpu_ns)
}

/// Resolves a GPU marker sample into a status string and its timestamp, falling back to the recorded status when no sample exists.
fn marker_fields(
    sample: Option<&Sample>,
    status: Option<&'static str>,
) -> (Option<&'static str>, Option<Timestamp>) {
    match sample {
        Some(sample) => match *sample.lock() {
            SampleResult::Ready(timestamp) => (Some("ready"), Some(timestamp)),
            SampleResult::Pending => (Some("pending"), None),
            SampleResult::Error => (Some("error"), None),
        },
        None => (status, None),
    }
}

/// Writes the frame CSV: a summary comment line, a header, then one row per captured frame.
fn write_capture(writer: &mut impl Write, session: &Session, frames: &[Frame]) -> io::Result<()> {
    writeln!(
        writer,
        "# duration_ns={},start_unix_ns={},frames={},dropped={},skipped_entries={},incomplete={},acquisition_dropped={},acquisition_failed={}",
        session.end.duration_since(session.start).as_nanos(),
        value(session.unix_start_ns),
        frames.len(),
        session.dropped.load(Ordering::Relaxed),
        session.skipped_entries.load(Ordering::Relaxed),
        session.incomplete.load(Ordering::Relaxed),
        session.acquisition_dropped.load(Ordering::Relaxed),
        session.acquisition_failed.load(Ordering::Relaxed)
    )?;
    writeln!(writer, "queue,swapchain,swapchain_count,initial_present_mode,entry_ns,lookup_done_ns,context_done_ns,schedule_ns,previous_deadline_ns,previous_target_fps,target_fps,deadline_ns,wait_start_ns,wake_ns,wait_failed,work_lock_start_ns,work_lock_acquired_ns,metrics_ns,auto_ns,overlay_ns,overlay_refreshes,overlay_mode,overlay_renderer,auto_mode,auto_evaluations,target_after_fps,gpu_load_percent,gpu_timestamp_ns,submit_start_ns,submit_end_ns,submit_result,present_start_ns,present_end_ns,present_result,metrics_lock_start_ns,metrics_lock_acquired_ns,gpu_marker_status,gpu_marker_ticks,gpu_marker_period_ns,gpu_marker_valid_bits,gpu_marker_submit_start_ns,gpu_marker_submit_end_ns,image_index,acquire_start_ns,acquire_end_ns,acquire_result,present_mode_override,present_id_khr,present_id_google,gpu_calibration_status,gpu_calibration_cpu_ns,gpu_calibration_gpu_ticks,gpu_calibration_max_deviation_ns,gpu_calibration_reference_ns,gpu_calibration_reference_uncertainty_ns,gpu_overlay_marker_status,gpu_overlay_marker_ticks,gpu_overlay_marker_period_ns,gpu_overlay_marker_valid_bits,gpu_overlay_marker_prepare_start_ns,gpu_overlay_marker_prepare_end_ns")?;
    for frame in frames {
        let t = |time| offset(session.start, time);
        let (marker_status, timestamp) =
            marker_fields(frame.gpu_marker.as_ref(), frame.gpu_marker_status);
        let (overlay_marker_status, overlay_timestamp) = marker_fields(
            frame.gpu_overlay_marker.as_ref(),
            frame.gpu_overlay_marker_status,
        );
        write!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            frame.queue, frame.swapchain, frame.swapchain_count, value(frame.present_mode),
            t(frame.entry), t(frame.lookup_done), t(frame.context_done), t(frame.schedule_at),
            t(frame.previous_deadline), value(frame.previous_target), value(frame.target),
            t(frame.deadline), t(frame.wait_start), t(frame.wake), frame.wait_failed,
            t(frame.work_lock_start), t(frame.work_lock_acquired), frame.metrics_ns,
            frame.auto_ns, frame.overlay_ns, frame.overlay_refreshes, value(frame.overlay_mode),
            value(frame.overlay_renderer), value(frame.auto_mode), frame.auto_evaluations, value(frame.target_after),
            value(frame.gpu_load), t(frame.gpu_timestamp), t(frame.submit_start), t(frame.submit_end),
            value(frame.submit_result), t(frame.present_start), t(frame.present_end),
            value(frame.present_result), t(frame.metrics_lock_start), t(frame.metrics_lock_acquired),
            value(marker_status), value(timestamp.map(|result| result.ticks)),
            value(timestamp.map(|result| result.period_ns)),
            value(timestamp.map(|result| result.valid_bits)),
            t(frame.gpu_marker_submit_start), t(frame.gpu_marker_submit_end),
            value(frame.image_index), t(frame.acquire_start), t(frame.acquire_end),
            value(frame.acquire_result), value(frame.present_mode_override),
            value(frame.present_id), value(frame.google_present_id)
        )?;
        let calibration = frame.gpu_calibration;
        writeln!(
            writer,
            ",{},{},{},{},{},{},{},{},{},{},{},{}",
            value(frame.gpu_calibration_status),
            value(calibration.map(|sample| calibration_cpu_offset(session.start, sample))),
            value(calibration.map(|sample| sample.gpu_ticks)),
            value(calibration.map(|sample| sample.max_deviation_ns)),
            t(calibration.map(|sample| sample.reference_instant)),
            value(calibration.map(|sample| sample.reference_uncertainty_ns)),
            value(overlay_marker_status),
            value(overlay_timestamp.map(|result| result.ticks)),
            value(overlay_timestamp.map(|result| result.period_ns)),
            value(overlay_timestamp.map(|result| result.valid_bits)),
            t(frame.gpu_overlay_marker_prepare_start),
            t(frame.gpu_overlay_marker_prepare_end)
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::gpu_capture::Timestamp;

    #[test]
    fn trace_exports_completed_events_and_accounts_for_losses() {
        let session = CaptureFrame::for_test().session;
        let manager = CaptureManager::default();
        *manager.session.lock() = Some(Arc::clone(&session));
        manager.active.store(true, Ordering::Release);
        trace_from(&manager, "wait_fences", 42, 17, 2, Some(99))
            .unwrap()
            .finish(2);
        let guard = session.trace.events.lock();
        trace_from(&manager, "queue_submit", 42, 0, 1, None)
            .unwrap()
            .finish(0);
        drop(guard);
        let pending = trace_from(&manager, "present", 42, 0, 0, None).unwrap();
        let mut output = Vec::new();
        write_trace(&mut output, &session).unwrap();
        let text = String::from_utf8(output).unwrap();
        let mut lines = text.lines();
        let metadata = lines.next().unwrap();
        assert!(metadata.contains("events=1,dropped=1,skipped_entries=0,incomplete=1"));
        let header: Vec<_> = lines.next().unwrap().split(',').collect();
        let event: Vec<_> = lines.next().unwrap().split(',').collect();
        assert_eq!(header.len(), event.len());
        assert_eq!(event[0], "wait_fences");
        assert!(event[1].parse::<i64>().unwrap() > 0);
        assert_eq!(&event[2..6], &["42", "17", "2", "99"]);
        assert!(event[6].parse::<u64>().unwrap() <= event[7].parse::<u64>().unwrap());
        assert_eq!(event[8], "2");
        drop(pending);
        manager.active.store(false, Ordering::Release);
        assert!(trace_from(&manager, "present", 42, 0, 0, None).is_none());
    }

    #[test]
    fn csv_distinguishes_completed_and_pending_gpu_markers() {
        let start = Instant::now();
        let session = Session {
            start,
            end: start + CAPTURE_DURATION,
            unix_start_ns: None,
            frames: Mutex::new(Vec::new()),
            acquisitions: Mutex::new(HashMap::new()),
            started: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            skipped_entries: AtomicU64::new(0),
            incomplete: AtomicU64::new(0),
            acquisition_dropped: AtomicU64::new(0),
            acquisition_failed: AtomicU64::new(0),
            trace: TraceBuffer::default(),
        };
        let ready = Frame {
            gpu_marker_status: Some("pending"),
            gpu_marker: Some(Arc::new(Mutex::new(SampleResult::Ready(Timestamp {
                ticks: 12345,
                period_ns: 10.0,
                valid_bits: 64,
            })))),
            gpu_overlay_marker_status: Some("pending"),
            gpu_overlay_marker: Some(Arc::new(Mutex::new(SampleResult::Ready(Timestamp {
                ticks: 12400,
                period_ns: 10.0,
                valid_bits: 64,
            })))),
            gpu_overlay_marker_prepare_start: Some(start),
            gpu_overlay_marker_prepare_end: Some(start + Duration::from_micros(20)),
            image_index: Some(3),
            acquire_start: Some(start),
            acquire_end: Some(start + Duration::from_millis(2)),
            acquire_result: Some(0),
            present_mode_override: Some(1),
            present_id: Some(42),
            google_present_id: Some(17),
            gpu_calibration_status: Some("ready"),
            gpu_calibration: Some(Calibration {
                gpu_ticks: 67890,
                cpu_ns: 900,
                max_deviation_ns: 20,
                reference_cpu_ns: 1000,
                reference_instant: start + Duration::from_micros(1),
                reference_uncertainty_ns: 4,
            }),
            ..Frame::default()
        };
        let pending = Frame {
            gpu_marker_status: Some("pending"),
            gpu_marker: Some(Arc::new(Mutex::new(SampleResult::Pending))),
            gpu_overlay_marker_status: Some("pending"),
            gpu_overlay_marker: Some(Arc::new(Mutex::new(SampleResult::Pending))),
            ..Frame::default()
        };
        let failed = Frame {
            gpu_overlay_marker: Some(Arc::new(Mutex::new(SampleResult::Error))),
            ..Frame::default()
        };
        let no_overlay = Frame {
            gpu_overlay_marker_status: Some("no_overlay"),
            ..Frame::default()
        };
        let mut output = Vec::new();
        write_capture(&mut output, &session, &[ready, pending, failed, no_overlay]).unwrap();
        let text = String::from_utf8(output).unwrap();
        let mut lines = text.lines().skip(1);
        let header: Vec<_> = lines.next().unwrap().split(',').collect();
        let ready: Vec<_> = lines.next().unwrap().split(',').collect();
        let pending: Vec<_> = lines.next().unwrap().split(',').collect();
        let failed: Vec<_> = lines.next().unwrap().split(',').collect();
        let no_overlay: Vec<_> = lines.next().unwrap().split(',').collect();
        assert_eq!(header.len(), ready.len());
        assert_eq!(header.len(), pending.len());
        assert_eq!(header.len(), failed.len());
        assert_eq!(header.len(), no_overlay.len());
        let column = |name| header.iter().position(|column| *column == name).unwrap();
        assert_eq!(ready[column("gpu_marker_status")], "ready");
        assert_eq!(ready[column("gpu_marker_ticks")], "12345");
        assert_eq!(ready[column("gpu_marker_period_ns")], "10");
        assert_eq!(ready[column("gpu_marker_valid_bits")], "64");
        assert_eq!(ready[column("gpu_overlay_marker_status")], "ready");
        assert_eq!(ready[column("gpu_overlay_marker_ticks")], "12400");
        assert_eq!(ready[column("gpu_overlay_marker_period_ns")], "10");
        assert_eq!(ready[column("gpu_overlay_marker_valid_bits")], "64");
        assert_eq!(ready[column("gpu_overlay_marker_prepare_start_ns")], "0");
        assert_eq!(ready[column("gpu_overlay_marker_prepare_end_ns")], "20000");
        assert_eq!(ready[column("image_index")], "3");
        assert_eq!(ready[column("acquire_start_ns")], "0");
        assert_eq!(ready[column("acquire_end_ns")], "2000000");
        assert_eq!(ready[column("acquire_result")], "0");
        assert_eq!(ready[column("present_mode_override")], "1");
        assert_eq!(ready[column("present_id_khr")], "42");
        assert_eq!(ready[column("present_id_google")], "17");
        assert_eq!(ready[column("gpu_calibration_status")], "ready");
        assert_eq!(ready[column("gpu_calibration_cpu_ns")], "900");
        assert_eq!(ready[column("gpu_calibration_gpu_ticks")], "67890");
        assert_eq!(ready[column("gpu_calibration_max_deviation_ns")], "20");
        assert_eq!(ready[column("gpu_calibration_reference_ns")], "1000");
        assert_eq!(
            ready[column("gpu_calibration_reference_uncertainty_ns")],
            "4"
        );
        assert_eq!(pending[column("gpu_marker_status")], "pending");
        assert!(pending[column("gpu_marker_ticks")].is_empty());
        assert_eq!(pending[column("gpu_overlay_marker_status")], "pending");
        assert!(pending[column("gpu_overlay_marker_ticks")].is_empty());
        assert_eq!(failed[column("gpu_overlay_marker_status")], "error");
        assert!(failed[column("gpu_overlay_marker_ticks")].is_empty());
        assert_eq!(
            no_overlay[column("gpu_overlay_marker_status")],
            "no_overlay"
        );
        assert!(no_overlay[column("gpu_overlay_marker_ticks")].is_empty());
        assert!(pending[column("acquire_start_ns")].is_empty());
    }

    #[test]
    fn calibration_offsets_handle_before_and_after_reference() {
        let start = Instant::now();
        let mut sample = Calibration {
            gpu_ticks: 0,
            cpu_ns: 1200,
            max_deviation_ns: 10,
            reference_cpu_ns: 1000,
            reference_instant: start + Duration::from_micros(1),
            reference_uncertainty_ns: 1,
        };
        assert_eq!(calibration_cpu_offset(start, sample), 1200);
        sample.cpu_ns = 600;
        assert_eq!(calibration_cpu_offset(start, sample), 600);
        sample.reference_instant = start - Duration::from_micros(1);
        assert_eq!(calibration_cpu_offset(start, sample), -1400);
    }

    #[test]
    fn calibration_records_only_recent_samples_from_this_capture() {
        let mut frame = CaptureFrame::for_test();
        let now = Instant::now();
        let mut sample = Calibration {
            gpu_ticks: 1,
            cpu_ns: 1,
            max_deviation_ns: 10,
            reference_cpu_ns: 1,
            reference_instant: now,
            reference_uncertainty_ns: 1,
        };
        frame.record_calibration(None);
        assert_eq!(frame.frame.gpu_calibration_status, Some("unavailable"));
        frame.record_calibration(Some(CalibrationState::Waiting));
        assert_eq!(frame.frame.gpu_calibration_status, Some("waiting"));
        frame.record_calibration(Some(CalibrationState::Busy));
        assert_eq!(frame.frame.gpu_calibration_status, Some("busy"));
        frame.record_calibration(Some(CalibrationState::Error));
        assert_eq!(frame.frame.gpu_calibration_status, Some("error"));
        sample.reference_instant = frame.session.start - Duration::from_nanos(1);
        frame.record_calibration(Some(CalibrationState::Ready(sample)));
        assert_eq!(frame.frame.gpu_calibration_status, Some("waiting"));
        frame.session = Arc::new(Session {
            start: now - Duration::from_secs(2),
            end: now + CAPTURE_DURATION,
            unix_start_ns: None,
            frames: Mutex::new(Vec::new()),
            acquisitions: Mutex::new(HashMap::new()),
            started: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            skipped_entries: AtomicU64::new(0),
            incomplete: AtomicU64::new(0),
            acquisition_dropped: AtomicU64::new(0),
            acquisition_failed: AtomicU64::new(0),
            trace: TraceBuffer::default(),
        });
        sample.reference_instant = now - Duration::from_secs(1);
        frame.record_calibration(Some(CalibrationState::Ready(sample)));
        assert_eq!(frame.frame.gpu_calibration_status, Some("stale"));
        assert!(frame.frame.gpu_calibration.is_none());
        sample.reference_instant = Instant::now();
        frame.record_calibration(Some(CalibrationState::Ready(sample)));
        assert_eq!(frame.frame.gpu_calibration_status, Some("ready"));
        assert!(frame.frame.gpu_calibration.is_some());
    }

    #[test]
    fn capture_collects_in_memory_and_exports_only_after_deadline() {
        static MANAGER: OnceLock<CaptureManager> = OnceLock::new();
        let manager = MANAGER.get_or_init(CaptureManager::default);
        let path = std::env::temp_dir().join(format!(
            "framecap-capture-test-{}-{:?}.csv",
            std::process::id(),
            std::thread::current().id()
        ));
        let export_path = path.clone();
        let export_times = Arc::new(Mutex::new(None));
        let recorded_times = Arc::clone(&export_times);
        let started = Instant::now();
        let duration = Duration::from_secs(1);
        assert!(start_with(manager, duration, move |session, frames| {
            *recorded_times.lock() = Some((Instant::now(), session.end));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&export_path)?;
            write_capture(&mut file, session, frames)?;
            Ok(export_path)
        })
        .unwrap());
        assert!(!start_with(manager, duration, save).unwrap());
        let end = manager.session.lock().as_ref().unwrap().end;

        let mut frame = frame_from(manager, 42).unwrap();
        frame.frame.target = Some(60.0);
        frame.frame.deadline = Some(Instant::now() + Duration::from_millis(10));
        frame.frame.present_start = Some(Instant::now());
        frame.frame.present_end = Some(Instant::now());
        frame.frame.metrics_lock_start = Some(Instant::now());
        frame.frame.metrics_lock_acquired = Some(Instant::now());
        frame.finish(0);
        let incomplete = frame_from(manager, 43).unwrap();
        {
            let _slot = manager.session.lock();
            assert!(frame_from(manager, 44).is_none());
        }

        std::thread::sleep(
            (started + duration - Duration::from_millis(200))
                .saturating_duration_since(Instant::now()),
        );
        if Instant::now() < end {
            assert!(!path.exists(), "capture was written before its deadline");
        }
        let timeout = Instant::now() + Duration::from_secs(3);
        while manager.session.lock().is_some() && Instant::now() < timeout {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(manager.session.lock().is_none(), "capture did not finish");
        let (exported, end) = export_times.lock().unwrap();
        assert!(exported >= end, "export started before its deadline");
        let csv = std::fs::read_to_string(&path).unwrap();
        let mut lines = csv.lines();
        let metadata = lines.next().unwrap();
        assert!(metadata.contains("duration_ns=1000000000"));
        assert!(metadata.contains("skipped_entries=1"));
        assert!(metadata.contains("incomplete=1"));
        let header = lines.next().unwrap().split(',').collect::<Vec<_>>();
        let row = lines.next().unwrap().split(',').collect::<Vec<_>>();
        assert_eq!(header.len(), row.len());
        assert_eq!(
            row[header.iter().position(|name| *name == "queue").unwrap()],
            "42"
        );
        assert_eq!(
            row[header
                .iter()
                .position(|name| *name == "target_fps")
                .unwrap()],
            "60"
        );
        assert!(!row[header
            .iter()
            .position(|name| *name == "deadline_ns")
            .unwrap()]
        .is_empty());
        assert_eq!(
            row[header
                .iter()
                .position(|name| *name == "present_result")
                .unwrap()],
            "0"
        );
        for name in ["metrics_lock_start_ns", "metrics_lock_acquired_ns"] {
            assert!(!row[header.iter().position(|column| *column == name).unwrap()].is_empty());
        }
        assert!(lines.next().is_none());
        assert!(frame_from(manager, 42).is_none());
        incomplete.finish(0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn late_frames_are_excluded_and_full_buffer_is_counted() {
        let start = Instant::now();
        let session = Arc::new(Session {
            start,
            end: start,
            unix_start_ns: None,
            frames: Mutex::new(Vec::new()),
            acquisitions: Mutex::new(HashMap::new()),
            started: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            skipped_entries: AtomicU64::new(0),
            incomplete: AtomicU64::new(0),
            acquisition_dropped: AtomicU64::new(0),
            acquisition_failed: AtomicU64::new(0),
            trace: TraceBuffer::default(),
        });
        CaptureFrame {
            session: Arc::clone(&session),
            frame: Frame::default(),
        }
        .finish(0);
        assert_eq!(session.dropped.load(Ordering::Relaxed), 0);
        assert!(session.frames.lock().is_empty());

        let full = Arc::new(Session {
            start,
            end: start + Duration::from_secs(60),
            unix_start_ns: None,
            frames: Mutex::new((0..MAX_FRAMES).map(|_| Frame::default()).collect()),
            acquisitions: Mutex::new(HashMap::new()),
            started: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            skipped_entries: AtomicU64::new(0),
            incomplete: AtomicU64::new(0),
            acquisition_dropped: AtomicU64::new(0),
            acquisition_failed: AtomicU64::new(0),
            trace: TraceBuffer::default(),
        });
        CaptureFrame {
            session: Arc::clone(&full),
            frame: Frame::default(),
        }
        .finish(0);
        assert_eq!(full.dropped.load(Ordering::Relaxed), 1);
        assert_eq!(full.frames.lock().len(), MAX_FRAMES);
    }

    #[test]
    fn acquisitions_match_device_swapchain_and_image_once() {
        let mut frame = CaptureFrame::for_test();
        let session = Arc::clone(&frame.session);
        let start = Instant::now();
        CaptureAcquire {
            session: Arc::clone(&session),
            start,
        }
        .finish(1, 2, Some(3), 0);
        frame.associate_acquisition(1, 2, 4);
        assert!(frame.frame.acquire_start.is_none());
        frame.associate_acquisition(9, 2, 3);
        assert!(frame.frame.acquire_start.is_none());
        frame.associate_acquisition(1, 2, 3);
        assert_eq!(frame.frame.image_index, Some(3));
        assert_eq!(frame.frame.acquire_start, Some(start));
        assert_eq!(frame.frame.acquire_result, Some(0));
        assert!(frame.frame.acquire_end >= frame.frame.acquire_start);
        let mut next = CaptureFrame {
            session: Arc::clone(&session),
            frame: Frame::default(),
        };
        next.associate_acquisition(1, 2, 3);
        assert!(next.frame.acquire_start.is_none());
        CaptureAcquire {
            session: Arc::clone(&session),
            start,
        }
        .finish(1, 2, None, 2);
        assert_eq!(session.acquisition_failed.load(Ordering::Relaxed), 1);
    }
}
