use super::canvas::FrameKey;
use super::{OverlayFeedback, OverlayFrame, OverlayMode};
use crate::layer::config::{AutoMetric, AutoTarget, Hotkey};
use crate::layer::lib::MetricsSnapshot;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};

/// Frame data copied out of borrowed references so it can cross to the worker thread.
struct OwnedFrame {
    metrics: MetricsSnapshot,
    mode: OverlayMode,
    target_fps: Option<f32>,
    auto_algorithm_index: Option<usize>,
    auto_algorithm_label: Option<String>,
    hysteresis: Option<f32>,
    auto_target: Option<(AutoMetric, AutoTarget)>,
    hotkeys: [Hotkey; 4],
    manual_step_fps: f32,
    feedback: OverlayFeedback,
}

impl OwnedFrame {
    /// Clones the borrowed frame into an owned snapshot.
    fn new(frame: &OverlayFrame<'_>) -> Self {
        Self {
            metrics: frame.metrics.clone(),
            mode: frame.mode,
            target_fps: frame.target_fps,
            auto_algorithm_index: frame.auto_algorithm_index,
            auto_algorithm_label: frame.auto_algorithm_label.map(str::to_owned),
            hysteresis: frame.hysteresis,
            auto_target: frame.auto_target,
            hotkeys: frame.hotkeys,
            manual_step_fps: frame.manual_step_fps,
            feedback: frame.feedback,
        }
    }

    /// Borrows the snapshot back as an [`OverlayFrame`].
    fn as_frame(&self) -> OverlayFrame<'_> {
        OverlayFrame {
            metrics: &self.metrics,
            mode: self.mode,
            target_fps: self.target_fps,
            auto_algorithm_index: self.auto_algorithm_index,
            auto_algorithm_label: self.auto_algorithm_label.as_deref(),
            hysteresis: self.hysteresis,
            auto_target: self.auto_target,
            hotkeys: self.hotkeys,
            manual_step_fps: self.manual_step_fps,
            feedback: self.feedback,
        }
    }
}

/// Request sent to the worker: the frame to prepare and the key identifying it.
pub(super) struct Preparation {
    pub(super) key: FrameKey,
    frame: OwnedFrame,
}

/// Worker thread that prepares overlay frames off the render path, keeping only the newest request.
pub(super) struct BackgroundPreparation<T> {
    pub(super) requests: Option<SyncSender<Preparation>>,
    pub(super) completed: Receiver<(FrameKey, OverlayMode, T)>,
    pub(super) thread: Option<JoinHandle<()>>,
    pub(super) requested: Option<FrameKey>,
    pub(super) failed: bool,
    pub(super) name: &'static str,
}

impl<T: Send + 'static> BackgroundPreparation<T> {
    /// Spawns the named worker thread, which runs `prepare` on each request, skipping stale ones.
    pub(super) fn new(
        name: &'static str,
        prepare: impl Fn(&OverlayFrame<'_>) -> T + Send + 'static,
    ) -> std::io::Result<Self> {
        let (requests, receiver) = mpsc::sync_channel::<Preparation>(1);
        let (sender, completed) = mpsc::channel();
        let thread = thread::Builder::new()
            .name(format!("framecap-{name}"))
            .spawn(move || {
                while let Ok(mut request) = receiver.recv() {
                    while let Ok(newer) = receiver.try_recv() {
                        request = newer;
                    }
                    let frame = request.frame.as_frame();
                    let mode = frame.mode;
                    let prepared = prepare(&frame);
                    if sender.send((request.key, mode, prepared)).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            requests: Some(requests),
            completed,
            thread: Some(thread),
            requested: None,
            failed: false,
            name,
        })
    }

    /// Collects finished results matching the current controls and queues the frame if not yet requested.
    pub(super) fn update(
        &mut self,
        frame: &OverlayFrame<'_>,
        prepared: &mut Option<(FrameKey, OverlayMode, T)>,
    ) {
        let key = FrameKey::new(frame);
        loop {
            match self.completed.try_recv() {
                Ok((result_key, mode, result)) => {
                    if result_key.same_controls(&key) {
                        *prepared = Some((result_key, mode, result));
                    } else if self.requested.as_ref() == Some(&result_key) {
                        self.requested = None;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.report_failure();
                    break;
                }
            }
        }
        if self.failed || self.requested.as_ref() == Some(&key) {
            return;
        }
        let request = Preparation {
            key: key.clone(),
            frame: OwnedFrame::new(frame),
        };
        match self
            .requests
            .as_ref()
            .expect("worker sender available")
            .try_send(request)
        {
            Ok(()) => self.requested = Some(key),
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => self.report_failure(),
        }
    }
}

impl<T> BackgroundPreparation<T> {
    /// Logs once that the worker has stopped and marks the preparation as failed.
    fn report_failure(&mut self) {
        if !self.failed {
            crate::layer::diagnostics::error(format!(
                "{} overlay preparation worker stopped unexpectedly",
                self.name
            ));
            self.failed = true;
        }
    }
}

impl<T> Drop for BackgroundPreparation<T> {
    /// Closes the request channel and joins the worker thread.
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                self.report_failure();
            }
        }
    }
}
