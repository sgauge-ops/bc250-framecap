use crate::layer::config::Hotkey;
use crate::layer::settings::OverlayRenderer;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

const FEEDBACK_VISIBLE_FOR: Duration = Duration::from_secs(1);

/// Amount of information shown by the overlay; discriminants are the persisted values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum OverlayMode {
    Off = 0,
    Fps = 1,
    Full = 2,
}

/// Thread-safe holder of the current [`OverlayMode`].
pub struct OverlayModeState(AtomicU8);

/// Thread-safe holder of the currently selected visual model.
pub struct OverlayRendererState(AtomicU8);

impl OverlayRendererState {
    /// Number of visual models; `toggle` cycles through the indexes `0..MODEL_COUNT`.
    const MODEL_COUNT: u8 = 3;

    /// Creates the state with `renderer` selected.
    pub fn new(renderer: OverlayRenderer) -> Self {
        Self(AtomicU8::new(match renderer {
            OverlayRenderer::Ascii => 0,
            OverlayRenderer::Modern => 1,
            OverlayRenderer::Helix => 2,
        }))
    }

    /// Returns the currently selected visual model.
    pub fn current(&self) -> OverlayRenderer {
        Self::from_index(self.0.load(Ordering::Relaxed))
    }

    /// Switches to the next visual model, wrapping around after the last one, and returns it.
    pub fn toggle(&self) -> OverlayRenderer {
        let previous = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |index| {
                Some((index + 1) % Self::MODEL_COUNT)
            })
            .expect("overlay renderer update must succeed");
        Self::from_index((previous + 1) % Self::MODEL_COUNT)
    }

    /// Converts a stored index back to its model; panics on an invalid index.
    fn from_index(index: u8) -> OverlayRenderer {
        match index {
            0 => OverlayRenderer::Ascii,
            1 => OverlayRenderer::Modern,
            2 => OverlayRenderer::Helix,
            _ => unreachable!("invalid overlay renderer"),
        }
    }
}

impl OverlayModeState {
    /// Creates the state from the raw startup mode value (0 = off, 1 = FPS, 2 = full).
    pub fn new(startup: u8) -> Self {
        Self(AtomicU8::new(startup))
    }

    /// Returns the current mode; panics if the stored value is invalid.
    pub fn mode(&self) -> OverlayMode {
        match self.0.load(Ordering::Relaxed) {
            0 => OverlayMode::Off,
            1 => OverlayMode::Fps,
            2 => OverlayMode::Full,
            value => panic!("invalid overlay mode: {value}"),
        }
    }

    /// Atomically advances Full -> FPS -> Off -> Full and returns the new mode.
    pub fn cycle(&self) -> OverlayMode {
        let previous = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |mode| match mode {
                2 => Some(1),
                1 => Some(0),
                0 => Some(2),
                _ => None,
            })
            .expect("overlay mode must be valid");
        match previous {
            2 => OverlayMode::Fps,
            1 => OverlayMode::Off,
            0 => OverlayMode::Full,
            _ => unreachable!(),
        }
    }
}

/// User-visible control events that trigger a brief overlay highlight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlayFeedbackEvent {
    HotkeyPressed(Hotkey),
    CapChanged,
    HysteresisChanged,
    TargetChanged,
}

/// Snapshot of which feedback highlights are currently visible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OverlayFeedback {
    pub hotkey: Option<Hotkey>,
    pub cap_changed: bool,
    pub hysteresis_changed: bool,
    pub target_changed: bool,
}

/// Time of the most recent occurrence of each feedback event.
#[derive(Default)]
struct OverlayFeedbackTimestamps {
    hotkey: Option<(Hotkey, Instant)>,
    cap: Option<Instant>,
    hysteresis: Option<Instant>,
    target: Option<Instant>,
}

/// Mutex-protected feedback timestamps shared between control threads and the renderer.
pub struct OverlayFeedbackState {
    events: parking_lot::Mutex<OverlayFeedbackTimestamps>,
}

impl OverlayFeedbackState {
    /// Creates an empty set of short-lived overlay feedback events.
    pub fn new() -> Self {
        Self {
            events: parking_lot::Mutex::new(OverlayFeedbackTimestamps::default()),
        }
    }

    /// Records a hotkey press or an effective control change.
    pub fn record(&self, event: OverlayFeedbackEvent) {
        let mut feedback = self.events.lock();
        let now = Instant::now();
        match event {
            OverlayFeedbackEvent::HotkeyPressed(hotkey) => {
                feedback.hotkey = Some((hotkey, now));
            }
            OverlayFeedbackEvent::CapChanged => feedback.cap = Some(now),
            OverlayFeedbackEvent::HysteresisChanged => feedback.hysteresis = Some(now),
            OverlayFeedbackEvent::TargetChanged => feedback.target = Some(now),
        }
    }

    /// Returns feedback events that remain visible at the supplied time.
    pub fn current(&self, now: Instant) -> OverlayFeedback {
        let Some(feedback) = self.events.try_lock() else {
            return OverlayFeedback::default();
        };
        let visible =
            |timestamp: Instant| now.saturating_duration_since(timestamp) <= FEEDBACK_VISIBLE_FOR;
        OverlayFeedback {
            hotkey: feedback
                .hotkey
                .filter(|(_, timestamp)| visible(*timestamp))
                .map(|(hotkey, _)| hotkey),
            cap_changed: feedback.cap.is_some_and(visible),
            hysteresis_changed: feedback.hysteresis.is_some_and(visible),
            target_changed: feedback.target.is_some_and(visible),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Checks that toggling visits every visual model once before wrapping around.
    fn renderer_toggle_cycles_through_every_model() {
        let state = OverlayRendererState::new(OverlayRenderer::Ascii);
        let mut visited = vec![state.current()];
        for _ in 1..OverlayRendererState::MODEL_COUNT {
            visited.push(state.toggle());
        }
        for (index, renderer) in visited.iter().enumerate() {
            assert!(
                !visited[..index].contains(renderer),
                "{renderer:?} repeated"
            );
        }
        assert_eq!(state.toggle(), OverlayRenderer::Ascii);
        assert_eq!(state.current(), OverlayRenderer::Ascii);
    }

    #[test]
    /// Checks independent expiry of the overlay feedback fields.
    fn feedback_fields_expire_after_one_second_independently() {
        let feedback = OverlayFeedbackState::new();
        let start = Instant::now();
        let later = start + Duration::from_millis(500);
        *feedback.events.lock() = OverlayFeedbackTimestamps {
            hotkey: Some((Hotkey::F6, start)),
            cap: Some(later),
            hysteresis: Some(later),
            target: Some(later),
        };
        let current = feedback.current(start + FEEDBACK_VISIBLE_FOR);
        assert_eq!(current.hotkey, Some(Hotkey::F6));
        assert!(current.cap_changed && current.hysteresis_changed && current.target_changed);
        let independently_expired =
            feedback.current(start + FEEDBACK_VISIBLE_FOR + Duration::from_millis(1));
        assert_eq!(independently_expired.hotkey, None);
        assert!(
            independently_expired.cap_changed
                && independently_expired.hysteresis_changed
                && independently_expired.target_changed
        );
        let expired = feedback.current(later + FEEDBACK_VISIBLE_FOR + Duration::from_millis(1));
        assert_eq!(expired, OverlayFeedback::default());
    }
}
