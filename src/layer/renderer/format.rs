//! Text formatting shared by every visual model.

use crate::layer::config::{AutoMetric, AutoTarget};
use crate::layer::lib::{MemoryTemperature, MetricsSnapshot};

/// Formats a memory temperature in °C (prefixed `>=` for a lower bound), or `--` if unavailable.
pub(super) fn format_memory_temperature(value: Option<MemoryTemperature>) -> String {
    value
        .filter(|value| value.celsius.is_finite())
        .map(|value| {
            let prefix = if value.lower_bound { ">=" } else { "" };
            format!("{prefix}{:.0}", value.celsius)
        })
        .unwrap_or_else(|| "--".to_owned())
}

/// Formats the displayed FPS value, or `COLLECT` while no measurement is available.
pub(super) fn fps_value(metrics: &MetricsSnapshot) -> String {
    let fps_value = metrics.fps();
    if fps_value.is_finite() && fps_value > 0.0 {
        format!("{fps_value:.0}")
    } else {
        "COLLECT".to_owned()
    }
}

/// Formats the frametime in ms (scientific notation for huge values), or `COLLECT` if unavailable.
pub(super) fn frametime_value(metrics: &MetricsSnapshot) -> String {
    let frametime = metrics.frametime_ms();
    if !frametime.is_finite() || frametime <= 0.0 {
        "COLLECT".to_owned()
    } else if frametime >= 10_000_000.0 {
        format!("{frametime:.2e} ms")
    } else {
        format!("{frametime:.2} ms")
    }
}

/// Formats a finite value with the given decimal precision, or `--` when missing or not finite.
pub(super) fn format_optional(value: Option<f32>, precision: usize) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.precision$}"))
        .unwrap_or_else(|| "--".to_owned())
}

/// Formats an FPS value without unnecessary trailing zeroes.
pub(super) fn format_fps(fps: f32) -> String {
    if fps.fract().abs() < 0.001 {
        format!("{fps:.0}")
    } else {
        format!("{fps:.1}")
    }
}

/// Formats the active AUTO target with its unit (FPS, °C or RPM), or `COLLECT FPS`/`N/A` when unavailable.
pub(super) fn format_auto_target(
    target: (AutoMetric, AutoTarget),
    metrics: &MetricsSnapshot,
) -> String {
    match target {
        (AutoMetric::Fps, AutoTarget::Low10Fps) => {
            let fps = metrics.low_10_fps();
            if fps.is_finite() && fps > 0.0 {
                format!("{fps:.0} FPS")
            } else {
                "COLLECT FPS".to_owned()
            }
        }
        (AutoMetric::GpuVrmTempC, AutoTarget::Value(value)) => {
            format!("{value} °C")
        }
        (AutoMetric::FanRpm, AutoTarget::Value(value)) => {
            format!("{value} RPM")
        }
        _ => "N/A".to_owned(),
    }
}

/// Formats the auto-mode F6/F7 hints: VRM/FAN move the target by the hysteresis, FPS by its metric step.
pub(super) fn auto_key_hints(
    metric: Option<AutoMetric>,
    hysteresis: Option<f32>,
) -> (String, String) {
    let Some(metric) = metric else {
        return ("--".into(), "--".into());
    };
    let step = match metric {
        AutoMetric::Fps => Some(metric.hysteresis_step()),
        AutoMetric::GpuVrmTempC | AutoMetric::FanRpm => hysteresis,
    };
    match step.filter(|step| step.is_finite() && *step > 0.0) {
        Some(step) => {
            let (step, unit) = (step.to_string(), metric.hysteresis_unit());
            (format!("-{step} {unit}"), format!("+{step} {unit}"))
        }
        None => ("--".into(), "--".into()),
    }
}
