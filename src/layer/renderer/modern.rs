//! Modern overlay model: a rounded dashboard drawn with the SDF background shader.

use super::background::BackgroundPreparation;
use super::canvas::{draw_rectangles, Canvas, FontFamily, FrameKey, PreparedCanvas};
use super::format::{
    auto_key_hints, format_auto_target, format_fps, format_memory_temperature, format_optional,
    fps_value, frametime_value,
};
use super::pipeline::OverlayPipeline;
use super::{OverlayFeedback, OverlayFrame, OverlayMode, OverlayModel, OverlayRect, OverlayStyle};
use crate::layer::config::{AutoMetric, AutoTarget, Hotkey};
use crate::layer::lib::{MetricsSnapshot, TelemetryStatus};
use ash::vk;

const WIDTH: u32 = 370;
const HEIGHT: u32 = 230;
const HARDWARE_TEXT_SIZE: u32 = 14;
const HARDWARE_LABEL_SIZE: u32 = 12;
const FPS_VALUE_WIDTH: u32 = 94;
const FPS_LABEL_GAP: u32 = 6;
const LABEL: usize = 0;
const VALUE: usize = 1;
const ACCENT: usize = 2;
const FEEDBACK: usize = 3;
const LINE: usize = 4;
const MODE: usize = 5;
const TITLE: usize = 6;
const FRAMETIME: usize = 7;
const GROUPS: usize = 8;

/// Position of the panel in physical swapchain pixels, independent of scale.
const ORIGIN_X: u32 = 2;
const ORIGIN_Y: u32 = 2;

// Palette, as RGB in 0..=1. The FPS value uses `[overlay].fps_color` instead.
/// Captions, hardware labels and key hints.
const LABEL_COLOR: [f32; 3] = [0.55, 0.62, 0.70];
/// Measured values and settings.
const METRIC_COLOR: [f32; 3] = [0.92, 0.94, 0.97];
/// Horizontal section dividers.
const BORDER_COLOR: [f32; 3] = [0.28, 0.33, 0.39];
/// The "FRAMECAP" title.
const TITLE_COLOR: [f32; 3] = [0.55, 0.62, 0.70];
/// Mode badge text and outline.
const HIGHLIGHT_COLOR: [f32; 3] = [1.0, 0.75, 0.0];
/// Dark fill behind the mode badge text.
const INK_COLOR: [f32; 3] = [0.18, 0.13, 0.025];
/// Changed settings and pressed key text and outline.
const FEEDBACK_COLOR: [f32; 3] = [1.0, 0.72, 0.20];
const BACKGROUND_COLOR: [f32; 3] = [0.035, 0.045, 0.06];
const PRESSED_KEY_FILL_COLOR: [f32; 3] = [0.30, 0.20, 0.08];
const PANEL_SHADOW_COLOR: [f32; 3] = [0.0, 0.0, 0.0];
const PANEL_BORDER_COLOR: [f32; 3] = [0.50, 0.57, 0.65];
const KEY_FILL_COLOR: [f32; 3] = [0.15, 0.18, 0.22];
const KEY_BORDER_COLOR: [f32; 3] = [0.38, 0.45, 0.54];

/// Converts an RGB color and alpha to RGBA.
fn rgba([red, green, blue]: [f32; 3], alpha: f32) -> [f32; 4] {
    [red, green, blue, alpha]
}

/// Modern canvas: `GROUPS` color groups with `BoxStyle` boxes.
type Dashboard = Canvas<GROUPS, BoxStyle>;
/// Modern canvas converted for drawing at the swapchain extent.
type PreparedDashboard = PreparedCanvas<GROUPS, BoxStyle>;

/// Rounded boxes drawn with the SDF pipeline: mode badge, idle key, and pressed key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoxStyle {
    Mode,
    Key,
    ActiveKey,
}

impl Dashboard {
    /// Draws bold baseline-aligned sensor text; labels use a smaller size and values are right-aligned.
    fn hardware_text(&mut self, text: &str, x: u32, y: u32, width: u32, group: usize) {
        let size = if group == LABEL {
            HARDWARE_LABEL_SIZE
        } else {
            HARDWARE_TEXT_SIZE
        };
        self.draw_text(
            text,
            x,
            y + HARDWARE_TEXT_SIZE - size,
            size,
            width,
            true,
            group,
            true,
            group == VALUE,
        );
    }
}

impl PreparedDashboard {
    /// Draws the shadowed panel, its border, the rounded boxes, then every text color group.
    pub fn draw(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        opacity: f32,
        fps_color: [u8; 3],
        capture_active: bool,
    ) {
        if extent.width == 0 || extent.height == 0 {
            return;
        }
        let scale = self.scale;
        let panel_width = self.width * scale;
        let panel_height = self.height * scale;
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline)
        };
        let panel = OverlayRect {
            x: ORIGIN_X,
            y: ORIGIN_Y,
            width: panel_width,
            height: panel_height,
        };
        let shadow = OverlayRect {
            x: 2 * scale,
            y: 3 * scale,
            ..panel
        };
        if opacity > 0.0 {
            draw_rectangles(
                device,
                command_buffer,
                pipeline_layout,
                extent,
                &[shadow],
                rgba(PANEL_SHADOW_COLOR, opacity * 0.3),
                10.0 * scale as f32,
                0.0,
            );
            draw_rectangles(
                device,
                command_buffer,
                pipeline_layout,
                extent,
                &[panel],
                rgba(BACKGROUND_COLOR, opacity.clamp(0.0, 1.0)),
                9.0 * scale as f32,
                -1.0,
            );
        }
        draw_rectangles(
            device,
            command_buffer,
            pipeline_layout,
            extent,
            &[panel],
            rgba(PANEL_BORDER_COLOR, 0.9),
            9.0 * scale as f32,
            1.2 * scale as f32,
        );
        for item in &self.boxes {
            let (fill, stroke, radius) = match item.style {
                BoxStyle::Mode => (rgba(INK_COLOR, 0.75), rgba(HIGHLIGHT_COLOR, 1.0), 11.0),
                BoxStyle::Key => (rgba(KEY_FILL_COLOR, 0.9), rgba(KEY_BORDER_COLOR, 1.0), 5.0),
                BoxStyle::ActiveKey => (
                    rgba(PRESSED_KEY_FILL_COLOR, 0.9),
                    rgba(FEEDBACK_COLOR, 1.0),
                    5.0,
                ),
            };
            draw_rectangles(
                device,
                command_buffer,
                pipeline_layout,
                extent,
                &[item.rect],
                fill,
                radius * scale as f32,
                -1.0,
            );
            draw_rectangles(
                device,
                command_buffer,
                pipeline_layout,
                extent,
                &[item.rect],
                stroke,
                radius * scale as f32,
                scale as f32,
            );
        }
        let [red, green, blue] = fps_color.map(|channel| f32::from(channel) / 255.0);
        let colors = [
            rgba(LABEL_COLOR, 1.0),
            rgba(METRIC_COLOR, 1.0),
            [red, green, blue, 1.0],
            rgba(FEEDBACK_COLOR, 1.0),
            rgba(BORDER_COLOR, 1.0),
            rgba(HIGHLIGHT_COLOR, 1.0),
            rgba(TITLE_COLOR, 1.0),
            rgba(
                if capture_active {
                    [1.0, 0.15, 0.15]
                } else {
                    METRIC_COLOR
                },
                1.0,
            ),
        ];
        self.draw_groups(device, command_buffer, extent, pipeline_layout, colors);
    }
}

/// Picks 2× scale when the extent reaches the threshold and fits the doubled panel, otherwise 1×.
fn scale_for(mode: OverlayMode, extent: vk::Extent2D, threshold: u32) -> u32 {
    let (width, height) = if mode == OverlayMode::Full {
        (WIDTH, HEIGHT)
    } else {
        (142, 62)
    };
    if extent.width >= threshold && width * 2 <= extent.width && height * 2 <= extent.height {
        2
    } else {
        1
    }
}

/// Lays out the whole panel (compact or full) for the frame at the given scale.
fn layout(
    metrics: &MetricsSnapshot,
    mode: OverlayMode,
    target_fps: Option<f32>,
    auto_algorithm_index: Option<usize>,
    auto_algorithm_label: Option<&str>,
    hysteresis: Option<f32>,
    auto_target: Option<(AutoMetric, AutoTarget)>,
    hotkeys: [Hotkey; 4],
    manual_step_fps: f32,
    feedback: OverlayFeedback,
    scale: u32,
) -> Dashboard {
    let full = mode == OverlayMode::Full;
    let mut dashboard = Dashboard::new(
        if full { WIDTH } else { 142 },
        if full { HEIGHT } else { 62 },
        scale,
        FontFamily::Sans,
    );
    let fps = fps_value(metrics);
    if !full {
        dashboard.text("FPS", 12, 8, 12, 110, false, LABEL);
        dashboard.text(
            &fps,
            12,
            23,
            if fps == "COLLECT" { 18 } else { 28 },
            120,
            true,
            ACCENT,
        );
        return dashboard;
    }
    dashboard.text("FRAMECAP", 16, 8, 12, 120, true, TITLE);
    let fps_right = dashboard.text(
        &fps,
        16,
        24,
        if fps == "COLLECT" { 20 } else { 36 },
        FPS_VALUE_WIDTH,
        true,
        ACCENT,
    );
    dashboard.text("FPS", fps_right + FPS_LABEL_GAP, 45, 14, 30, true, VALUE);
    let low = metrics.low_10_fps();
    let low = if low.is_finite() && low >= 0.0 {
        format!("{low:.0} FPS")
    } else {
        "COLLECT".to_owned()
    };
    let frametime = frametime_value(metrics);
    for (x, width, label, value, group) in [
        (150, 104, "10% LOW", low, VALUE),
        (258, 96, "FRAMETIME", frametime, FRAMETIME),
    ] {
        dashboard.text(label, x, 26, 12, width, true, LABEL);
        dashboard.text(&value, x, 42, 18, width, true, group);
    }
    dashboard.rect(LINE, 16, 70, WIDTH - 32, 1);
    let mode_label = match (target_fps, auto_algorithm_label) {
        (_, Some(label)) => format!("AUTO · {label}"),
        (Some(fps), None) if fps.is_finite() && fps > 0.0 => "MANUAL".to_owned(),
        _ => "OFF".to_owned(),
    };
    dashboard.styled_box(16, 75, 104, 23, BoxStyle::Mode);
    dashboard.text(&mode_label, 22, 78, 13, 92, true, MODE);
    let cap = target_fps
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(format_fps)
        .unwrap_or_else(|| "OFF".into());
    let hys = hysteresis
        .zip(auto_target.map(|(metric, _)| metric))
        .map(|(value, metric)| format!("{value:.1} {}", metric.hysteresis_unit()))
        .unwrap_or_else(|| "--".into());
    let target = auto_target
        .map(|v| format_auto_target(v, metrics))
        .unwrap_or_else(|| "--".into());
    dashboard.text(
        "CAP",
        128,
        76,
        11,
        45,
        true,
        if feedback.cap_changed {
            FEEDBACK
        } else {
            LABEL
        },
    );
    dashboard.text(
        &format!("{cap} FPS"),
        128,
        88,
        14,
        68,
        true,
        if feedback.cap_changed {
            FEEDBACK
        } else {
            VALUE
        },
    );
    dashboard.text(
        "HYS",
        196,
        76,
        11,
        45,
        true,
        if feedback.hysteresis_changed {
            FEEDBACK
        } else {
            LABEL
        },
    );
    dashboard.text(
        &hys,
        196,
        88,
        14,
        56,
        true,
        if feedback.hysteresis_changed {
            FEEDBACK
        } else {
            VALUE
        },
    );
    dashboard.text(
        "TARGET",
        259,
        76,
        11,
        70,
        true,
        if feedback.target_changed {
            FEEDBACK
        } else {
            LABEL
        },
    );
    dashboard.text(
        &target,
        259,
        88,
        14,
        95,
        true,
        if feedback.target_changed {
            FEEDBACK
        } else {
            VALUE
        },
    );
    dashboard.rect(LINE, 16, 105, WIDTH - 32, 1);
    let hardware = metrics.hardware_metrics();
    let display = |value: Option<f32>, precision: usize, unit: &str| {
        value
            .filter(|v| v.is_finite())
            .map(|v| format!("{v:.precision$} {unit}"))
            .unwrap_or_else(|| "--".into())
    };
    let gpu_load = match metrics.gpu_measurement().status {
        TelemetryStatus::Available => display(Some(metrics.gpu_measurement().value), 0, "%"),
        TelemetryStatus::Expired => "STALE".into(),
        TelemetryStatus::Error => "ERR".into(),
        TelemetryStatus::Unsupported => "N/A".into(),
    };
    for (y, name, temp, vrm, load) in [
        (
            111,
            "GPU",
            hardware.gpu_edge_temp_c,
            hardware.gpu_vrm_temp_c,
            gpu_load,
        ),
        (
            132,
            "CPU",
            hardware.cpu_temp_c,
            hardware.cpu_vrm_temp_c,
            display(hardware.cpu_load_percent, 0, "%"),
        ),
    ] {
        for (x, value_x, value_width, label, value) in [
            (16, 66, 46, name, display(temp, 0, "°C")),
            (130, 166, 52, "VRM", display(vrm, 0, "°C")),
            (238, 275, 53, "LOAD", load),
        ] {
            dashboard.hardware_text(label, x, y, value_x - x, LABEL);
            dashboard.hardware_text(&value, value_x, y, value_width, VALUE);
        }
    }
    for (x, value_x, width, label, value) in [
        (
            16,
            66,
            46,
            "GDDR6",
            format!("{} °C", format_memory_temperature(hardware.memory_temp)),
        ),
        (130, 166, 52, "PPT", display(hardware.ppt_w, 1, "W")),
        (238, 275, 53, "FAN", display(hardware.fan_rpm, 0, "RPM")),
    ] {
        dashboard.hardware_text(label, x, 153, value_x - x, LABEL);
        dashboard.hardware_text(&value, value_x, 153, width, VALUE);
    }
    dashboard.hardware_text("GTT", 16, 174, 50, LABEL);
    dashboard.hardware_text(
        &format!(
            "{} / {} MB",
            format_optional(hardware.gtt_used_mb, 0),
            format_optional(hardware.gtt_total_mb, 0)
        ),
        66,
        174,
        152,
        VALUE,
    );
    dashboard.hardware_text("RAM", 238, 174, 37, LABEL);
    dashboard.hardware_text(
        &format!("{} MB", format_optional(hardware.ram_used_mb, 0)),
        275,
        174,
        53,
        VALUE,
    );
    dashboard.rect(LINE, 16, 195, WIDTH - 32, 1);
    let (down, up) = if auto_algorithm_index.is_some() {
        auto_target.map_or_else(
            || ("--".into(), "--".into()),
            |(metric, _)| auto_key_hints(Some(metric), hysteresis),
        )
    } else if target_fps.is_some_and(|fps| fps.is_finite() && fps > 0.0) {
        (
            format!("-{} FPS", format_fps(manual_step_fps)),
            format!("+{} FPS", format_fps(manual_step_fps)),
        )
    } else {
        ("--".into(), "--".into())
    };
    for (x, width, hotkey, action) in [
        (16, 50, hotkeys[0], "MODE".to_owned()),
        (104, 50, hotkeys[2], down),
        (192, 50, hotkeys[1], up),
        (280, 38, hotkeys[3], "OSD".to_owned()),
    ] {
        let color = if feedback.hotkey == Some(hotkey) {
            FEEDBACK
        } else {
            LABEL
        };
        dashboard.styled_box(
            x,
            201,
            30,
            20,
            if feedback.hotkey == Some(hotkey) {
                BoxStyle::ActiveKey
            } else {
                BoxStyle::Key
            },
        );
        dashboard.text(hotkey.label(), x + 4, 203, 11, 23, true, color);
        dashboard.text(&action, x + 36, 203, 11, width, true, LABEL);
    }
    dashboard
}

/// Spawns the modern-overlay worker, which lays out each frame for the extent.
fn background_modern(
    extent: vk::Extent2D,
    threshold: u32,
) -> std::io::Result<BackgroundPreparation<PreparedDashboard>> {
    BackgroundPreparation::new("modern", move |frame| {
        layout(
            frame.metrics,
            frame.mode,
            frame.target_fps,
            frame.auto_algorithm_index,
            frame.auto_algorithm_label,
            frame.hysteresis,
            frame.auto_target,
            frame.hotkeys,
            frame.manual_step_fps,
            frame.feedback,
            scale_for(frame.mode, extent, threshold),
        )
        .prepare(extent)
    })
}

/// Modern overlay model: owns the pipeline and the background-prepared dashboard.
pub(super) struct ModernModel {
    extent: vk::Extent2D,
    background_opacity: f32,
    fps_color: [u8; 3],
    scale_2x_min_width_px: u32,
    pipeline: OverlayPipeline,
    background: Option<BackgroundPreparation<PreparedDashboard>>,
    worker_failed: bool,
    dashboard: Option<(FrameKey, OverlayMode, PreparedDashboard)>,
}

impl OverlayModel for ModernModel {
    /// Creates the model and its dashboard-shader pipeline.
    fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        style: OverlayStyle,
    ) -> Result<Self, vk::Result> {
        let pipeline = OverlayPipeline::new(
            device,
            render_pass,
            style.extent,
            include_bytes!("../shaders/dashboard.vert.spv"),
            include_bytes!("../shaders/dashboard.frag.spv"),
            128,
        )?;
        Ok(Self {
            extent: style.extent,
            background_opacity: style.background_opacity,
            fps_color: style.fps_color,
            scale_2x_min_width_px: style.scale_2x_min_width_px,
            pipeline,
            background: None,
            worker_failed: false,
            dashboard: None,
        })
    }

    /// Starts the worker if needed, collects the prepared dashboard, and draws it when the mode matches.
    fn record(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame: &OverlayFrame<'_>,
    ) {
        if self.background.is_none() && !self.worker_failed {
            match background_modern(self.extent, self.scale_2x_min_width_px) {
                Ok(background) => self.background = Some(background),
                Err(error) => {
                    crate::layer::diagnostics::error(format!(
                        "failed to start modern overlay preparation worker: {error}"
                    ));
                    self.worker_failed = true;
                }
            }
        }
        if let Some(background) = self.background.as_mut() {
            background.update(frame, &mut self.dashboard);
        }
        let Some((_, mode, dashboard)) = &self.dashboard else {
            return;
        };
        if *mode != frame.mode {
            return;
        }
        dashboard.draw(
            device,
            command_buffer,
            self.extent,
            self.pipeline.pipeline,
            self.pipeline.layout,
            self.background_opacity,
            self.fps_color,
            crate::layer::capture::is_active(),
        );
    }

    /// Releases the Vulkan pipeline resources.
    fn destroy(&self, device: &ash::Device) {
        self.pipeline.destroy(device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::lib::{GpuMeasurement, HardwareMetrics, MemoryTemperature};
    use crate::layer::renderer::canvas::{clip_rect, FontCache, COVERAGE_LEVELS};
    use std::time::Instant;

    const KEYS: [Hotkey; 4] = [Hotkey::F5, Hotkey::F7, Hotkey::F6, Hotkey::F8];

    fn snapshot(fps: f32, hardware: HardwareMetrics, status: TelemetryStatus) -> MetricsSnapshot {
        let now = Instant::now();
        MetricsSnapshot::new_with_hardware_metrics(
            16.0,
            fps,
            if fps > 0.0 { 0.5 } else { -1.0 },
            if fps > 0.0 { 55.0 } else { -1.0 },
            GpuMeasurement {
                value: 55.0,
                timestamp: now,
                status,
                device_id: None,
            },
            hardware,
            now,
        )
    }

    #[test]
    fn modern_prepares_full_and_compact_layouts_in_the_background() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let metrics = snapshot(60.0, HardwareMetrics::default(), TelemetryStatus::Available);
        let mut frame = OverlayFrame {
            metrics: &metrics,
            mode: OverlayMode::Full,
            target_fps: Some(60.0),
            auto_algorithm_index: None,
            auto_algorithm_label: None,
            hysteresis: None,
            auto_target: None,
            hotkeys: KEYS,
            manual_step_fps: 5.0,
            feedback: OverlayFeedback::default(),
        };
        let mut worker = background_modern(extent, 3840).unwrap();
        let mut prepared = None;
        worker.update(&frame, &mut prepared);
        assert!(prepared.is_none());
        for mode in [OverlayMode::Full, OverlayMode::Fps] {
            frame.mode = mode;
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            while !prepared
                .as_ref()
                .is_some_and(|(_, result_mode, _)| *result_mode == mode)
                && Instant::now() < deadline
            {
                worker.update(&frame, &mut prepared);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let (_, result_mode, dashboard) = prepared.as_ref().unwrap();
            assert_eq!(*result_mode, mode);
            assert!(!dashboard.groups.iter().all(Vec::is_empty));
        }
    }

    fn assert_text(
        dashboard: &Dashboard,
        text: &str,
        x: u32,
        y: u32,
        size: u32,
        width: u32,
        group: usize,
    ) {
        let mut reference = Dashboard::new(WIDTH, HEIGHT, dashboard.scale, FontFamily::Sans);
        if group == VALUE && size == HARDWARE_TEXT_SIZE && (111..=174).contains(&y) {
            reference.hardware_text(text, x, y, width, group);
        } else {
            reference.text(text, x, y, size, width, true, group);
        }
        assert!(!reference.groups[group].is_empty());
        assert!(
            reference.groups[group]
                .iter()
                .all(|rect| dashboard.groups[group].contains(rect)),
            "missing {text:?}"
        );
        for (expected, actual) in reference.edges[group].iter().zip(&dashboard.edges[group]) {
            assert!(
                expected.iter().all(|rect| actual.contains(rect)),
                "missing antialiasing for {text:?}"
            );
        }
    }

    #[test]
    fn dashboard_displays_metrics_targets_and_feedback() {
        let metrics = snapshot(
            60.0,
            HardwareMetrics {
                gpu_edge_temp_c: Some(43.0),
                gpu_vrm_temp_c: Some(57.0),
                cpu_temp_c: Some(70.0),
                cpu_vrm_temp_c: Some(61.0),
                cpu_load_percent: Some(35.0),
                ppt_w: Some(32.6),
                fan_rpm: Some(1240.0),
                memory_temp: Some(MemoryTemperature {
                    celsius: 56.0,
                    lower_bound: false,
                }),
                gtt_used_mb: Some(512.0),
                gtt_total_mb: Some(12000.0),
                ram_used_mb: Some(4000.0),
                ..HardwareMetrics::default()
            },
            TelemetryStatus::Available,
        );
        let dashboard = layout(
            &metrics,
            OverlayMode::Full,
            Some(60.0),
            Some(0),
            Some("JIT"),
            Some(5.0),
            Some((AutoMetric::Fps, AutoTarget::Low10Fps)),
            KEYS,
            5.0,
            OverlayFeedback {
                hotkey: Some(Hotkey::F6),
                cap_changed: true,
                hysteresis_changed: true,
                target_changed: false,
            },
            1,
        );
        assert_eq!((dashboard.width, dashboard.height), (370, 230));
        for (text, x, y, size, width, group) in [
            ("FRAMECAP", 16, 8, 12, 120, TITLE),
            ("60", 16, 24, 36, FPS_VALUE_WIDTH, ACCENT),
            ("55 FPS", 150, 42, 18, 104, VALUE),
            ("FRAMETIME", 258, 26, 12, 96, LABEL),
            ("16.00 ms", 258, 42, 18, 96, FRAMETIME),
            ("AUTO · JIT", 22, 78, 13, 92, MODE),
            ("60 FPS", 128, 88, 14, 68, FEEDBACK),
            ("5.0 FPS", 196, 88, 14, 56, FEEDBACK),
            ("55 FPS", 259, 88, 14, 95, VALUE),
            ("GPU", 16, 113, 12, 50, LABEL),
            ("43 °C", 66, 111, 14, 46, VALUE),
            ("VRM", 130, 113, 12, 36, LABEL),
            ("LOAD", 238, 113, 12, 37, LABEL),
            ("CPU", 16, 134, 12, 50, LABEL),
            ("VRM", 130, 134, 12, 36, LABEL),
            ("LOAD", 238, 134, 12, 37, LABEL),
            ("35 %", 275, 132, 14, 53, VALUE),
            ("GDDR6", 16, 155, 12, 50, LABEL),
            ("56 °C", 66, 153, 14, 46, VALUE),
            ("PPT", 130, 155, 12, 36, LABEL),
            ("32.6 W", 166, 153, 14, 52, VALUE),
            ("FAN", 238, 155, 12, 37, LABEL),
            ("1240 RPM", 275, 153, 14, 53, VALUE),
            ("GTT", 16, 176, 12, 50, LABEL),
            ("512 / 12000 MB", 66, 174, 14, 152, VALUE),
            ("RAM", 238, 176, 12, 37, LABEL),
            ("4000 MB", 275, 174, 14, 53, VALUE),
            ("F6", 108, 203, 11, 23, FEEDBACK),
            ("-1 FPS", 140, 203, 11, 50, LABEL),
        ] {
            assert_text(&dashboard, text, x, y, size, width, group);
        }

        assert!(dashboard
            .groups
            .iter()
            .flatten()
            .chain(dashboard.edges.iter().flatten().flatten())
            .all(|rect| rect.x + rect.width <= WIDTH && rect.y + rect.height <= HEIGHT));
        assert!(dashboard.boxes.iter().all(|item| {
            item.rect.x + item.rect.width <= WIDTH && item.rect.y + item.rect.height <= HEIGHT
        }));
        assert_eq!(
            dashboard
                .boxes
                .iter()
                .map(|item| item.style)
                .collect::<Vec<_>>(),
            [
                BoxStyle::Mode,
                BoxStyle::Key,
                BoxStyle::ActiveKey,
                BoxStyle::Key,
                BoxStyle::Key,
            ]
        );
    }

    #[test]
    fn dashboard_hysteresis_and_hotkeys_use_metric_units() {
        let metrics = snapshot(60.0, HardwareMetrics::default(), TelemetryStatus::Available);
        for (metric, target, value, hys, down, up) in [
            (
                AutoMetric::Fps,
                AutoTarget::Low10Fps,
                5.0,
                "5.0 FPS",
                "-1 FPS",
                "+1 FPS",
            ),
            (
                AutoMetric::GpuVrmTempC,
                AutoTarget::Value(100.0),
                5.0,
                "5.0 °C",
                "-5 °C",
                "+5 °C",
            ),
            (
                AutoMetric::FanRpm,
                AutoTarget::Value(2000.0),
                50.0,
                "50.0 RPM",
                "-50 RPM",
                "+50 RPM",
            ),
        ] {
            let dashboard = layout(
                &metrics,
                OverlayMode::Full,
                Some(60.0),
                Some(0),
                Some("AUTO"),
                Some(value),
                Some((metric, target)),
                KEYS,
                5.0,
                OverlayFeedback::default(),
                1,
            );
            assert_text(&dashboard, hys, 196, 88, 14, 56, VALUE);
            assert_text(&dashboard, down, 140, 203, 11, 50, LABEL);
            assert_text(&dashboard, up, 228, 203, 11, 50, LABEL);
        }
    }

    #[test]
    fn dashboard_highlights_adjusted_sensor_target() {
        let metrics = snapshot(60.0, HardwareMetrics::default(), TelemetryStatus::Available);
        let dashboard = layout(
            &metrics,
            OverlayMode::Full,
            Some(60.0),
            Some(1),
            Some("VRM"),
            Some(5.0),
            Some((AutoMetric::GpuVrmTempC, AutoTarget::Value(105.0))),
            KEYS,
            5.0,
            OverlayFeedback {
                target_changed: true,
                ..OverlayFeedback::default()
            },
            1,
        );
        assert_text(&dashboard, "TARGET", 259, 76, 11, 70, FEEDBACK);
        assert_text(&dashboard, "105 °C", 259, 88, 14, 95, FEEDBACK);
    }

    #[test]
    fn hysteresis_fields_shift_left_at_both_scales() {
        let metrics = snapshot(60.0, HardwareMetrics::default(), TelemetryStatus::Available);
        for scale in [1, 2] {
            let dashboard = layout(
                &metrics,
                OverlayMode::Full,
                Some(239.9),
                Some(2),
                Some("FAN"),
                Some(50.0),
                Some((AutoMetric::FanRpm, AutoTarget::Value(2000.0))),
                KEYS,
                5.0,
                OverlayFeedback::default(),
                scale,
            );
            assert_text(&dashboard, "HYS", 196, 76, 11, 45, LABEL);
            assert_text(&dashboard, "50.0 RPM", 196, 88, 14, 56, VALUE);
            assert_text(&dashboard, "239.9 FPS", 128, 88, 14, 68, VALUE);
            assert_text(&dashboard, "TARGET", 259, 76, 11, 70, LABEL);
            assert_text(&dashboard, "2000 RPM", 259, 88, 14, 95, VALUE);
        }
    }

    #[test]
    fn dashboard_shows_collect_and_unavailable_states_and_fps_only() {
        let metrics = snapshot(-1.0, HardwareMetrics::default(), TelemetryStatus::Expired);
        let dashboard = layout(
            &metrics,
            OverlayMode::Full,
            None,
            None,
            None,
            None,
            None,
            KEYS,
            5.0,
            OverlayFeedback::default(),
            1,
        );
        assert_text(&dashboard, "COLLECT", 16, 24, 20, FPS_VALUE_WIDTH, ACCENT);
        assert_text(&dashboard, "COLLECT", 150, 42, 18, 104, VALUE);
        assert_text(&dashboard, "OFF", 22, 78, 13, 92, MODE);
        assert_text(&dashboard, "STALE", 275, 111, 14, 53, VALUE);
        assert_text(&dashboard, "--", 275, 153, 14, 53, VALUE);
        assert_text(&dashboard, "-- °C", 66, 153, 14, 46, VALUE);
        assert_text(&dashboard, "-- / -- MB", 66, 174, 14, 152, VALUE);
        assert_text(&dashboard, "-- MB", 275, 174, 14, 53, VALUE);
        let fps_only = layout(
            &metrics,
            OverlayMode::Fps,
            None,
            None,
            None,
            None,
            None,
            KEYS,
            5.0,
            OverlayFeedback::default(),
            1,
        );
        assert_eq!((fps_only.width, fps_only.height), (142, 62));
        assert_text(&fps_only, "COLLECT", 12, 23, 18, 120, ACCENT);
        assert!(
            fps_only.groups[VALUE].is_empty()
                && fps_only.groups[FEEDBACK].is_empty()
                && fps_only.groups[MODE].is_empty()
        );
        assert!(fps_only.boxes.is_empty());
    }

    #[test]
    fn memory_values_show_lower_bounds_and_independent_unavailability() {
        for scale in [1, 2] {
            let metrics = snapshot(
                60.0,
                HardwareMetrics {
                    memory_temp: Some(MemoryTemperature {
                        celsius: 120.0,
                        lower_bound: true,
                    }),
                    gtt_used_mb: Some(f32::NAN),
                    gtt_total_mb: Some(12000.0),
                    ram_used_mb: Some(4000.0),
                    ..HardwareMetrics::default()
                },
                TelemetryStatus::Available,
            );
            let dashboard = layout(
                &metrics,
                OverlayMode::Full,
                None,
                None,
                None,
                None,
                None,
                KEYS,
                5.0,
                OverlayFeedback::default(),
                scale,
            );
            assert_text(&dashboard, ">=120 °C", 66, 153, 14, 46, VALUE);
            assert_text(&dashboard, "-- / 12000 MB", 66, 174, 14, 152, VALUE);
            assert_text(&dashboard, "4000 MB", 275, 174, 14, 53, VALUE);
        }
    }

    #[test]
    fn fps_label_tracks_value_without_colliding_with_next_column() {
        for scale in [1, 2] {
            for fps in [-1.0, 9.0, 60.0, 144.0, 9999.0, 999999.0] {
                let metrics = snapshot(fps, HardwareMetrics::default(), TelemetryStatus::Available);
                let dashboard = layout(
                    &metrics,
                    OverlayMode::Full,
                    None,
                    None,
                    None,
                    None,
                    None,
                    KEYS,
                    5.0,
                    OverlayFeedback::default(),
                    scale,
                );
                let right = dashboard.groups[ACCENT]
                    .iter()
                    .chain(dashboard.edges[ACCENT].iter().flatten())
                    .map(|rect| rect.x + rect.width)
                    .max()
                    .unwrap()
                    .div_ceil(scale);
                assert_text(&dashboard, "FPS", right + FPS_LABEL_GAP, 45, 14, 30, VALUE);
                assert!(
                    right + FPS_LABEL_GAP + 30 <= 150,
                    "fps={fps}, scale={scale}"
                );
            }
        }
    }

    #[test]
    fn common_values_keep_their_original_font_size() {
        for (text, size, width) in [
            ("9999", 36, FPS_VALUE_WIDTH),
            ("COLLECT", 20, FPS_VALUE_WIDTH),
            ("COLLECT", 18, 104),
            ("1000 FPS", 18, 104),
            ("240 FPS", 14, 72),
            ("100.0%", 14, 56),
            ("100 °C", 14, 52),
            ("100 %", 14, 40),
            ("STALE", 14, 40),
            ("COLLECT FPS", 14, 95),
            ("GPU", 12, 50),
            ("CPU", 12, 50),
            ("VRM", 12, 36),
            ("LOAD", 12, 37),
            ("GDDR6", 12, 50),
            ("PPT", 12, 36),
            ("FAN", 12, 37),
            ("GTT", 12, 50),
            ("RAM", 12, 37),
            ("56 °C", 14, 46),
            ("32.6 W", 14, 52),
            ("512 / 12000 MB", 14, 152),
            ("-5 FPS", 11, 50),
            ("+5 FPS", 11, 50),
            ("-25 FPS", 11, 50),
            ("+25 FPS", 11, 50),
            ("-100 FPS", 11, 50),
            ("+100 FPS", 11, 50),
        ] {
            for scale in [1, 2] {
                let make_dashboard = || Dashboard::new(WIDTH, HEIGHT, scale, FontFamily::Sans);
                let mut constrained = make_dashboard();
                let mut unconstrained = make_dashboard();
                constrained.text(text, 16, 27, size, width, true, VALUE);
                unconstrained.text(text, 16, 27, size, 1000, true, VALUE);
                assert_eq!(
                    constrained.groups[VALUE], unconstrained.groups[VALUE],
                    "{text} at {scale}×"
                );
                assert_eq!(
                    constrained.edges[VALUE], unconstrained.edges[VALUE],
                    "{text} at {scale}×"
                );
            }
        }
    }

    #[test]
    fn hardware_labels_match_metric_label_size_and_share_value_baseline() {
        for scale in [1, 2] {
            let metrics = snapshot(60.0, HardwareMetrics::default(), TelemetryStatus::Available);
            let dashboard = layout(
                &metrics,
                OverlayMode::Full,
                None,
                None,
                None,
                None,
                None,
                KEYS,
                5.0,
                OverlayFeedback::default(),
                scale,
            );
            for (label, x, y, width) in [
                ("GPU", 16, 111, 50),
                ("VRM", 130, 111, 36),
                ("LOAD", 238, 111, 37),
                ("CPU", 16, 132, 50),
                ("VRM", 130, 132, 36),
                ("LOAD", 238, 132, 37),
                ("GDDR6", 16, 153, 50),
                ("PPT", 130, 153, 36),
                ("FAN", 238, 153, 37),
                ("GTT", 16, 174, 50),
                ("RAM", 238, 174, 37),
            ] {
                assert_text(
                    &dashboard,
                    label,
                    x,
                    y + HARDWARE_TEXT_SIZE - HARDWARE_LABEL_SIZE,
                    HARDWARE_LABEL_SIZE,
                    width,
                    LABEL,
                );
            }
        }
    }

    #[test]
    fn reduced_hardware_text_keeps_its_baseline() {
        for scale in [1, 2] {
            let make_dashboard = || Dashboard::new(WIDTH, HEIGHT, scale, FontFamily::Sans);
            for group in [LABEL, VALUE] {
                let mut actual = make_dashboard();
                actual.hardware_text("1240 RPM", 16, 27, 45, group);

                let mut matched_size = None;
                let initial_size = if group == LABEL {
                    HARDWARE_LABEL_SIZE
                } else {
                    HARDWARE_TEXT_SIZE
                };
                for size in 8..initial_size {
                    let mut reference = make_dashboard();
                    reference.draw_text(
                        "1240 RPM",
                        16,
                        27 + HARDWARE_TEXT_SIZE - size,
                        size,
                        45,
                        true,
                        group,
                        false,
                        group == VALUE,
                    );
                    if actual.groups[group] == reference.groups[group]
                        && actual.edges[group] == reference.edges[group]
                    {
                        matched_size = Some(size);
                        break;
                    }
                }
                assert!(matched_size.is_some(), "baseline mismatch at {scale}×");
            }
        }
    }

    #[test]
    fn compact_hardware_values_fit_without_ellipsis() {
        let mut cache = FontCache::new(FontFamily::Sans);
        for scale in [1, 2] {
            for (text, width) in [
                (">=120 °C", 46),
                ("100 °C", 52),
                ("100.0 W", 52),
                ("9999 RPM", 53),
                ("4000 MB", 53),
                ("512 / 12000 MB", 152),
            ] {
                let advance: f32 = text
                    .chars()
                    .map(|c| cache.glyph(c, 8 * scale, true).metrics.advance_width)
                    .sum();
                assert!(
                    advance <= (width * scale) as f32,
                    "{text} does not fit at {scale}×"
                );
            }
        }
    }

    #[test]
    fn hardware_value_units_align_to_the_right_at_both_scales() {
        for scale in [1, 2] {
            for (width, short, long) in [
                (46, "9 °C", "100 °C"),
                (52, "9 °C", "100 °C"),
                (53, "5 %", "100 %"),
                (53, "5 MB", "4000 MB"),
                (152, "1 / 9 MB", "512 / 12000 MB"),
            ] {
                let render = |text| {
                    let mut dashboard = Dashboard::new(WIDTH, HEIGHT, scale, FontFamily::Sans);
                    dashboard.hardware_text(text, 70, 111, width, VALUE);
                    dashboard
                };
                let short = render(short);
                let long = render(long);
                let right_edge = |dashboard: &Dashboard| {
                    dashboard.groups[VALUE]
                        .iter()
                        .chain(dashboard.edges[VALUE].iter().flatten())
                        .map(|rect| rect.x + rect.width)
                        .max()
                        .unwrap()
                };
                assert_eq!(
                    right_edge(&short),
                    right_edge(&long),
                    "unit alignment at {scale}×, width {width}"
                );
                for dashboard in [&short, &long] {
                    assert!(dashboard.groups[VALUE]
                        .iter()
                        .chain(dashboard.edges[VALUE].iter().flatten())
                        .all(|rect| rect.x >= 70 * scale
                            && rect.x + rect.width <= (70 + width) * scale));
                }
            }
        }
    }

    #[test]
    fn hardware_values_are_closer_to_their_labels() {
        for scale in [1, 2] {
            for (text, old_x, old_width, new_x, new_width, shift) in [
                ("43 °C", 66, 60, 66, 46, 14),
                ("57 °C", 180, 52, 166, 52, 14),
                ("55 %", 275, 67, 275, 53, 14),
                ("512 / 12000 MB", 66, 166, 66, 152, 14),
                ("4000 MB", 275, 67, 275, 53, 14),
            ] {
                let right_edge = |x, width| {
                    let mut dashboard = Dashboard::new(WIDTH, HEIGHT, scale, FontFamily::Sans);
                    dashboard.hardware_text(text, x, 111, width, VALUE);
                    dashboard.groups[VALUE]
                        .iter()
                        .chain(dashboard.edges[VALUE].iter().flatten())
                        .map(|rect| rect.x + rect.width)
                        .max()
                        .unwrap()
                };
                assert_eq!(
                    right_edge(new_x, new_width) + shift * scale,
                    right_edge(old_x, old_width),
                    "{text} at {scale}×"
                );
            }
        }
    }

    #[test]
    fn compact_rows_and_key_hints_do_not_overlap() {
        let hardware = HardwareMetrics {
            gpu_edge_temp_c: Some(100.0),
            gpu_vrm_temp_c: Some(100.0),
            cpu_temp_c: Some(100.0),
            cpu_vrm_temp_c: Some(100.0),
            cpu_load_percent: Some(100.0),
            ppt_w: Some(100.0),
            fan_rpm: Some(9999.0),
            memory_temp: Some(MemoryTemperature {
                celsius: 120.0,
                lower_bound: true,
            }),
            gtt_used_mb: Some(512.0),
            gtt_total_mb: Some(12000.0),
            ram_used_mb: Some(4000.0),
            ..HardwareMetrics::default()
        };
        let intersects = |a: &OverlayRect, b: &OverlayRect| {
            a.x < b.x + b.width
                && b.x < a.x + a.width
                && a.y < b.y + b.height
                && b.y < a.y + a.height
        };
        for scale in [1, 2] {
            for fps in [-1.0, 60.0, 9999.0] {
                let metrics = snapshot(fps, hardware.clone(), TelemetryStatus::Available);
                let dashboard = layout(
                    &metrics,
                    OverlayMode::Full,
                    Some(240.0),
                    Some(0),
                    Some("JIT"),
                    Some(100.0),
                    Some((AutoMetric::Fps, AutoTarget::Low10Fps)),
                    KEYS,
                    5.0,
                    OverlayFeedback::default(),
                    scale,
                );
                let text = dashboard
                    .groups
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != LINE)
                    .flat_map(|(_, group)| group.iter())
                    .chain(dashboard.edges.iter().flatten().flatten());
                for rect in text {
                    assert!(
                        dashboard.groups[LINE]
                            .iter()
                            .all(|line| !intersects(rect, line)),
                        "text intersects separator: fps={fps}, scale={scale}, {rect:?}"
                    );
                }
                for key in &dashboard.boxes {
                    assert!(dashboard.groups[LINE]
                        .iter()
                        .all(|line| !intersects(&key.rect, line)));
                }
            }
            for (x, action, next_key) in [
                (16, "MODE", 104),
                (104, "-25 FPS", 192),
                (192, "+25 FPS", 280),
            ] {
                let mut reference = Dashboard::new(WIDTH, HEIGHT, scale, FontFamily::Sans);
                reference.text(action, x + 36, 203, 11, 50, true, LABEL);
                assert!(reference.groups[LABEL]
                    .iter()
                    .chain(reference.edges[LABEL].iter().flatten())
                    .all(|rect| rect.x + rect.width <= next_key * scale));
            }
        }
    }

    #[test]
    fn prepared_dashboard_scales_only_when_it_fits_and_clips_to_swapchain() {
        let metrics = snapshot(60.0, HardwareMetrics::default(), TelemetryStatus::Available);
        for (extent, threshold, expected_scale) in [
            (
                vk::Extent2D {
                    width: 200,
                    height: 100,
                },
                100,
                1,
            ),
            (
                vk::Extent2D {
                    width: 1000,
                    height: 400,
                },
                100,
                1,
            ),
            (
                vk::Extent2D {
                    width: 1000,
                    height: 600,
                },
                100,
                2,
            ),
            (
                vk::Extent2D {
                    width: 1000,
                    height: 600,
                },
                2000,
                1,
            ),
            (
                vk::Extent2D {
                    width: WIDTH * 2 - 1,
                    height: HEIGHT * 2,
                },
                100,
                1,
            ),
            (
                vk::Extent2D {
                    width: WIDTH * 2,
                    height: HEIGHT * 2 - 1,
                },
                100,
                1,
            ),
            (
                vk::Extent2D {
                    width: WIDTH * 2,
                    height: HEIGHT * 2,
                },
                100,
                2,
            ),
        ] {
            let prepared = layout(
                &metrics,
                OverlayMode::Full,
                Some(60.0),
                None,
                None,
                None,
                None,
                KEYS,
                5.0,
                OverlayFeedback::default(),
                scale_for(OverlayMode::Full, extent, threshold),
            )
            .prepare(extent);
            assert_eq!(prepared.scale, expected_scale);
            assert!(prepared.groups.iter().flatten().all(|rect| {
                rect.rect.offset.x >= 0
                    && rect.rect.offset.y >= 0
                    && rect.rect.offset.x as u32 + rect.rect.extent.width <= extent.width
                    && rect.rect.offset.y as u32 + rect.rect.extent.height <= extent.height
            }));
            assert!(prepared.edges.iter().flatten().flatten().all(|rect| {
                rect.x + rect.width <= extent.width && rect.y + rect.height <= extent.height
            }));
            if expected_scale == 2 {
                assert!(prepared
                    .edges
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|rect| rect.height == 1));
            }
        }
        assert_eq!(
            clip_rect(
                OverlayRect {
                    x: 8,
                    y: 9,
                    width: 4,
                    height: 3,
                },
                vk::Extent2D {
                    width: 10,
                    height: 10,
                }
            ),
            Some(OverlayRect {
                x: 8,
                y: 9,
                width: 2,
                height: 1,
            })
        );
        assert_eq!(
            clip_rect(
                OverlayRect {
                    x: 10,
                    y: 0,
                    width: 1,
                    height: 1,
                },
                vk::Extent2D {
                    width: 10,
                    height: 10,
                }
            ),
            None
        );
    }

    #[test]
    fn glyph_cache_preserves_quantized_coverage_at_both_scales() {
        let mut cache = FontCache::new(FontFamily::Sans);
        for size in [36, 72] {
            let (_, bitmap) = cache.bold.rasterize('6', size as f32);
            let mut expected = [0usize; COVERAGE_LEVELS + 1];
            for pixel in bitmap {
                let coverage = (usize::from(pixel) * COVERAGE_LEVELS + 127) / 255;
                if coverage > 0 {
                    expected[coverage] += 1;
                }
            }
            let glyph = cache.glyph('6', size, true);
            let mut counts = [0usize; COVERAGE_LEVELS + 1];
            for &(_, _, width, coverage) in &glyph.spans {
                counts[coverage] += width as usize;
            }
            assert_eq!(counts, expected);
            assert!(counts[1..COVERAGE_LEVELS].iter().sum::<usize>() > 30);
            assert!(counts[COVERAGE_LEVELS] > 30);
        }
    }
}
