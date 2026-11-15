//! Helix overlay model: a flat, square-cornered panel in monospace type, framed by teal corner
//! brackets and split into numbered sections (`01 // MODE`, `02 // SENSORS`, `03 // KEY BINDINGS`).
//!
//! It is drawn on a [`Canvas`] with the SDF background shader; every shape is a flat rectangle, so
//! the panel keeps crisp edges at both scales.

use super::background::BackgroundPreparation;
use super::canvas::{draw_rectangles, Canvas, FontFamily, FrameKey, PreparedCanvas};
use super::format::{
    auto_key_hints, format_auto_target, format_fps, format_memory_temperature, format_optional,
    fps_value, frametime_value,
};
use super::pipeline::OverlayPipeline;
use super::{OverlayFrame, OverlayMode, OverlayModel, OverlayRect, OverlayStyle};
use crate::layer::lib::TelemetryStatus;
use ash::vk;

const WIDTH: u32 = 384;
const HEIGHT: u32 = 254;
const COMPACT_WIDTH: u32 = 130;
const COMPACT_HEIGHT: u32 = 52;
const MARGIN: u32 = 10;

/// Position of the panel in physical swapchain pixels, independent of scale.
const ORIGIN_X: u32 = 2;
const ORIGIN_Y: u32 = 2;

/// Length of each corner bracket arm and its thickness, in logical pixels.
const BRACKET_ARM: u32 = 14;
const BRACKET_THICKNESS: u32 = 2;

const TITLE_SIZE: u32 = 14;
const FPS_SIZE: u32 = 29;
const FPS_COLLECT_SIZE: u32 = 21;
const SMALL_LABEL_SIZE: u32 = 11;
const LARGE_VALUE_SIZE: u32 = 17;
const CAP_VALUE_SIZE: u32 = 13;
const HEADING_SIZE: u32 = 10;
const MODE_SIZE: u32 = 13;
const SENSOR_SIZE: u32 = 12;
const KEY_SIZE: u32 = 11;
const COMPACT_LABEL_SIZE: u32 = 11;
const COMPACT_VALUE_SIZE: u32 = 27;
const COMPACT_COLLECT_SIZE: u32 = 19;

const LABEL: usize = 0;
const VALUE: usize = 1;
const ACCENT: usize = 2;
const FEEDBACK: usize = 3;
const LINE: usize = 4;
const TEAL: usize = 5;
// Drawn after TEAL so that text lies over the teal mode box.
const INK: usize = 6;
const CORNER: usize = 7;
const TITLE: usize = 8;
const FRAMETIME: usize = 9;
const GROUPS: usize = 10;

// Palette, as RGB in 0..=1. The FPS value uses `[overlay].fps_color` instead.
/// Corner brackets framing the panel.
const CORNER_COLOR: [f32; 3] = [0.10, 0.80, 0.85];
/// Captions: "10% LOW", "FRAMETIME", "CAP", sensor names and key actions.
const LABEL_COLOR: [f32; 3] = [0.55, 0.55, 0.55];
/// Measured values and settings (55 FPS, 43 °C, ...).
const METRIC_COLOR: [f32; 3] = [0.90, 0.92, 0.95];
/// Section headings and rules, mode badge frame, vertical dividers and empty GTT segments.
const BORDER_COLOR: [f32; 3] = [0.09, 0.25, 0.26];
/// The "FRAMECAP" title.
const TITLE_COLOR: [f32; 3] = [0.90, 0.92, 0.95];
/// Title bar, mode badge fill, key boxes and filled GTT segments.
const HIGHLIGHT_COLOR: [f32; 3] = [0.10, 0.70, 0.70];
/// Text on the highlighted mode badge.
const INK_COLOR: [f32; 3] = [0.02, 0.05, 0.07];
/// Changed settings and pressed key.
const FEEDBACK_COLOR: [f32; 3] = [1.0, 0.72, 0.20];
const BACKGROUND_COLOR: [f32; 3] = [0.05, 0.05, 0.05];
const PRESSED_KEY_FILL_COLOR: [f32; 3] = [0.30, 0.20, 0.08];
/// Opacity of the mode badge in MANUAL mode (AUTO is opaque).
const MODE_MANUAL_ALPHA: f32 = 0.6;

/// Converts an RGB color to RGBA with full opacity.
fn opaque([red, green, blue]: [f32; 3]) -> [f32; 4] {
    [red, green, blue, 1.0]
}

const GTT_SEGMENTS: u32 = 10;
const SENSOR_BASELINES: [u32; 4] = [139, 158, 177, 196];
/// Label x, label width, value x and value width of the three sensor columns.
const SENSOR_COLUMNS: [(u32, u32, u32, u32); 3] =
    [(10, 55, 67, 78), (151, 45, 198, 66), (270, 44, 316, 60)];

/// Boxes are flat fills drawn under the color groups; only a pressed key has one.
/// Boxes are flat fills drawn under the color groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoxStyle {
    PressedKey,
    ModeAuto,
    ModeManual,
}

/// Helix canvas: `GROUPS` color groups with `BoxStyle` boxes.
type Helix = Canvas<GROUPS, BoxStyle>;
/// Helix canvas converted for drawing at the swapchain extent.
type PreparedHelix = PreparedCanvas<GROUPS, BoxStyle>;

impl Helix {
    /// Draws text whose baseline is `baseline`, even when it has to shrink; returns its right edge.
    fn put(
        &mut self,
        text: &str,
        x: u32,
        baseline: u32,
        size: u32,
        width: u32,
        bold: bool,
        group: usize,
    ) -> u32 {
        self.draw_text(
            text,
            x,
            baseline - size,
            size,
            width,
            bold,
            group,
            true,
            false,
        )
    }

    /// Draws a one pixel outline made of four flat rectangles.
    fn outline(&mut self, group: usize, x: u32, y: u32, width: u32, height: u32) {
        self.rect(group, x, y, width, 1);
        self.rect(group, x, y + height - 1, width, 1);
        self.rect(group, x, y, 1, height);
        self.rect(group, x + width - 1, y, 1, height);
    }

    /// Draws an L-shaped bracket in each corner of the panel.
    fn corner_brackets(&mut self) {
        let (width, height) = (self.width, self.height);
        for (right, bottom) in [(false, false), (true, false), (false, true), (true, true)] {
            let arm_x = if right { width - BRACKET_ARM } else { 0 };
            let arm_y = if bottom {
                height - BRACKET_THICKNESS
            } else {
                0
            };
            self.rect(CORNER, arm_x, arm_y, BRACKET_ARM, BRACKET_THICKNESS);
            let bar_x = if right { width - BRACKET_THICKNESS } else { 0 };
            let bar_y = if bottom { height - BRACKET_ARM } else { 0 };
            self.rect(CORNER, bar_x, bar_y, BRACKET_THICKNESS, BRACKET_ARM);
        }
    }

    /// Draws a section heading followed by a rule of the same border color up to the right margin.
    fn section(&mut self, title: &str, baseline: u32, rule_y: u32) {
        let right = self.put(title, MARGIN, baseline, HEADING_SIZE, 100, true, LINE) + 5;
        self.rect(LINE, right, rule_y, WIDTH - MARGIN - right, 1);
    }
}

impl PreparedHelix {
    /// Returns the panel's pixel rectangle at its on-screen origin.
    fn panel_rect(&self) -> OverlayRect {
        OverlayRect {
            x: ORIGIN_X,
            y: ORIGIN_Y,
            width: self.width * self.scale,
            height: self.height * self.scale,
        }
    }

    /// Draws the panel background, styled boxes, then every text color group.
    fn draw(
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
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline)
        };
        let panel = self.panel_rect();
        if opacity > 0.0 {
            draw_rectangles(
                device,
                command_buffer,
                pipeline_layout,
                extent,
                &[panel],
                [
                    BACKGROUND_COLOR[0],
                    BACKGROUND_COLOR[1],
                    BACKGROUND_COLOR[2],
                    opacity.clamp(0.0, 1.0),
                ],
                0.0,
                0.0,
            );
        }
        for item in &self.boxes {
            let [red, green, blue] = match item.style {
                BoxStyle::PressedKey => PRESSED_KEY_FILL_COLOR,
                BoxStyle::ModeAuto | BoxStyle::ModeManual => HIGHLIGHT_COLOR,
            };
            let alpha = match item.style {
                BoxStyle::PressedKey => 0.9,
                BoxStyle::ModeAuto => 1.0,
                BoxStyle::ModeManual => MODE_MANUAL_ALPHA,
            };
            draw_rectangles(
                device,
                command_buffer,
                pipeline_layout,
                extent,
                &[item.rect],
                [red, green, blue, alpha],
                0.0,
                0.0,
            );
        }
        let [red, green, blue] = fps_color.map(|channel| f32::from(channel) / 255.0);
        let colors = [
            opaque(LABEL_COLOR),
            opaque(METRIC_COLOR),
            [red, green, blue, 1.0],
            opaque(FEEDBACK_COLOR),
            opaque(BORDER_COLOR),
            opaque(HIGHLIGHT_COLOR),
            opaque(INK_COLOR),
            opaque(CORNER_COLOR),
            opaque(TITLE_COLOR),
            opaque(if capture_active {
                [1.0, 0.15, 0.15]
            } else {
                METRIC_COLOR
            }),
        ];
        self.draw_groups(device, command_buffer, extent, pipeline_layout, colors);
    }
}

/// Panel size in logical units: full or compact depending on the mode.
fn dimensions(mode: OverlayMode) -> (u32, u32) {
    if mode == OverlayMode::Full {
        (WIDTH, HEIGHT)
    } else {
        (COMPACT_WIDTH, COMPACT_HEIGHT)
    }
}

/// Picks 2× scale when the extent reaches the threshold and fits the doubled panel, otherwise 1×.
fn scale_for(mode: OverlayMode, extent: vk::Extent2D, threshold: u32) -> u32 {
    let (width, height) = dimensions(mode);
    if extent.width >= threshold
        && width * 2 <= extent.width.saturating_sub(ORIGIN_X)
        && height * 2 <= extent.height.saturating_sub(ORIGIN_Y)
    {
        2
    } else {
        1
    }
}

/// Lays out the whole panel (compact or full) for the frame at the given scale.
fn layout(frame: &OverlayFrame<'_>, scale: u32) -> Helix {
    let OverlayFrame {
        metrics,
        mode,
        target_fps,
        auto_algorithm_index,
        auto_algorithm_label,
        hysteresis,
        auto_target,
        hotkeys,
        manual_step_fps,
        feedback,
    } = *frame;
    let (width, height) = dimensions(mode);
    let mut helix = Helix::new(width, height, scale, FontFamily::Mono);
    helix.corner_brackets();
    let fps = fps_value(metrics);
    let collecting = fps == "COLLECT";
    if mode != OverlayMode::Full {
        helix.put("FPS", 14, 19, COMPACT_LABEL_SIZE, 60, false, LABEL);
        helix.put(
            &fps,
            14,
            46,
            if collecting {
                COMPACT_COLLECT_SIZE
            } else {
                COMPACT_VALUE_SIZE
            },
            108,
            true,
            ACCENT,
        );
        return helix;
    }

    helix.put("FRAMECAP", 10, 22, TITLE_SIZE, 120, false, TITLE);
    let fps_text = if collecting {
        fps
    } else {
        format!("{fps} FPS")
    };
    helix.put(
        &fps_text,
        MARGIN,
        58,
        if collecting {
            FPS_COLLECT_SIZE
        } else {
            FPS_SIZE
        },
        118,
        true,
        ACCENT,
    );
    let low = metrics.low_10_fps();
    let low = if low.is_finite() && low >= 0.0 {
        format!("{low:.0} FPS")
    } else {
        "COLLECT".to_owned()
    };
    let frametime = frametime_value(metrics);
    for (divider, x, width, label, value, group) in [
        (130, 145, 110, "10% LOW", low, VALUE),
        (260, 275, 99, "FRAMETIME", frametime, FRAMETIME),
    ] {
        helix.rect(LINE, divider, 35, 1, 24);
        helix.put(label, x, 42, SMALL_LABEL_SIZE, width, false, LABEL);
        helix.put(&value, x, 60, LARGE_VALUE_SIZE, width, true, group);
    }

    helix.section("01 // MODE", 77, 73);
    let (mode_label, badge, mode_text) = match (target_fps, auto_algorithm_label) {
        (_, Some(label)) => (format!("AUTO • {label}"), Some(BoxStyle::ModeAuto), INK),
        (Some(fps), None) if fps.is_finite() && fps > 0.0 => {
            ("MANUAL".to_owned(), Some(BoxStyle::ModeManual), VALUE)
        }
        _ => ("OFF".to_owned(), None, LABEL),
    };
    // The frame is always drawn; the fill is full in AUTO, half in MANUAL and absent when OFF.
    helix.outline(LINE, MARGIN, 83, 92, 25);
    if let Some(style) = badge {
        helix.styled_box(MARGIN, 83, 92, 25, style);
    }
    helix.put(&mode_label, 16, 100, MODE_SIZE, 80, true, mode_text);
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
    for (divider, x, width, label, value, changed) in [
        (
            116,
            129,
            68,
            "CAP",
            format!("{cap} FPS"),
            feedback.cap_changed,
        ),
        (201, 214, 70, "HYS", hys, feedback.hysteresis_changed),
        (291, 303, 71, "TARGET", target, feedback.target_changed),
    ] {
        helix.rect(LINE, divider, 83, 1, 25);
        let (label_group, value_group) = if changed {
            (FEEDBACK, FEEDBACK)
        } else {
            (LABEL, VALUE)
        };
        helix.put(label, x, 92, SMALL_LABEL_SIZE, width, false, label_group);
        helix.put(&value, x, 105, CAP_VALUE_SIZE, width, true, value_group);
    }

    helix.section("02 // SENSORS", 124, 120);
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
    let gtt = format!(
        "{} / {} MB",
        format_optional(hardware.gtt_used_mb, 0),
        format_optional(hardware.gtt_total_mb, 0)
    );
    let rows: [[(&str, String); 3]; 4] = [
        [
            ("GPU", display(hardware.gpu_edge_temp_c, 0, "°C")),
            ("VRM", display(hardware.gpu_vrm_temp_c, 0, "°C")),
            ("LOAD", gpu_load),
        ],
        [
            ("CPU", display(hardware.cpu_temp_c, 0, "°C")),
            ("VRM", display(hardware.cpu_vrm_temp_c, 0, "°C")),
            ("LOAD", display(hardware.cpu_load_percent, 0, "%")),
        ],
        [
            (
                "GDDR6",
                format!("{} °C", format_memory_temperature(hardware.memory_temp)),
            ),
            ("PPT", display(hardware.ppt_w, 1, "W")),
            ("FAN", display(hardware.fan_rpm, 0, "RPM")),
        ],
        [
            ("GTT", gtt),
            ("", String::new()),
            (
                "RAM",
                format!("{} MB", format_optional(hardware.ram_used_mb, 0)),
            ),
        ],
    ];
    for (baseline, row) in SENSOR_BASELINES.into_iter().zip(rows) {
        for ((label_x, label_width, value_x, value_width), (label, value)) in
            SENSOR_COLUMNS.into_iter().zip(row)
        {
            if label.is_empty() {
                continue;
            }
            helix.put(
                label,
                label_x,
                baseline,
                SENSOR_SIZE,
                label_width,
                false,
                LABEL,
            );
            // The GTT value may spread over the empty middle column.
            let value_width = if label == "GTT" { 180 } else { value_width };
            helix.put(
                &value,
                value_x,
                baseline,
                SENSOR_SIZE,
                value_width,
                false,
                VALUE,
            );
        }
    }
    if let (Some(used), Some(total)) = (hardware.gtt_used_mb, hardware.gtt_total_mb) {
        if used.is_finite() && total.is_finite() && total > 0.0 {
            let filled = ((used / total) * GTT_SEGMENTS as f32)
                .round()
                .clamp(0.0, GTT_SEGMENTS as f32) as u32;
            for index in 0..GTT_SEGMENTS {
                let x = 67 + index * 9;
                if index < filled {
                    helix.rect(TEAL, x, 200, 6, 6);
                } else {
                    helix.outline(LINE, x, 200, 6, 6);
                }
            }
        }
    }

    helix.section("03 // KEY BINDINGS", 225, 221);
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
    for (index, (x, width, hotkey, action)) in [
        (10, 45, hotkeys[0], "MODE".to_owned()),
        (101, 56, hotkeys[2], down),
        (203, 55, hotkeys[1], up),
        (304, 38, hotkeys[3], "OSD".to_owned()),
    ]
    .into_iter()
    .enumerate()
    {
        if index > 0 {
            helix.rect(LINE, x - 10, 230, 1, 16);
        }
        let group = if feedback.hotkey == Some(hotkey) {
            helix.styled_box(x, 230, 26, 16, BoxStyle::PressedKey);
            FEEDBACK
        } else {
            TEAL
        };
        helix.outline(group, x, 230, 26, 16);
        helix.put(hotkey.label(), x + 4, 242, KEY_SIZE, 20, true, group);
        helix.put(&action, x + 32, 242, KEY_SIZE, width, false, LABEL);
    }
    helix
}

/// Background worker that lays out Helix frames.
type BackgroundHelix = BackgroundPreparation<PreparedHelix>;

/// Spawns the Helix worker, which lays out each frame and positions it at the panel origin.
fn background_helix(
    extent: vk::Extent2D,
    scale_2x_min_width_px: u32,
) -> std::io::Result<BackgroundHelix> {
    BackgroundHelix::new("helix", move |frame| {
        let scale = scale_for(frame.mode, extent, scale_2x_min_width_px);
        layout(frame, scale).prepare_at(extent, ORIGIN_X, ORIGIN_Y)
    })
}

/// Helix overlay model: owns the pipeline and the background-prepared panel layout.
pub(super) struct HelixModel {
    extent: vk::Extent2D,
    background_opacity: f32,
    fps_color: [u8; 3],
    scale_2x_min_width_px: u32,
    pipeline: OverlayPipeline,
    background: Option<BackgroundHelix>,
    worker_failed: bool,
    prepared: Option<(FrameKey, OverlayMode, PreparedHelix)>,
}

impl OverlayModel for HelixModel {
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
            prepared: None,
        })
    }

    /// Starts the worker if needed, collects the prepared panel, and draws it when the mode matches.
    fn record(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame: &OverlayFrame<'_>,
    ) {
        if self.background.is_none() && !self.worker_failed {
            match background_helix(self.extent, self.scale_2x_min_width_px) {
                Ok(background) => self.background = Some(background),
                Err(error) => {
                    crate::layer::diagnostics::error(format!(
                        "failed to start Helix overlay preparation worker: {error}"
                    ));
                    self.worker_failed = true;
                }
            }
        }
        if let Some(background) = self.background.as_mut() {
            background.update(frame, &mut self.prepared);
        }
        let Some((_, mode, prepared)) = &self.prepared else {
            return;
        };
        if *mode != frame.mode {
            return;
        }
        prepared.draw(
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
    use crate::layer::config::{AutoMetric, AutoTarget, Hotkey};
    use crate::layer::lib::{GpuMeasurement, HardwareMetrics, MemoryTemperature, MetricsSnapshot};
    use crate::layer::renderer::canvas::FontCache;
    use crate::layer::renderer::OverlayFeedback;
    use std::sync::mpsc;
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

    fn frame(
        metrics: &MetricsSnapshot,
        mode: OverlayMode,
        feedback: OverlayFeedback,
    ) -> OverlayFrame<'_> {
        OverlayFrame {
            metrics,
            mode,
            target_fps: Some(60.0),
            auto_algorithm_index: Some(0),
            auto_algorithm_label: Some("JIT"),
            hysteresis: Some(5.0),
            auto_target: Some((AutoMetric::Fps, AutoTarget::Low10Fps)),
            hotkeys: KEYS,
            manual_step_fps: 5.0,
            feedback,
        }
    }

    fn full_hardware() -> HardwareMetrics {
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
            gtt_used_mb: Some(6000.0),
            gtt_total_mb: Some(12000.0),
            ram_used_mb: Some(4000.0),
            ..HardwareMetrics::default()
        }
    }

    #[test]
    fn background_helix_prepares_without_blocking_the_presentation() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let mut worker = background_helix(extent, 3840).unwrap();
        let metrics = snapshot(60.0, full_hardware(), TelemetryStatus::Available);
        let current = frame(&metrics, OverlayMode::Full, OverlayFeedback::default());
        let key = FrameKey::new(&current);
        let mut prepared = None;
        worker.update(&current, &mut prepared);
        assert!(prepared.is_none());

        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while prepared.is_none() && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
            worker.update(&current, &mut prepared);
        }
        assert!(prepared.as_ref().is_some_and(|(result_key, mode, _)| {
            *result_key == key && *mode == OverlayMode::Full
        }));
    }

    #[test]
    fn background_helix_accepts_older_metrics_but_rejects_obsolete_controls() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let (requests, receiver) = mpsc::sync_channel(1);
        let (sender, completed) = mpsc::channel();
        let mut worker = BackgroundHelix {
            requests: Some(requests),
            completed,
            thread: None,
            requested: None,
            failed: false,
            name: "helix",
        };
        let old_metrics = snapshot(60.0, full_hardware(), TelemetryStatus::Available);
        let old = frame(&old_metrics, OverlayMode::Full, OverlayFeedback::default());
        let old_key = FrameKey::new(&old);
        let old_layout = layout(&old, 1).prepare_at(extent, ORIGIN_X, ORIGIN_Y);
        let current_metrics = snapshot(61.0, full_hardware(), TelemetryStatus::Available);
        let current = frame(
            &current_metrics,
            OverlayMode::Full,
            OverlayFeedback::default(),
        );
        let current_key = FrameKey::new(&current);
        assert!(old_key != current_key);
        assert!(sender
            .send((old_key, OverlayMode::Full, old_layout))
            .is_ok());
        let mut prepared = None;
        worker.update(&current, &mut prepared);
        assert!(prepared.is_some());
        assert!(receiver.try_recv().is_ok());

        let old_layout = layout(&current, 1).prepare_at(extent, ORIGIN_X, ORIGIN_Y);
        assert!(sender
            .send((current_key, OverlayMode::Full, old_layout))
            .is_ok());
        let compact = frame(
            &current_metrics,
            OverlayMode::Fps,
            OverlayFeedback::default(),
        );
        worker.update(&compact, &mut prepared);
        assert!(prepared
            .as_ref()
            .is_some_and(|(_, mode, _)| *mode == OverlayMode::Full));
        let request = receiver.try_recv().unwrap();
        assert!(request.key == FrameKey::new(&compact));
    }

    #[test]
    fn background_helix_retries_the_latest_request_when_the_queue_is_full() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let (requests, receiver) = mpsc::sync_channel(1);
        let (sender, completed) = mpsc::channel();
        let mut worker = BackgroundHelix {
            requests: Some(requests),
            completed,
            thread: None,
            requested: None,
            failed: false,
            name: "helix",
        };
        let metrics = snapshot(60.0, full_hardware(), TelemetryStatus::Available);
        let full = frame(&metrics, OverlayMode::Full, OverlayFeedback::default());
        let compact = frame(&metrics, OverlayMode::Fps, OverlayFeedback::default());
        let mut prepared = None;
        worker.update(&full, &mut prepared);
        worker.update(&compact, &mut prepared);
        assert!(worker.requested.as_ref() == Some(&FrameKey::new(&full)));

        assert!(sender
            .send((
                FrameKey::new(&full),
                OverlayMode::Full,
                layout(&full, 1).prepare_at(extent, ORIGIN_X, ORIGIN_Y),
            ))
            .is_ok());
        worker.update(&compact, &mut prepared);
        assert!(prepared.is_none());
        assert!(receiver.try_recv().is_ok());
        worker.update(&compact, &mut prepared);
        assert!(receiver.try_recv().unwrap().key == FrameKey::new(&compact));
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_text(
        helix: &Helix,
        text: &str,
        x: u32,
        baseline: u32,
        size: u32,
        width: u32,
        bold: bool,
        group: usize,
    ) {
        let mut reference = Helix::new(helix.width, helix.height, helix.scale, FontFamily::Mono);
        reference.put(text, x, baseline, size, width, bold, group);
        let drawn = reference.groups[group].len()
            + reference.edges[group].iter().map(Vec::len).sum::<usize>();
        assert!(drawn > 0, "no pixels for {text:?}");
        assert!(
            reference.groups[group]
                .iter()
                .all(|rect| helix.groups[group].contains(rect)),
            "missing {text:?}"
        );
        for (expected, actual) in reference.edges[group].iter().zip(&helix.edges[group]) {
            assert!(
                expected.iter().all(|rect| actual.contains(rect)),
                "missing antialiasing for {text:?}"
            );
        }
    }

    #[test]
    fn helix_displays_metrics_targets_and_feedback_in_the_designed_sizes() {
        let metrics = snapshot(60.0, full_hardware(), TelemetryStatus::Available);
        let feedback = OverlayFeedback {
            hotkey: Some(Hotkey::F6),
            cap_changed: true,
            hysteresis_changed: false,
            target_changed: false,
        };
        let helix = layout(&frame(&metrics, OverlayMode::Full, feedback), 1);
        assert_eq!((helix.width, helix.height), (WIDTH, HEIGHT));
        for (text, x, baseline, size, width, bold, group) in [
            ("FRAMECAP", 10, 22, TITLE_SIZE, 120, false, TITLE),
            ("60 FPS", 10, 58, 29, 118, true, ACCENT),
            ("10% LOW", 145, 42, 11, 110, false, LABEL),
            ("55 FPS", 145, 60, 17, 110, true, VALUE),
            ("FRAMETIME", 275, 42, 11, 99, false, LABEL),
            ("16.00 ms", 275, 60, 17, 99, true, FRAMETIME),
            ("01 // MODE", 10, 77, HEADING_SIZE, 100, true, LINE),
            ("02 // SENSORS", 10, 124, HEADING_SIZE, 100, true, LINE),
            ("03 // KEY BINDINGS", 10, 225, HEADING_SIZE, 100, true, LINE),
            ("AUTO • JIT", 16, 100, 13, 80, true, INK),
            ("CAP", 129, 92, 11, 68, false, FEEDBACK),
            ("60 FPS", 129, 105, 13, 68, true, FEEDBACK),
            ("HYS", 214, 92, 11, 70, false, LABEL),
            ("5.0 FPS", 214, 105, 13, 70, true, VALUE),
            ("TARGET", 303, 92, 11, 71, false, LABEL),
            ("55 FPS", 303, 105, 13, 71, true, VALUE),
            ("GPU", 10, 139, 12, 55, false, LABEL),
            ("43 °C", 67, 139, 12, 78, false, VALUE),
            ("VRM", 151, 139, 12, 45, false, LABEL),
            ("57 °C", 198, 139, 12, 66, false, VALUE),
            ("LOAD", 270, 139, 12, 44, false, LABEL),
            ("55 %", 316, 139, 12, 60, false, VALUE),
            ("CPU", 10, 158, 12, 55, false, LABEL),
            ("70 °C", 67, 158, 12, 78, false, VALUE),
            ("35 %", 316, 158, 12, 60, false, VALUE),
            ("GDDR6", 10, 177, 12, 55, false, LABEL),
            ("56 °C", 67, 177, 12, 78, false, VALUE),
            ("32.6 W", 198, 177, 12, 66, false, VALUE),
            ("1240 RPM", 316, 177, 12, 60, false, VALUE),
            ("GTT", 10, 196, 12, 55, false, LABEL),
            ("6000 / 12000 MB", 67, 196, 12, 180, false, VALUE),
            ("RAM", 270, 196, 12, 44, false, LABEL),
            ("4000 MB", 316, 196, 12, 60, false, VALUE),
            ("F6", 105, 242, 11, 20, true, FEEDBACK),
            ("-1 FPS", 133, 242, 11, 56, false, LABEL),
            ("F5", 14, 242, 11, 20, true, TEAL),
            ("OSD", 336, 242, 11, 38, false, LABEL),
        ] {
            assert_text(&helix, text, x, baseline, size, width, bold, group);
        }
        // The AUTO mode badge and the pressed key are the only fills.
        assert_eq!(helix.boxes.len(), 2);
    }

    #[test]
    fn corner_brackets_frame_the_panel_with_longer_but_not_thicker_arms() {
        let metrics = snapshot(60.0, full_hardware(), TelemetryStatus::Available);
        for (mode, scale) in [(OverlayMode::Full, 1), (OverlayMode::Fps, 2)] {
            let helix = layout(&frame(&metrics, mode, OverlayFeedback::default()), scale);
            let (width, height) = (helix.width * scale, helix.height * scale);
            let (arm, thickness) = (BRACKET_ARM * scale, BRACKET_THICKNESS * scale);
            for (x, y, w, h) in [
                (0, 0, arm, thickness),
                (0, 0, thickness, arm),
                (width - arm, 0, arm, thickness),
                (width - thickness, 0, thickness, arm),
                (0, height - thickness, arm, thickness),
                (0, height - arm, thickness, arm),
                (width - arm, height - thickness, arm, thickness),
                (width - thickness, height - arm, thickness, arm),
            ] {
                assert!(
                    helix.groups[CORNER].contains(&OverlayRect {
                        x,
                        y,
                        width: w,
                        height: h
                    }),
                    "missing bracket at {x},{y}"
                );
            }
        }
        // Arms are 4 px longer than the previous 10 px, with the same 2 px stroke.
        assert_eq!(BRACKET_ARM, 14);
        assert_eq!(BRACKET_THICKNESS, 2);
    }

    #[test]
    fn full_layout_stays_inside_the_panel_at_both_scales() {
        let mut hardware = full_hardware();
        hardware.fan_rpm = Some(12_400.0);
        hardware.gtt_total_mb = Some(120_000.0);
        let metrics = snapshot(144.0, hardware, TelemetryStatus::Available);
        for scale in [1, 2] {
            let helix = layout(
                &frame(&metrics, OverlayMode::Full, OverlayFeedback::default()),
                scale,
            );
            let (width, height) = (WIDTH * scale, HEIGHT * scale);
            assert!(helix
                .groups
                .iter()
                .flatten()
                .chain(helix.edges.iter().flatten().flatten())
                .chain(helix.boxes.iter().map(|item| &item.rect))
                .all(|rect| rect.x + rect.width <= width && rect.y + rect.height <= height));
        }
    }

    #[test]
    fn prepared_layout_moves_every_element_with_the_panel() {
        let metrics = snapshot(60.0, full_hardware(), TelemetryStatus::Available);
        for mode in [OverlayMode::Full, OverlayMode::Fps] {
            for scale in [1, 2] {
                let frame = frame(&metrics, mode, OverlayFeedback::default());
                let (width, height) = dimensions(mode);
                let extent = vk::Extent2D {
                    width: width * scale + ORIGIN_X,
                    height: height * scale + ORIGIN_Y,
                };
                let original = layout(&frame, scale).prepare(extent);
                let moved = layout(&frame, scale).prepare_at(extent, ORIGIN_X, ORIGIN_Y);
                assert_eq!(
                    moved.panel_rect(),
                    OverlayRect {
                        x: ORIGIN_X,
                        y: ORIGIN_Y,
                        width: width * scale,
                        height: height * scale,
                    }
                );
                for (before, after) in original.groups.iter().zip(&moved.groups) {
                    assert_eq!(before.len(), after.len());
                    for (before, after) in before.iter().zip(after) {
                        assert_eq!(after.rect.offset.x, before.rect.offset.x + ORIGIN_X as i32);
                        assert_eq!(after.rect.offset.y, before.rect.offset.y + ORIGIN_Y as i32);
                        assert_eq!(after.rect.extent, before.rect.extent);
                    }
                }
                for (before, after) in original
                    .edges
                    .iter()
                    .flatten()
                    .zip(moved.edges.iter().flatten())
                {
                    assert_eq!(before.len(), after.len());
                    for (before, after) in before.iter().zip(after) {
                        assert_eq!(
                            *after,
                            OverlayRect {
                                x: before.x + ORIGIN_X,
                                y: before.y + ORIGIN_Y,
                                ..*before
                            }
                        );
                    }
                }
                assert_eq!(original.boxes.len(), moved.boxes.len());
                for (before, after) in original.boxes.iter().zip(&moved.boxes) {
                    assert_eq!(after.style, before.style);
                    assert_eq!(
                        after.rect,
                        OverlayRect {
                            x: before.rect.x + ORIGIN_X,
                            y: before.rect.y + ORIGIN_Y,
                            ..before.rect
                        }
                    );
                }
                assert!(!moved.groups[CORNER].is_empty());
                assert!(moved.edges.iter().flatten().any(|edges| !edges.is_empty()));
                if mode == OverlayMode::Full {
                    assert!(!moved.boxes.is_empty());
                }
            }
        }
    }

    #[test]
    fn prepared_layout_clips_after_applying_origin() {
        let mut canvas = Helix::new(20, 20, 1, FontFamily::Mono);
        canvas.rect(CORNER, 12, 12, 3, 3);
        canvas.edges[LABEL][0].push(OverlayRect {
            x: 12,
            y: 12,
            width: 3,
            height: 1,
        });
        canvas.styled_box(12, 12, 3, 3, BoxStyle::ModeAuto);
        let prepared = canvas.prepare_at(
            vk::Extent2D {
                width: 15,
                height: 15,
            },
            2,
            2,
        );
        assert_eq!(
            prepared.groups[CORNER][0].rect.offset,
            vk::Offset2D { x: 14, y: 14 }
        );
        assert_eq!(
            prepared.groups[CORNER][0].rect.extent,
            vk::Extent2D {
                width: 1,
                height: 1
            }
        );
        assert_eq!(
            prepared.edges[LABEL][0],
            [OverlayRect {
                x: 14,
                y: 14,
                width: 1,
                height: 1
            }]
        );
        assert_eq!(
            prepared.boxes[0].rect,
            OverlayRect {
                x: 14,
                y: 14,
                width: 3,
                height: 3
            }
        );
    }

    #[test]
    fn compact_mode_shows_only_the_fps_value() {
        let metrics = snapshot(60.0, HardwareMetrics::default(), TelemetryStatus::Available);
        let helix = layout(
            &frame(&metrics, OverlayMode::Fps, OverlayFeedback::default()),
            1,
        );
        assert_eq!((helix.width, helix.height), (COMPACT_WIDTH, COMPACT_HEIGHT));
        assert_text(&helix, "FPS", 14, 19, 11, 60, false, LABEL);
        assert_text(&helix, "60", 14, 46, 27, 108, true, ACCENT);
        assert!(helix.groups[VALUE].is_empty() && helix.groups[INK].is_empty());
        assert!(helix.boxes.is_empty());
    }

    #[test]
    fn collect_and_unavailable_states_use_placeholders() {
        let metrics = snapshot(
            0.0,
            HardwareMetrics::default(),
            TelemetryStatus::Unsupported,
        );
        let mut frame = frame(&metrics, OverlayMode::Full, OverlayFeedback::default());
        frame.target_fps = None;
        frame.auto_algorithm_index = None;
        frame.auto_algorithm_label = None;
        frame.hysteresis = None;
        frame.auto_target = None;
        let helix = layout(&frame, 1);
        for (text, x, baseline, size, width, bold, group) in [
            ("COLLECT", 10, 58, 21, 118, true, ACCENT),
            ("COLLECT", 145, 60, 17, 110, true, VALUE),
            ("OFF", 16, 100, 13, 80, true, LABEL),
            ("OFF FPS", 129, 105, 13, 68, true, VALUE),
            ("--", 214, 105, 13, 70, true, VALUE),
            ("N/A", 316, 139, 12, 60, false, VALUE),
            ("-- / -- MB", 67, 196, 12, 180, false, VALUE),
            ("--", 133, 242, 11, 56, false, LABEL),
        ] {
            assert_text(&helix, text, x, baseline, size, width, bold, group);
        }
        // Without a GTT total there is no bar.
        assert!(helix.groups[TEAL]
            .iter()
            .all(|rect| (rect.y, rect.width, rect.height) != (200, 6, 6)));
        assert!(helix.groups[LINE]
            .iter()
            .all(|rect| !(rect.y == 200 && rect.width == 6)));
    }

    #[test]
    fn gtt_bar_fills_segments_proportionally() {
        for (used, expected) in [
            (0.0, 0),
            (6000.0, 5),
            (11_000.0, 9),
            (12_000.0, 10),
            (20_000.0, 10),
        ] {
            let mut hardware = full_hardware();
            hardware.gtt_used_mb = Some(used);
            let metrics = snapshot(60.0, hardware, TelemetryStatus::Available);
            let helix = layout(
                &frame(&metrics, OverlayMode::Full, OverlayFeedback::default()),
                1,
            );
            let filled = helix.groups[TEAL]
                .iter()
                .filter(|rect| (rect.y, rect.width, rect.height) == (200, 6, 6))
                .count();
            assert_eq!(filled, expected, "{used} MB");
            let empty = helix.groups[LINE]
                .iter()
                .filter(|rect| rect.y == 200 && rect.width == 6)
                .count();
            assert_eq!(empty, GTT_SEGMENTS as usize - expected, "{used} MB");
        }
    }

    #[test]
    fn mode_badge_is_framed_and_filled_fully_for_auto_half_for_manual_and_not_for_off() {
        let metrics = snapshot(60.0, full_hardware(), TelemetryStatus::Available);
        let badge = |auto: Option<&'static str>, target: Option<f32>| {
            let mut frame = frame(&metrics, OverlayMode::Full, OverlayFeedback::default());
            frame.auto_algorithm_label = auto;
            frame.auto_algorithm_index = auto.map(|_| 0);
            frame.target_fps = target;
            let helix = layout(&frame, 1);
            let badges: Vec<BoxStyle> = helix
                .boxes
                .iter()
                .filter(|item| item.rect.y == 83)
                .map(|item| item.style)
                .collect();
            (helix, badges)
        };
        let (helix, auto) = badge(Some("JIT"), Some(60.0));
        assert_eq!(auto, [BoxStyle::ModeAuto]);
        assert_text(&helix, "AUTO • JIT", 16, 100, 13, 80, true, INK);
        let (helix, manual) = badge(None, Some(60.0));
        assert_eq!(manual, [BoxStyle::ModeManual]);
        assert_text(&helix, "MANUAL", 16, 100, 13, 80, true, VALUE);
        let (helix, off) = badge(None, None);
        assert!(off.is_empty());
        assert_text(&helix, "OFF", 16, 100, 13, 80, true, LABEL);
        // The teal frame surrounds the badge in every state.
        for helix in [
            &helix,
            &badge(None, Some(60.0)).0,
            &badge(Some("JIT"), Some(60.0)).0,
        ] {
            for (x, y, width, height) in [
                (10, 83, 92, 1),
                (10, 107, 92, 1),
                (10, 83, 1, 25),
                (101, 83, 1, 25),
            ] {
                assert!(helix.groups[LINE].contains(&OverlayRect {
                    x,
                    y,
                    width,
                    height
                }));
            }
        }
    }

    #[test]
    fn scale_doubles_only_when_the_panel_fits() {
        let extent = |width, height| vk::Extent2D { width, height };
        assert_eq!(scale_for(OverlayMode::Full, extent(3840, 2160), 3840), 2);
        assert_eq!(scale_for(OverlayMode::Fps, extent(3840, 2160), 3840), 2);
        assert_eq!(scale_for(OverlayMode::Full, extent(1920, 1080), 3840), 1);
        assert_eq!(scale_for(OverlayMode::Full, extent(3840, 400), 3840), 1);
        assert_eq!(
            scale_for(OverlayMode::Full, extent(WIDTH * 2, HEIGHT * 2), 1),
            1
        );
        assert_eq!(
            scale_for(
                OverlayMode::Full,
                extent(WIDTH * 2 + ORIGIN_X, HEIGHT * 2 + ORIGIN_Y),
                1
            ),
            2
        );
    }

    #[test]
    fn bundled_monospace_fonts_cover_every_symbol_helix_draws() {
        let fonts = FontCache::new(FontFamily::Mono);
        for character in "•°…%/+-.0123456789".chars() {
            assert_ne!(
                fonts.regular.lookup_glyph_index(character),
                0,
                "{character}"
            );
            assert_ne!(fonts.bold.lookup_glyph_index(character), 0, "{character}");
        }
    }
}
