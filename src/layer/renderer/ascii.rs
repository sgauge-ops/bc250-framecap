//! Text-grid overlay drawn with a built-in 5×7 bitmap font.

use super::background::BackgroundPreparation;
use super::canvas::FrameKey;
use super::format::{
    auto_key_hints, format_auto_target, format_fps, format_memory_temperature, format_optional,
    fps_value, frametime_value,
};
use super::pipeline::OverlayPipeline;
use super::{OverlayFeedback, OverlayFrame, OverlayMode, OverlayModel, OverlayRect, OverlayStyle};
use crate::layer::config::{AutoMetric, Hotkey};
use crate::layer::lib::{MetricsSnapshot, TelemetryStatus};
use ash::vk;
use std::collections::VecDeque;
use std::time::Instant;

const FIRST_COLUMN: usize = 17;
const SECOND_COLUMN: usize = 36;
const THIRD_COLUMN: usize = 55;
const GRAPH_MAX_MS: f32 = 25.0;
const GRAPH_HEIGHT: u32 = 26;

/// Rolling history of frame times, drawn as a one-pixel-per-sample line graph.
struct FrametimeGraph {
    last_present: Option<Instant>,
    samples: VecDeque<f32>,
    capacity: usize,
}

impl FrametimeGraph {
    /// Creates an empty graph whose capacity is the pixel width left of the third column.
    fn new(extent: vk::Extent2D) -> Self {
        let left = 8 + (THIRD_COLUMN as u32 + 1) * 6;
        Self {
            last_present: None,
            samples: VecDeque::new(),
            capacity: extent.width.saturating_sub(left + 8) as usize,
        }
    }

    /// Drops all samples and the last presentation timestamp.
    fn reset(&mut self) {
        self.last_present = None;
        self.samples.clear();
    }

    /// Appends the time since the previous present, evicting the oldest sample when full.
    fn record_present(&mut self, now: Instant) {
        let Some(previous) = self.last_present else {
            self.last_present = Some(now);
            return;
        };
        let Some(elapsed) = now.checked_duration_since(previous) else {
            return;
        };
        self.last_present = Some(now);
        let frametime_ms = elapsed.as_secs_f32() * 1000.0;
        if self.capacity > 0 {
            if self.samples.len() == self.capacity {
                self.samples.pop_front();
            }
            self.samples.push_back(frametime_ms);
        }
    }

    /// Converts samples into 1×1 clear rects, newest at the right edge, capped at `GRAPH_MAX_MS`.
    fn rectangles(
        &self,
        extent: vk::Extent2D,
        background_width: u32,
        delimiter_column: usize,
        scale: u32,
    ) -> Vec<vk::ClearRect> {
        let left = 8u32.saturating_add(
            (delimiter_column as u32)
                .saturating_add(1)
                .saturating_mul(6)
                .saturating_mul(scale),
        );
        let right = background_width.min(extent.width).saturating_sub(8);
        let width = right.saturating_sub(left) as usize;
        let top = 8;
        let height = (GRAPH_HEIGHT * scale).min(extent.height.saturating_sub(top));
        if width == 0 || height == 0 {
            return Vec::new();
        }
        self.samples
            .iter()
            .rev()
            .take(width)
            .enumerate()
            .map(|(index, frametime_ms)| {
                let offset = (frametime_ms.min(GRAPH_MAX_MS) / GRAPH_MAX_MS * (height - 1) as f32)
                    .round() as u32;
                vk::ClearRect::default()
                    .rect(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: (right - 1 - index as u32) as i32,
                                y: (top + height - 1 - offset) as i32,
                            })
                            .extent(vk::Extent2D {
                                width: 1,
                                height: 1,
                            }),
                    )
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect()
    }
}

/// Namespace and settings for the bitmap-font text formatting and rasterization.
pub struct Renderer {
    pub color: [u8; 4],
}

impl Renderer {
    /// Creates a text renderer with opaque white foreground text.
    pub fn new() -> Self {
        Self {
            color: [255, 255, 255, 255],
        }
    }

    /// Formats the displayed frame-rate and frame-pacing metrics.
    pub fn fps_text(metrics: &MetricsSnapshot) -> String {
        format!("{:>5} FPS", fps_value(metrics))
    }

    /// Formats the main row: FPS, 10% low, and frame time in fixed columns.
    pub fn text(metrics: &MetricsSnapshot) -> String {
        let low_10_fps = metrics.low_10_fps();
        let frametime = frametime_value(metrics).to_ascii_uppercase();
        let low_10 = if low_10_fps.is_finite() && low_10_fps >= 0.0 {
            format!("{low_10_fps:.0} FPS")
        } else {
            "COLLECT".to_owned()
        };
        let mut text = format!("FPS: {:>4}", fps_value(metrics));
        append_at_column(&mut text, FIRST_COLUMN, &format!("| 10% LOW: {low_10}"));
        append_at_column(&mut text, SECOND_COLUMN, &format!("| FT: {frametime:>8}"));
        append_at_column(&mut text, THIRD_COLUMN, "|");
        text
    }

    /// Formats the GPU temperature, VRM temperature, utilization, and PPT row.
    pub fn gpu_sensor_text(metrics: &MetricsSnapshot) -> String {
        let hardware = metrics.hardware_metrics();
        let gpu = match metrics.gpu_measurement().status {
            TelemetryStatus::Available if metrics.gpu_measurement().value.is_finite() => {
                format!("{:.0} %", metrics.gpu_measurement().value)
            }
            TelemetryStatus::Expired => "STALE".to_owned(),
            TelemetryStatus::Error => "ERR".to_owned(),
            TelemetryStatus::Unsupported | TelemetryStatus::Available => "N/A".to_owned(),
        };
        sensor_text(
            "GPU",
            hardware.gpu_edge_temp_c,
            hardware.gpu_vrm_temp_c,
            gpu,
            "PPT",
            hardware.ppt_w,
            1,
            "W",
        )
    }

    /// Formats the CPU temperature, VRM temperature, utilization, and fan-speed row.
    pub fn cpu_sensor_text(metrics: &MetricsSnapshot) -> String {
        let hardware = metrics.hardware_metrics();
        sensor_text(
            "CPU",
            hardware.cpu_temp_c,
            hardware.cpu_vrm_temp_c,
            hardware
                .cpu_load_percent
                .filter(|load| load.is_finite())
                .map(|load| format!("{load:.0} %"))
                .unwrap_or_else(|| "N/A".to_owned()),
            "RPM",
            hardware.fan_rpm,
            0,
            "RPM",
        )
    }

    /// Formats the GDDR6 temperature, GTT usage, and system RAM usage row.
    pub fn memory_sensor_text(metrics: &MetricsSnapshot) -> String {
        let hardware = metrics.hardware_metrics();
        let temperature = format_memory_temperature(hardware.memory_temp);
        let mut text = if temperature.len() > 3 {
            format!("GDDR6: {temperature} °C")
        } else {
            format!("GDDR6:{temperature:>3} °C")
        };
        append_at_column(
            &mut text,
            FIRST_COLUMN,
            &format!(
                "| GTT: {:>5} / {:>5} MB",
                format_optional(hardware.gtt_used_mb, 0),
                format_optional(hardware.gtt_total_mb, 0),
            ),
        );
        append_at_column(
            &mut text,
            THIRD_COLUMN,
            &format!("| RAM: {:>5} MB", format_optional(hardware.ram_used_mb, 0)),
        );
        text
    }

    /// Formats the target row and its configured hotkey hints.
    pub fn target_text(
        target_fps: Option<f32>,
        auto_algorithm_index: Option<usize>,
        auto_metric: Option<AutoMetric>,
        hysteresis: Option<f32>,
        hotkeys: [Hotkey; 4],
        manual_step_fps: f32,
    ) -> String {
        let (down_hint, up_hint) = if auto_algorithm_index.is_some() {
            auto_key_hints(auto_metric, hysteresis)
        } else if target_fps.is_some_and(|fps| fps.is_finite() && fps > 0.0) {
            (
                format!("-{} FPS", format_fps(manual_step_fps)),
                format!("+{} FPS", format_fps(manual_step_fps)),
            )
        } else {
            ("--".into(), "--".into())
        };
        let mut text = format!("[{}] CYCLE MODE", hotkeys[0].label());
        append_at_column(
            &mut text,
            FIRST_COLUMN,
            &format!("| [{}]  {}", hotkeys[2].label(), down_hint),
        );
        append_at_column(
            &mut text,
            SECOND_COLUMN,
            &format!("| [{}]  {}", hotkeys[1].label(), up_hint),
        );
        append_at_column(
            &mut text,
            THIRD_COLUMN,
            &format!("| [{}] CYCLE OSD", hotkeys[3].label()),
        );
        text
    }

    /// Formats the current mode, cap, and active algorithm hysteresis.
    pub fn status_text(
        target_fps: Option<f32>,
        auto_algorithm_label: Option<&str>,
        hysteresis: Option<f32>,
        auto_metric: Option<AutoMetric>,
        auto_target: Option<&str>,
    ) -> String {
        let target = target_fps
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .map(format_fps)
            .unwrap_or_else(|| "OFF".to_owned());
        let mode = auto_algorithm_label.map_or_else(
            || {
                if target_fps.is_some_and(|fps| fps.is_finite() && fps > 0.0) {
                    "MANUAL".to_owned()
                } else {
                    "OFF".to_owned()
                }
            },
            |label| format!("AUTO ({label})"),
        );
        let mut text = format!("MODE: {mode}");
        append_at_column(
            &mut text,
            FIRST_COLUMN,
            &format!("| CAP: {:>6} FPS", target),
        );
        append_at_column(
            &mut text,
            SECOND_COLUMN,
            &format!(
                "| {}",
                Self::format_hysteresis_field(hysteresis, auto_metric)
            ),
        );
        append_at_column(
            &mut text,
            THIRD_COLUMN,
            &format!("| TARGET: {}", auto_target.unwrap_or("--")),
        );
        text
    }

    /// Formats the hysteresis value with its metric unit, or placeholders when unavailable.
    fn format_hysteresis_field(hysteresis: Option<f32>, metric: Option<AutoMetric>) -> String {
        match (hysteresis, metric) {
            (Some(value), Some(metric)) => {
                format!("HYS: {value:>5.1} {}", metric.hysteresis_unit())
            }
            _ => "HYS:    --".to_owned(),
        }
    }

    /// Returns rectangles highlighting the selected status text.
    fn feedback_rects(
        text: &str,
        extent: [u32; 2],
        scale: u32,
        top: u32,
        highlighted_text: &str,
    ) -> Vec<OverlayRect> {
        let Some(start) = text.find(highlighted_text) else {
            return Vec::new();
        };
        let left = 8u32.saturating_add((start as u32).saturating_mul(6 * scale));
        Self::text_rects_at_x(highlighted_text, extent, scale, top, left)
    }

    /// Returns rectangles covering only the numeric FPS value within the row text.
    fn fps_value_rects(
        text: &str,
        metrics: &MetricsSnapshot,
        extent: [u32; 2],
        scale: u32,
        top: u32,
    ) -> Vec<OverlayRect> {
        Self::feedback_rects(text, extent, scale, top, &fps_value(metrics))
    }

    /// Returns rectangles covering the last occurrence of the frame-time value in the row text.
    fn frametime_value_rects(
        text: &str,
        metrics: &MetricsSnapshot,
        extent: [u32; 2],
        scale: u32,
        top: u32,
    ) -> Vec<OverlayRect> {
        let value = frametime_value(metrics).to_ascii_uppercase();
        let Some(start) = text.rfind(&value) else {
            return Vec::new();
        };
        let left = 8u32.saturating_add((start as u32).saturating_mul(6 * scale));
        Self::text_rects_at_x(&value, extent, scale, top, left)
    }

    /// Rasterizes text at the default top margin and clips it to the extent.
    pub fn text_rects(text: &str, extent: [u32; 2], scale: u32) -> Vec<OverlayRect> {
        Self::text_rects_at(text, extent, scale, 8)
    }

    /// Rasterizes text at a chosen vertical offset.
    fn text_rects_at(text: &str, extent: [u32; 2], scale: u32, top: u32) -> Vec<OverlayRect> {
        Self::text_rects_at_x(text, extent, scale, top, 8)
    }

    /// Rasterizes text at explicit coordinates, clipping glyphs to the extent.
    fn text_rects_at_x(
        text: &str,
        extent: [u32; 2],
        scale: u32,
        top: u32,
        left: u32,
    ) -> Vec<OverlayRect> {
        if scale == 0 || extent[0] <= 16 || extent[1] <= 16 {
            return Vec::new();
        }
        let mut rects = Vec::new();
        let mut cursor_x = left;
        for character in text.chars() {
            if cursor_x + 5 * scale > extent[0].saturating_sub(8) {
                break;
            }
            for (row, bits) in glyph(character).iter().copied().enumerate() {
                let cell_y = top.saturating_add(row as u32 * scale);
                if cell_y.saturating_add(scale) > extent[1] {
                    continue;
                }
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        rects.push(OverlayRect {
                            x: cursor_x + column * scale,
                            y: cell_y,
                            width: scale,
                            height: scale,
                        });
                    }
                }
            }
            cursor_x += 6 * scale;
        }
        rects
    }
}

/// Formats a hardware row: temperature, VRM temperature, load, and one extra labeled metric.
fn sensor_text(
    label: &str,
    temperature_c: Option<f32>,
    vrm_temperature_c: Option<f32>,
    load_display: String,
    final_metric_label: &str,
    final_metric_value: Option<f32>,
    final_metric_precision: usize,
    final_metric_unit: &str,
) -> String {
    let mut text = format!("{label}: {:>4} °C", format_optional(temperature_c, 0));
    append_at_column(
        &mut text,
        FIRST_COLUMN,
        &format!("| VRM: {:>4} °C", format_optional(vrm_temperature_c, 0)),
    );
    append_at_column(
        &mut text,
        SECOND_COLUMN,
        &format!("| Load: {:>6}", load_display),
    );
    append_at_column(
        &mut text,
        THIRD_COLUMN,
        &format!(
            "| {final_metric_label}: {:>5} {final_metric_unit}",
            format_optional(final_metric_value, final_metric_precision)
        ),
    );
    text
}

/// Pads text with spaces until it reaches the requested character column.
fn append_at_column(text: &mut String, column: usize, segment: &str) {
    let padding = column.saturating_sub(text.chars().count());
    text.push_str(&" ".repeat(padding));
    text.push_str(segment);
}

/// Computes the top Y of each text row plus separator Y positions; compact mode has one row.
fn ascii_row_layout(scale: u32, full: bool) -> (u32, u32, u32, u32, u32, Vec<u32>) {
    let metrics_top = 8 + (GRAPH_HEIGHT - 7) * scale / 2;
    if !full {
        return (
            metrics_top,
            metrics_top,
            metrics_top,
            metrics_top,
            metrics_top,
            Vec::new(),
        );
    }
    let line_step = 7 * scale + 3;
    let sep1 = metrics_top + (GRAPH_HEIGHT + 2) * scale;
    let status = sep1 + 4 * scale;
    let sep2 = status + 9 * scale;
    let gpu = sep2 + 4 * scale;
    let cpu = gpu + line_step;
    let memory = cpu + line_step;
    let sep3 = memory + 9 * scale;
    let target = sep3 + 4 * scale;
    (status, gpu, cpu, memory, target, vec![sep1, sep2, sep3])
}

/// Returns the seven-row bitmap for a supported overlay glyph.
fn glyph(character: char) -> [u8; 7] {
    match character {
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 2, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '°' => [6, 9, 9, 6, 0, 0, 0],
        '.' => [0, 0, 0, 0, 0, 12, 12],
        ':' => [0, 12, 12, 0, 12, 12, 0],
        '|' => [4, 4, 4, 4, 4, 4, 4],
        '/' => [1, 1, 2, 4, 8, 16, 16],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '%' => [17, 2, 4, 8, 16, 17, 0],
        '(' => [2, 4, 8, 8, 8, 4, 2],
        ')' => [8, 4, 2, 2, 2, 4, 8],
        '+' => [0, 4, 4, 31, 4, 4, 0],
        '>' => [16, 8, 4, 2, 4, 8, 16],
        '=' => [0, 0, 31, 0, 31, 0, 0],
        '[' => [14, 8, 8, 8, 8, 8, 14],
        ']' => [14, 2, 2, 2, 2, 2, 14],
        'a' => [0, 0, 14, 1, 15, 17, 15],
        'd' => [1, 1, 15, 17, 17, 17, 15],
        'e' => [0, 0, 14, 17, 31, 16, 14],
        'f' => [6, 8, 30, 8, 8, 8, 8],
        'h' => [16, 16, 22, 25, 17, 17, 17],
        'i' => [4, 0, 12, 4, 4, 4, 14],
        'o' => [0, 0, 14, 17, 17, 17, 14],
        'p' => [0, 0, 30, 17, 30, 16, 16],
        's' => [0, 0, 15, 16, 14, 1, 30],
        't' => [8, 8, 30, 8, 8, 9, 6],
        'u' => [0, 0, 17, 17, 17, 19, 13],
        'w' => [0, 0, 17, 17, 21, 21, 10],
        'y' => [0, 0, 17, 17, 15, 1, 14],
        _ => [0; 7],
    }
}

/// ASCII overlay model: owns the pipeline, frame-time graph, and background-prepared geometry.
pub(super) struct AsciiModel {
    extent: vk::Extent2D,
    background_opacity: f32,
    fps_color: [u8; 3],
    scale_2x_min_width_px: u32,
    pipeline: OverlayPipeline,
    frametime_graph: FrametimeGraph,
    background: Option<BackgroundPreparation<PreparedAscii>>,
    worker_failed: bool,
    prepared: Option<(FrameKey, OverlayMode, PreparedAscii)>,
}

impl AsciiModel {
    /// Feeds a presentation timestamp into the frame-time graph.
    pub(super) fn record_graph_present(&mut self, now: Instant) {
        self.frametime_graph.record_present(now);
    }

    /// Clears the frame-time graph history.
    pub(super) fn reset_graph(&mut self) {
        self.frametime_graph.reset();
    }
}

impl AsciiModel {
    /// Builds the background-quad pipeline and an empty model for the given style.
    fn create(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        style: OverlayStyle,
    ) -> Result<Self, vk::Result> {
        let pipeline = OverlayPipeline::new(
            device,
            render_pass,
            style.extent,
            include_bytes!("../shaders/overlay_background.vert.spv"),
            include_bytes!("../shaders/overlay_background.frag.spv"),
            20,
        )?;
        Ok(Self {
            extent: style.extent,
            background_opacity: style.background_opacity,
            fps_color: style.fps_color,
            scale_2x_min_width_px: style.scale_2x_min_width_px,
            pipeline,
            frametime_graph: FrametimeGraph::new(style.extent),
            background: None,
            worker_failed: false,
            prepared: None,
        })
    }
}

/// CPU-prepared layout and clear rectangles for one ASCII overlay frame.
struct PreparedAscii {
    scale: u32,
    background_width: u32,
    background_height: u32,
    delimiter_column: usize,
    separator_rectangles: Vec<vk::ClearRect>,
    metric_rectangles: Vec<vk::ClearRect>,
    fps_rectangles: Vec<vk::ClearRect>,
    frametime_rectangles: Vec<vk::ClearRect>,
    target_rectangles: Vec<vk::ClearRect>,
    status_rectangles: Vec<vk::ClearRect>,
    target_feedback_rectangles: Vec<vk::ClearRect>,
    status_feedback_rectangles: Vec<vk::ClearRect>,
}

/// Inputs the background worker needs to prepare ASCII frames.
struct AsciiPreparation {
    extent: vk::Extent2D,
    scale_2x_min_width_px: u32,
}

impl AsciiPreparation {
    /// Formats all text rows and rasterizes them into clear rectangles with feedback highlights.
    fn prepare(&self, frame: &OverlayFrame<'_>) -> PreparedAscii {
        let OverlayFrame {
            metrics,
            mode: overlay_mode,
            target_fps,
            auto_algorithm_index,
            auto_algorithm_label,
            hysteresis,
            auto_target,
            hotkeys,
            manual_step_fps,
            feedback,
        } = *frame;
        {
            let full = overlay_mode == OverlayMode::Full;
            let text = if full {
                Renderer::text(metrics)
            } else {
                Renderer::fps_text(metrics)
            };
            let (gpu_sensor_text, cpu_sensor_text, memory_sensor_text, target_text, status_text) =
                if full {
                    let formatted_auto_target =
                        auto_target.map(|target| format_auto_target(target, metrics));
                    (
                        Renderer::gpu_sensor_text(metrics),
                        Renderer::cpu_sensor_text(metrics),
                        Renderer::memory_sensor_text(metrics),
                        Renderer::target_text(
                            target_fps,
                            auto_algorithm_index,
                            auto_target.map(|(metric, _)| metric),
                            hysteresis,
                            hotkeys,
                            manual_step_fps,
                        ),
                        Renderer::status_text(
                            target_fps,
                            auto_algorithm_label,
                            hysteresis,
                            auto_target.map(|(metric, _)| metric),
                            formatted_auto_target.as_deref(),
                        ),
                    )
                } else {
                    (
                        String::new(),
                        String::new(),
                        String::new(),
                        String::new(),
                        String::new(),
                    )
                };
            let feedback = if full {
                feedback
            } else {
                OverlayFeedback::default()
            };
            let widest_line = [
                &text,
                &gpu_sensor_text,
                &cpu_sensor_text,
                &memory_sensor_text,
                &target_text,
                &status_text,
            ]
            .into_iter()
            .map(|line| line.chars().count() as u32)
            .max()
            .unwrap_or_default();
            let scale =
                overlay_text_scale(self.extent.width, self.scale_2x_min_width_px, widest_line);
            let metrics_top = if full {
                8 + (GRAPH_HEIGHT - 7) * scale / 2
            } else {
                8
            };
            let line_height = scale;
            let (
                status_top,
                gpu_sensor_top,
                cpu_sensor_top,
                memory_sensor_top,
                target_top,
                line_tops,
            ) = ascii_row_layout(scale, full);
            let background_width = (widest_line * 6 * scale + 20).min(self.extent.width);
            let background_height = (target_top + 7 * scale + 8).min(self.extent.height);
            let separator_rectangles = if full {
                let line_width =
                    (widest_line * 6 * scale).min(self.extent.width.saturating_sub(16));
                line_tops
                    .into_iter()
                    .map(|top| {
                        vk::ClearRect::default()
                            .rect(
                                vk::Rect2D::default()
                                    .offset(vk::Offset2D {
                                        x: 8,
                                        y: top as i32,
                                    })
                                    .extent(vk::Extent2D {
                                        width: line_width,
                                        height: line_height,
                                    }),
                            )
                            .base_array_layer(0)
                            .layer_count(1)
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

            let metric_rectangles = if full {
                vec![
                    (&text, metrics_top),
                    (&gpu_sensor_text, gpu_sensor_top),
                    (&cpu_sensor_text, cpu_sensor_top),
                    (&memory_sensor_text, memory_sensor_top),
                ]
            } else {
                vec![(&text, metrics_top)]
            }
            .into_iter()
            .flat_map(|(line, top)| {
                Renderer::text_rects_at(line, [self.extent.width, self.extent.height], scale, top)
            })
            .map(|rect| {
                vk::ClearRect::default()
                    .rect(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: rect.x as i32,
                                y: rect.y as i32,
                            })
                            .extent(vk::Extent2D {
                                width: rect.width,
                                height: rect.height,
                            }),
                    )
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect::<Vec<_>>();
            let fps_rectangles = Renderer::fps_value_rects(
                &text,
                metrics,
                [self.extent.width, self.extent.height],
                scale,
                metrics_top,
            )
            .into_iter()
            .map(|rect| {
                vk::ClearRect::default()
                    .rect(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: rect.x as i32,
                                y: rect.y as i32,
                            })
                            .extent(vk::Extent2D {
                                width: rect.width,
                                height: rect.height,
                            }),
                    )
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect::<Vec<_>>();
            let frametime_rectangles = if full {
                Renderer::frametime_value_rects(
                    &text,
                    metrics,
                    [self.extent.width, self.extent.height],
                    scale,
                    metrics_top,
                )
            } else {
                Vec::new()
            }
            .into_iter()
            .map(|rect| {
                vk::ClearRect::default()
                    .rect(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: rect.x as i32,
                                y: rect.y as i32,
                            })
                            .extent(vk::Extent2D {
                                width: rect.width,
                                height: rect.height,
                            }),
                    )
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect::<Vec<_>>();
            let target_rectangles = if full {
                Renderer::text_rects_at(
                    &target_text,
                    [self.extent.width, self.extent.height],
                    scale,
                    target_top,
                )
            } else {
                Vec::new()
            }
            .into_iter()
            .map(|rect| {
                vk::ClearRect::default()
                    .rect(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: rect.x as i32,
                                y: rect.y as i32,
                            })
                            .extent(vk::Extent2D {
                                width: rect.width,
                                height: rect.height,
                            }),
                    )
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect::<Vec<_>>();
            let status_rectangles = if full {
                Renderer::text_rects_at(
                    &status_text,
                    [self.extent.width, self.extent.height],
                    scale,
                    status_top,
                )
            } else {
                Vec::new()
            }
            .into_iter()
            .map(|rect| {
                vk::ClearRect::default()
                    .rect(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: rect.x as i32,
                                y: rect.y as i32,
                            })
                            .extent(vk::Extent2D {
                                width: rect.width,
                                height: rect.height,
                            }),
                    )
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect::<Vec<_>>();
            let target_feedback_rectangles = if full {
                feedback
                    .hotkey
                    .map(|hotkey| {
                        Renderer::feedback_rects(
                            &target_text,
                            [self.extent.width, self.extent.height],
                            scale,
                            target_top,
                            &format!("[{}]", hotkey.label()),
                        )
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            }
            .into_iter()
            .map(|rect| {
                vk::ClearRect::default()
                    .rect(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: rect.x as i32,
                                y: rect.y as i32,
                            })
                            .extent(vk::Extent2D {
                                width: rect.width,
                                height: rect.height,
                            }),
                    )
                    .base_array_layer(0)
                    .layer_count(1)
            })
            .collect::<Vec<_>>();
            let mut status_feedback_rectangles = Vec::new();
            if feedback.cap_changed {
                status_feedback_rectangles.extend(Renderer::feedback_rects(
                    &status_text,
                    [self.extent.width, self.extent.height],
                    scale,
                    status_top,
                    &format!(
                        "CAP: {:>6} FPS",
                        target_fps
                            .map(format_fps)
                            .unwrap_or_else(|| "OFF".to_owned())
                    ),
                ));
            }
            if feedback.hysteresis_changed {
                status_feedback_rectangles.extend(Renderer::feedback_rects(
                    &status_text,
                    [self.extent.width, self.extent.height],
                    scale,
                    status_top,
                    &Renderer::format_hysteresis_field(
                        hysteresis,
                        auto_target.map(|(metric, _)| metric),
                    ),
                ));
            }
            if feedback.target_changed {
                let target = auto_target
                    .map(|target| format_auto_target(target, metrics))
                    .unwrap_or_else(|| "--".to_owned());
                status_feedback_rectangles.extend(Renderer::feedback_rects(
                    &status_text,
                    [self.extent.width, self.extent.height],
                    scale,
                    status_top,
                    &format!("TARGET: {target}"),
                ));
            }
            let status_feedback_rectangles = status_feedback_rectangles
                .into_iter()
                .map(|rect| {
                    vk::ClearRect::default()
                        .rect(
                            vk::Rect2D::default()
                                .offset(vk::Offset2D {
                                    x: rect.x as i32,
                                    y: rect.y as i32,
                                })
                                .extent(vk::Extent2D {
                                    width: rect.width,
                                    height: rect.height,
                                }),
                        )
                        .base_array_layer(0)
                        .layer_count(1)
                })
                .collect::<Vec<_>>();
            PreparedAscii {
                scale,
                background_width,
                background_height,
                delimiter_column: text.chars().count().saturating_sub(1),
                separator_rectangles,
                metric_rectangles,
                fps_rectangles,
                frametime_rectangles,
                target_rectangles,
                status_rectangles,
                target_feedback_rectangles,
                status_feedback_rectangles,
            }
        }
    }
}

/// Spawns the background worker that prepares ASCII frames.
fn background_ascii(
    extent: vk::Extent2D,
    scale_2x_min_width_px: u32,
) -> std::io::Result<BackgroundPreparation<PreparedAscii>> {
    let preparation = AsciiPreparation {
        extent,
        scale_2x_min_width_px,
    };
    BackgroundPreparation::new("ascii", move |frame| preparation.prepare(frame))
}

impl OverlayModel for AsciiModel {
    /// Creates the model and its Vulkan pipeline.
    fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        style: OverlayStyle,
    ) -> Result<Self, vk::Result> {
        Self::create(device, render_pass, style)
    }

    /// Records the background quad and per-group text/graph clears for the latest prepared frame.
    fn record(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame: &OverlayFrame<'_>,
    ) {
        if self.background.is_none() && !self.worker_failed {
            match background_ascii(self.extent, self.scale_2x_min_width_px) {
                Ok(background) => self.background = Some(background),
                Err(error) => {
                    crate::layer::diagnostics::error(format!(
                        "failed to start ascii overlay preparation worker: {error}"
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
        let graph_rectangles = if frame.mode == OverlayMode::Full {
            self.frametime_graph.rectangles(
                self.extent,
                prepared.background_width,
                prepared.delimiter_column,
                prepared.scale,
            )
        } else {
            Vec::new()
        };
        if self.background_opacity > 0.0 {
            let rect = [
                -1.0,
                -1.0,
                2.0 * prepared.background_width as f32 / self.extent.width as f32 - 1.0,
                2.0 * prepared.background_height as f32 / self.extent.height as f32 - 1.0,
                self.background_opacity.clamp(0.0, 1.0),
            ];
            let constants = unsafe {
                std::slice::from_raw_parts(rect.as_ptr().cast::<u8>(), std::mem::size_of_val(&rect))
            };
            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline,
                );
                device.cmd_push_constants(
                    command_buffer,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    constants,
                );
                device.cmd_draw(command_buffer, 6, 1, 0, 0);
            }
        }
        {
            let separator_rectangles = &prepared.separator_rectangles;
            let metric_rectangles = &prepared.metric_rectangles;
            let fps_rectangles = &prepared.fps_rectangles;
            let target_rectangles = &prepared.target_rectangles;
            let status_rectangles = &prepared.status_rectangles;
            let target_feedback_rectangles = &prepared.target_feedback_rectangles;
            let status_feedback_rectangles = &prepared.status_feedback_rectangles;
            let foreground = [vk::ClearAttachment::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .color_attachment(0)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.92, 0.96, 0.92, 1.0],
                    },
                })];
            unsafe {
                if !separator_rectangles.is_empty() {
                    let separator_foreground = [vk::ClearAttachment::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .color_attachment(0)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.28, 0.33, 0.39, 1.0],
                            },
                        })];
                    device.cmd_clear_attachments(
                        command_buffer,
                        &separator_foreground,
                        separator_rectangles,
                    );
                }
                if !metric_rectangles.is_empty() {
                    device.cmd_clear_attachments(command_buffer, &foreground, metric_rectangles);
                }
                if !graph_rectangles.is_empty() {
                    let graph_foreground = [vk::ClearAttachment::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .color_attachment(0)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.35, 0.85, 1.0, 1.0],
                            },
                        })];
                    device.cmd_clear_attachments(
                        command_buffer,
                        &graph_foreground,
                        &graph_rectangles,
                    );
                }
                if !fps_rectangles.is_empty() {
                    let [red, green, blue] =
                        self.fps_color.map(|channel| f32::from(channel) / 255.0);
                    let fps_foreground = [vk::ClearAttachment::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .color_attachment(0)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [red, green, blue, 1.0],
                            },
                        })];
                    device.cmd_clear_attachments(command_buffer, &fps_foreground, fps_rectangles);
                }
                if crate::layer::capture::is_active() && !prepared.frametime_rectangles.is_empty() {
                    let capture_foreground = [vk::ClearAttachment::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .color_attachment(0)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [1.0, 0.15, 0.15, 1.0],
                            },
                        })];
                    device.cmd_clear_attachments(
                        command_buffer,
                        &capture_foreground,
                        &prepared.frametime_rectangles,
                    );
                }
                let status_foreground = [vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(0)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.35, 0.85, 1.0, 1.0],
                        },
                    })];
                let target_foreground = [vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(0)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.35, 0.85, 1.0, 1.0],
                        },
                    })];
                if !status_rectangles.is_empty() {
                    device.cmd_clear_attachments(
                        command_buffer,
                        &status_foreground,
                        status_rectangles,
                    );
                }
                if !target_rectangles.is_empty() {
                    device.cmd_clear_attachments(
                        command_buffer,
                        &target_foreground,
                        target_rectangles,
                    );
                }
                let feedback_foreground = [vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(0)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.72, 0.20, 1.0],
                        },
                    })];
                if !target_feedback_rectangles.is_empty() {
                    device.cmd_clear_attachments(
                        command_buffer,
                        &feedback_foreground,
                        target_feedback_rectangles,
                    );
                }
                if !status_feedback_rectangles.is_empty() {
                    device.cmd_clear_attachments(
                        command_buffer,
                        &feedback_foreground,
                        status_feedback_rectangles,
                    );
                }
            }
        }
    }

    /// Releases the Vulkan pipeline resources.
    fn destroy(&self, device: &ash::Device) {
        self.pipeline.destroy(device);
    }
}

/// Chooses 2× text scale when the extent is wide enough for the widest line, otherwise 1×.
fn overlay_text_scale(width: u32, minimum_width: u32, widest_line: u32) -> u32 {
    if width >= minimum_width && widest_line.saturating_mul(12).saturating_add(16) <= width {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::config::AutoTarget;
    use crate::layer::lib::{GpuMeasurement, HardwareMetrics};
    use std::time::{Duration, Instant};

    const HOTKEYS: [Hotkey; 4] = [Hotkey::F5, Hotkey::F7, Hotkey::F6, Hotkey::F8];

    /// Builds the target row using the test hotkey layout and standard step.
    fn target_text(target_fps: Option<f32>, auto_algorithm_index: Option<usize>) -> String {
        Renderer::target_text(
            target_fps,
            auto_algorithm_index,
            auto_algorithm_index.map(|_| AutoMetric::Fps),
            Some(5.0),
            HOTKEYS,
            5.0,
        )
    }

    fn char_column(line: &str, marker: &str) -> usize {
        line.find(marker)
            .map(|byte_index| line[..byte_index].chars().count())
            .unwrap()
    }

    #[test]
    fn ascii_prepares_in_background_while_graph_updates_per_presentation() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let now = Instant::now();
        let metrics = MetricsSnapshot::new_with_fps(
            16.0,
            60.0,
            0.5,
            55.0,
            GpuMeasurement {
                value: 50.0,
                timestamp: now,
                status: TelemetryStatus::Available,
                device_id: None,
            },
            now,
        );
        let mut frame = OverlayFrame {
            metrics: &metrics,
            mode: OverlayMode::Full,
            target_fps: Some(60.0),
            auto_algorithm_index: None,
            auto_algorithm_label: None,
            hysteresis: None,
            auto_target: None,
            hotkeys: HOTKEYS,
            manual_step_fps: 5.0,
            feedback: OverlayFeedback::default(),
        };
        let mut worker = background_ascii(extent, 3840).unwrap();
        let mut prepared = None;
        worker.update(&frame, &mut prepared);
        assert!(prepared.is_none());
        let deadline = Instant::now() + Duration::from_secs(5);
        while prepared.is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
            worker.update(&frame, &mut prepared);
        }
        let (_, mode, layout) = prepared.as_ref().unwrap();
        assert_eq!(*mode, OverlayMode::Full);
        assert!(!layout.metric_rectangles.is_empty());
        assert!(!layout.status_rectangles.is_empty());

        let mut graph = FrametimeGraph::new(extent);
        for ms in [0, 16, 34, 49] {
            graph.record_present(now + Duration::from_millis(ms));
        }
        let points = graph.rectangles(
            extent,
            layout.background_width,
            layout.delimiter_column,
            layout.scale,
        );
        assert_eq!(points.len(), 3);
        graph.record_present(now + Duration::from_millis(69));
        assert_eq!(
            graph
                .rectangles(
                    extent,
                    layout.background_width,
                    layout.delimiter_column,
                    layout.scale
                )
                .len(),
            4
        );

        frame.mode = OverlayMode::Fps;
        let deadline = Instant::now() + Duration::from_secs(5);
        while !prepared
            .as_ref()
            .is_some_and(|(_, mode, _)| *mode == OverlayMode::Fps)
            && Instant::now() < deadline
        {
            worker.update(&frame, &mut prepared);
            std::thread::sleep(Duration::from_millis(1));
        }
        let (_, mode, layout) = prepared.as_ref().unwrap();
        assert_eq!(*mode, OverlayMode::Fps);
        assert!(layout.status_rectangles.is_empty());
    }

    #[test]
    fn frametime_graph_tracks_intervals_and_bounds_history() {
        let left = 8 + (THIRD_COLUMN as u32 + 1) * 6;
        let mut graph = FrametimeGraph::new(vk::Extent2D {
            width: left + 8 + 3,
            height: 100,
        });
        let start = Instant::now();
        graph.record_present(start);
        assert!(graph.samples.is_empty());
        graph.record_present(start + Duration::from_millis(16));
        assert_eq!(graph.samples.iter().copied().collect::<Vec<_>>(), [16.0]);
        for ms in [34, 49, 69, 89] {
            graph.record_present(start + Duration::from_millis(ms));
        }
        assert_eq!(graph.samples.len(), 3);
        for (actual, expected) in graph.samples.iter().zip([15.0, 20.0, 20.0]) {
            assert!((actual - expected).abs() < 0.001);
        }
        graph.record_present(start + Duration::from_millis(80));
        assert_eq!(
            graph.samples.iter().copied().collect::<Vec<_>>(),
            [15.0, 20.0, 20.0]
        );
        graph.record_present(start + Duration::from_millis(99));
        assert_eq!(
            graph.samples.iter().copied().collect::<Vec<_>>(),
            [20.0, 20.0, 10.0]
        );
        graph.reset();
        assert!(graph.samples.is_empty());
        graph.record_present(start + Duration::from_secs(2));
        assert!(graph.samples.is_empty());
    }

    #[test]
    fn frametime_graph_uses_one_pixel_per_sample_and_clips_at_25_ms() {
        let extent = vk::Extent2D {
            width: 370,
            height: 100,
        };
        let mut graph = FrametimeGraph::new(extent);
        graph.samples.extend([0.0, 12.5, 25.0, 40.0]);
        let rects = graph.rectangles(extent, 370, THIRD_COLUMN, 1);
        let points = rects
            .iter()
            .map(|rect| (rect.rect.offset.x, rect.rect.offset.y))
            .collect::<Vec<_>>();
        assert_eq!(points, [(361, 8), (360, 8), (359, 20), (358, 33)]);
        assert!(rects
            .iter()
            .all(|rect| rect.rect.extent.width == 1 && rect.rect.extent.height == 1));

        let clipped = graph.rectangles(extent, 355, THIRD_COLUMN, 1);
        assert_eq!(clipped.len(), 3);
        let doubled = graph.rectangles(
            vk::Extent2D {
                width: 800,
                height: 100,
            },
            730,
            THIRD_COLUMN,
            2,
        );
        assert_eq!(doubled[0].rect.offset.y, 8);
        assert_eq!(doubled[3].rect.offset.y, 59);
        assert!(graph
            .rectangles(
                vk::Extent2D {
                    width: 100,
                    height: 100
                },
                100,
                THIRD_COLUMN,
                1
            )
            .is_empty());
        assert!(graph
            .rectangles(
                vk::Extent2D {
                    width: 370,
                    height: 8
                },
                370,
                THIRD_COLUMN,
                1
            )
            .is_empty());
    }

    #[test]
    fn ascii_memory_row_formats_values_and_independent_unavailability() {
        use crate::layer::lib::MemoryTemperature;

        let now = Instant::now();
        let gpu = GpuMeasurement {
            value: 0.0,
            timestamp: now,
            status: TelemetryStatus::Unsupported,
            device_id: None,
        };
        let snapshot = |hardware| {
            MetricsSnapshot::new_with_hardware_metrics(
                16.0,
                60.0,
                0.5,
                55.0,
                gpu.clone(),
                hardware,
                now,
            )
        };
        let metrics = snapshot(HardwareMetrics {
            memory_temp: Some(MemoryTemperature {
                celsius: 56.0,
                lower_bound: false,
            }),
            gtt_used_mb: Some(512.0),
            gtt_total_mb: Some(12000.0),
            ram_used_mb: Some(4000.0),
            ..HardwareMetrics::default()
        });
        let line = Renderer::memory_sensor_text(&metrics);
        assert_eq!(
            line,
            "GDDR6: 56 °C     | GTT:   512 / 12000 MB               | RAM:  4000 MB"
        );
        assert_eq!(char_column(&line, "| GTT:"), FIRST_COLUMN);
        assert_eq!(char_column(&line, "| RAM:"), THIRD_COLUMN);
        assert_eq!(
            char_column(&line, " °C"),
            char_column(&Renderer::gpu_sensor_text(&metrics), " °C")
        );
        assert_eq!(
            char_column(&line, " °C"),
            char_column(&Renderer::cpu_sensor_text(&metrics), " °C")
        );
        assert!(line
            .chars()
            .filter(|c| !c.is_whitespace())
            .all(|c| glyph(c) != [0; 7]));

        let partial = snapshot(HardwareMetrics {
            memory_temp: Some(MemoryTemperature {
                celsius: 120.0,
                lower_bound: true,
            }),
            gtt_total_mb: Some(12000.0),
            ..HardwareMetrics::default()
        });
        let line = Renderer::memory_sensor_text(&partial);
        assert!(line.contains("GDDR6: >=120 °C"));
        assert!(line.contains("| GTT:    -- / 12000 MB"));
        assert!(line.contains("| RAM:    -- MB"));
        assert!(line
            .chars()
            .filter(|c| !c.is_whitespace())
            .all(|c| glyph(c) != [0; 7]));
        let unavailable = Renderer::memory_sensor_text(&snapshot(HardwareMetrics::default()));
        assert!(unavailable.contains("GDDR6: -- °C"));
        assert!(unavailable.contains("| GTT:    -- /    -- MB"));
        assert!(unavailable.contains("| RAM:    -- MB"));
    }

    #[test]
    fn fps_only_text_contains_no_other_metrics() {
        let now = Instant::now();
        let metrics = MetricsSnapshot::new_with_fps(
            16.0,
            60.0,
            0.5,
            55.0,
            GpuMeasurement {
                value: 50.0,
                timestamp: now,
                status: TelemetryStatus::Available,
                device_id: None,
            },
            now,
        );
        assert_eq!(Renderer::fps_text(&metrics), "   60 FPS");
        assert!(Renderer::text(&metrics).contains("10% LOW"));
        let collecting = MetricsSnapshot::new_with_fps(
            0.0,
            -1.0,
            -1.0,
            -1.0,
            metrics.gpu_measurement().clone(),
            now,
        );
        assert_eq!(Renderer::fps_text(&collecting), "COLLECT FPS");
        for (snapshot, value, fps_column, full_column) in
            [(&metrics, "60", 3, 7), (&collecting, "COLLECT", 0, 5)]
        {
            for (text, column) in [
                (Renderer::fps_text(snapshot), fps_column),
                (Renderer::text(snapshot), full_column),
            ] {
                for scale in [1, 2] {
                    let rects = Renderer::fps_value_rects(&text, snapshot, [1200, 100], scale, 25);
                    assert!(!rects.is_empty());
                    assert_eq!(
                        rects,
                        Renderer::text_rects_at_x(
                            value,
                            [1200, 100],
                            scale,
                            25,
                            8 + column * 6 * scale,
                        )
                    );
                }
            }
        }
    }

    #[test]
    /// Checks the configured threshold and ensures doubled text remains within the swapchain.
    fn overlay_text_scale_obeys_threshold_and_fit() {
        assert_eq!(overlay_text_scale(3839, 3840, 100), 1);
        assert_eq!(overlay_text_scale(3840, 3840, 100), 2);
        assert_eq!(overlay_text_scale(3840, 3840, 320), 1);
    }

    #[test]
    /// Checks documented overlay row labels and ordering.
    fn overlay_rows_follow_the_documented_order_and_labels() {
        let manual = target_text(Some(60.0), None);
        let auto = target_text(Some(75.0), Some(1));
        let off = target_text(None, None);
        assert_eq!(
            manual,
            "[F5] CYCLE MODE  | [F6]  -5 FPS     | [F7]  +5 FPS     | [F8] CYCLE OSD"
        );
        assert_eq!(
            auto,
            "[F5] CYCLE MODE  | [F6]  -1 FPS     | [F7]  +1 FPS     | [F8] CYCLE OSD"
        );
        assert!(manual.contains("| [F6]  -5 FPS"));
        assert!(manual.contains("| [F7]  +5 FPS"));
        assert!(manual.contains("| [F8] CYCLE OSD"));
        assert!(auto.contains("| [F6]  -1 FPS"));
        assert!(auto.contains("| [F7]  +1 FPS"));
        assert!(auto.contains("| [F8] CYCLE OSD"));
        assert!(off.contains("[F6]  --") && off.contains("[F7]  --"));
        let status = Renderer::status_text(Some(60.0), None, None, None, None);
        assert!(status.contains("MODE: MANUAL"));
        assert!(status.contains("| CAP:     60 FPS"));
        assert!(status.contains("| HYS:    --"));
        let auto_status = Renderer::status_text(
            Some(75.0),
            Some("VRM"),
            Some(5.0),
            Some(AutoMetric::GpuVrmTempC),
            Some("100 °C"),
        );
        assert!(auto_status.contains("MODE: AUTO (VRM)"));
        assert!(Renderer::status_text(
            Some(75.0),
            Some("JIT"),
            Some(3.0),
            Some(AutoMetric::Fps),
            Some("60 FPS")
        )
        .contains("MODE: AUTO (JIT)"));
        assert!(Renderer::status_text(
            Some(75.0),
            Some("FAN"),
            Some(5.0),
            Some(AutoMetric::FanRpm),
            Some("2000 RPM")
        )
        .contains("MODE: AUTO (FAN)"));
        assert!(auto_status
            .chars()
            .filter(|character| !character.is_whitespace())
            .all(|character| glyph(character) != [0; 7]));
        assert!(Renderer::status_text(None, None, None, None, None).contains("CAP:    OFF FPS"));
        let text = target_text(Some(60.0), None);
        assert!(text
            .chars()
            .filter(|character| !character.is_whitespace())
            .all(|character| glyph(character) != [0; 7]));
        let rects = Renderer::text_rects_at(&text, [640, 480], 1, 8);
        assert!(!rects.is_empty());
        assert!(rects.iter().all(|rect| rect.y >= 8 && rect.y < 15));
        assert!(rects.iter().all(|rect| rect.x + rect.width <= 632));

        let now = Instant::now();
        let metrics_snapshot = MetricsSnapshot::new_with_hardware_metrics(
            16.0,
            60.0,
            0.5,
            55.0,
            GpuMeasurement {
                value: 55.0,
                timestamp: now,
                status: TelemetryStatus::Available,
                device_id: None,
            },
            HardwareMetrics {
                cpu_temp_c: Some(70.0),
                gpu_vrm_temp_c: Some(57.0),
                cpu_vrm_temp_c: Some(61.0),
                gpu_edge_temp_c: Some(43.0),
                cpu_load_percent: Some(35.0),
                gpu_power_w: Some(5.8),
                cpu_power_w: Some(5.2),
                ppt_w: Some(32.61),
                fan_rpm: Some(1240.0),
                ..HardwareMetrics::default()
            },
            now,
        );
        let metrics = Renderer::text(&metrics_snapshot);
        assert_eq!(
            metrics,
            "FPS:   60        | 10% LOW: 55 FPS  | FT: 16.00 MS     |"
        );
        assert_eq!(
            Renderer::status_text(
                Some(60.0),
                Some("JIT"),
                Some(5.0),
                Some(AutoMetric::Fps),
                Some("60 FPS")
            ),
            "MODE: AUTO (JIT) | CAP:     60 FPS  | HYS:   5.0 FPS   | TARGET: 60 FPS"
        );
        let metrics_rects = Renderer::text_rects_at(&metrics, [640, 480], 1, 17);
        assert!(metrics_rects.iter().all(|rect| rect.y >= 17 && rect.y < 31));
        let gpu_sensors = Renderer::gpu_sensor_text(&metrics_snapshot);
        let cpu_sensors = Renderer::cpu_sensor_text(&metrics_snapshot);
        assert_eq!(
            gpu_sensors,
            "GPU:   43 °C     | VRM:   57 °C     | Load:   55 %     | PPT:  32.6 W"
        );
        assert_eq!(
            cpu_sensors,
            "CPU:   70 °C     | VRM:   61 °C     | Load:   35 %     | RPM:  1240 RPM"
        );
        assert!(gpu_sensors.starts_with("GPU:   43 °C"));
        assert!(gpu_sensors.contains("| VRM:   57 °C"));
        assert!(gpu_sensors.contains("| Load:   55 %"));
        assert!(gpu_sensors.contains("| PPT:  32.6 W"));
        assert!(cpu_sensors.starts_with("CPU:   70 °C"));
        assert!(cpu_sensors.contains("| VRM:   61 °C"));
        assert!(cpu_sensors.contains("| Load:   35 %"));
        assert!(cpu_sensors.contains("| RPM:  1240 RPM"));
        assert!(gpu_sensors
            .chars()
            .filter(|character| !character.is_whitespace())
            .all(|character| glyph(character) != [0; 7]));
        for (line, marker) in [
            (&manual, "| [F6]"),
            (&status, "| CAP:"),
            (&metrics, "| 10% LOW"),
            (&gpu_sensors, "| VRM:"),
            (&cpu_sensors, "| VRM:"),
        ] {
            assert_eq!(
                char_column(line, marker),
                FIRST_COLUMN,
                "{marker:?} in {line:?}"
            );
        }
        for (line, marker) in [
            (&manual, "| [F7]"),
            (&status, "| HYS:"),
            (&gpu_sensors, "| Load:"),
            (&cpu_sensors, "| Load:"),
        ] {
            assert_eq!(char_column(line, marker), SECOND_COLUMN);
        }
        for (line, marker) in [
            (&manual, "| [F8]"),
            (&status, "| TARGET:"),
            (&gpu_sensors, "| PPT:"),
            (&cpu_sensors, "| RPM:"),
        ] {
            assert_eq!(char_column(line, marker), THIRD_COLUMN);
        }
        assert_eq!(char_column(&metrics, "| FT:"), SECOND_COLUMN);
        assert_eq!(metrics.chars().last(), Some('|'));
        assert_eq!(metrics.chars().count() - 1, THIRD_COLUMN);
        assert!([&manual, &status, &metrics, &gpu_sensors, &cpu_sensors,]
            .into_iter()
            .all(|line| !line.trim_start().starts_with('|')));
        for unit in [" °C", " %"] {
            assert_eq!(
                char_column(&gpu_sensors, unit),
                char_column(&cpu_sensors, unit),
                "unit {unit:?} is not aligned"
            );
        }
        assert_eq!(
            char_column(&gpu_sensors, " W"),
            cpu_sensors
                .rfind(" RPM")
                .map(|index| cpu_sensors[..index].chars().count())
                .unwrap()
        );
    }

    #[test]
    fn status_row_shows_the_active_algorithm_target_and_unit() {
        let now = Instant::now();
        let gpu = GpuMeasurement {
            value: 50.0,
            timestamp: now,
            status: TelemetryStatus::Available,
            device_id: None,
        };
        let metrics = MetricsSnapshot::new_with_fps(16.0, 60.0, 0.5, 57.4, gpu.clone(), now);
        let collecting = MetricsSnapshot::new_with_fps(0.0, -1.0, -1.0, -1.0, gpu, now);
        for (label, target, expected) in [
            ("JIT", (AutoMetric::Fps, AutoTarget::Low10Fps), "57 FPS"),
            (
                "VRM",
                (AutoMetric::GpuVrmTempC, AutoTarget::Value(100.25)),
                "100.25 °C",
            ),
            (
                "FAN",
                (AutoMetric::FanRpm, AutoTarget::Value(1000.0)),
                "1000 RPM",
            ),
        ] {
            let value = format_auto_target(target, &metrics);
            assert_eq!(value, expected);
            let status = Renderer::status_text(
                Some(90.0),
                Some(label),
                Some(5.0),
                Some(target.0),
                Some(&value),
            );
            assert!(status.contains(&format!("| TARGET: {expected}")));
            assert_eq!(char_column(&status, "| TARGET:"), THIRD_COLUMN);
            assert!(status
                .chars()
                .filter(|character| !character.is_whitespace())
                .all(|character| glyph(character) != [0; 7]));
        }
        assert_eq!(
            format_auto_target((AutoMetric::Fps, AutoTarget::Low10Fps), &collecting),
            "COLLECT FPS"
        );
        for cap in [Some(90.0), None] {
            assert!(Renderer::status_text(cap, None, None, None, None).contains("| TARGET: --"));
        }
    }

    #[test]
    /// Checks that text columns remain anchored with the largest manual step.
    fn overlay_column_anchors_are_stable_with_maximum_manual_step() {
        let keys = Renderer::target_text(Some(240.0), None, None, None, HOTKEYS, 240.0);
        let status = Renderer::status_text(Some(240.0), None, None, None, None);
        assert_eq!(keys.find("| [F6]").unwrap(), FIRST_COLUMN);
        assert_eq!(keys.find("| [F7]").unwrap(), SECOND_COLUMN);
        assert_eq!(keys.find("| [F8]").unwrap(), THIRD_COLUMN);
        assert_eq!(status.find("| CAP:").unwrap(), FIRST_COLUMN);
        assert_eq!(status.find("| HYS:").unwrap(), SECOND_COLUMN);
        assert_eq!(status.find("| TARGET:").unwrap(), THIRD_COLUMN);
    }

    #[test]
    /// Checks that fractional target FPS values retain their decimal precision.
    fn target_text_preserves_fractional_fps_settings() {
        let text = Renderer::target_text(Some(77.5), None, None, None, HOTKEYS, 5.5);
        assert!(text.contains("| [F6]  -5.5 FPS"));
        assert!(text.contains("| [F7]  +5.5 FPS"));
        assert!(Renderer::status_text(
            Some(77.5),
            Some("JIT"),
            Some(5.5),
            Some(AutoMetric::Fps),
            Some("60 FPS")
        )
        .contains("HYS:   5.5 FPS"));
    }

    #[test]
    /// Checks custom key hints and highlighting of the pressed key.
    fn overlay_key_hints_follow_custom_bindings_and_highlight_the_pressed_key() {
        let hotkeys = [Hotkey::F8, Hotkey::F5, Hotkey::F6, Hotkey::F7];
        let text = Renderer::target_text(Some(60.0), None, None, None, hotkeys, 2.5);
        assert!(text.contains("[F8] CYCLE MODE"));
        assert!(text.contains("[F6]  -2.5 FPS"));
        let highlights = Renderer::feedback_rects(&text, [1200, 100], 1, 8, "[F5]");
        let f5_start = text.find("[F5]").unwrap() as u32 * 6 + 8;
        assert!(!highlights.is_empty());
        assert!(highlights
            .iter()
            .all(|rect| { rect.x >= f5_start && rect.x + rect.width <= f5_start + 24 }));
    }

    #[test]
    /// Checks that cap and hysteresis feedback highlight their status fields.
    fn cap_and_hysteresis_feedback_highlight_their_status_fields() {
        let status = Renderer::status_text(
            Some(62.5),
            Some("JIT"),
            Some(10.0),
            Some(AutoMetric::Fps),
            Some("60 FPS"),
        );
        for field in ["CAP:   62.5 FPS", "HYS:  10.0 FPS"] {
            let highlights = Renderer::feedback_rects(&status, [1200, 100], 1, 17, field);
            assert!(!highlights.is_empty());
        }
    }

    #[test]
    fn auto_hysteresis_and_hotkey_hints_use_metric_units() {
        for (metric, value, unit, step) in [
            (AutoMetric::Fps, 5.0, "FPS", "1"),
            (AutoMetric::GpuVrmTempC, 5.0, "°C", "5"),
            (AutoMetric::FanRpm, 50.0, "RPM", "50"),
        ] {
            let status = Renderer::status_text(
                Some(60.0),
                Some("AUTO"),
                Some(value),
                Some(metric),
                Some("TARGET"),
            );
            let field = format!("HYS: {value:>5.1} {unit}");
            assert!(status.contains(&field));
            assert!(!Renderer::feedback_rects(&status, [1200, 100], 1, 17, &field).is_empty());
            assert_eq!(char_column(&status, "| TARGET:"), THIRD_COLUMN);
            let hints =
                Renderer::target_text(Some(60.0), Some(0), Some(metric), Some(value), HOTKEYS, 5.0);
            assert!(hints.contains(&format!("-{step} {unit}")));
            assert!(hints.contains(&format!("+{step} {unit}")));
            assert_eq!(char_column(&hints, "| [F8]"), THIRD_COLUMN);
        }
        let hints = Renderer::target_text(Some(60.0), Some(0), None, None, HOTKEYS, 5.0);
        assert!(hints.contains("| [F6]  --"));
        assert_eq!(
            auto_key_hints(Some(AutoMetric::FanRpm), Some(0.05)),
            ("-0.05 RPM".into(), "+0.05 RPM".into())
        );
        assert_eq!(
            auto_key_hints(Some(AutoMetric::FanRpm), Some(0.0)),
            ("--".into(), "--".into())
        );
        assert!(
            Renderer::status_text(Some(60.0), Some("AUTO"), Some(5.0), None, None)
                .contains("HYS:    --")
        );
    }

    #[test]
    /// Checks overlay labels while metrics are collecting or unavailable.
    fn overlay_text_shows_collection_and_unavailable_states() {
        let now = Instant::now();
        let metrics = MetricsSnapshot::new_with_fps(
            16.67,
            -1.0,
            -1.0,
            -1.0,
            GpuMeasurement {
                value: 0.0,
                timestamp: now,
                status: TelemetryStatus::Unsupported,
                device_id: None,
            },
            now,
        );
        let text = Renderer::text(&metrics);
        assert!(text.starts_with("FPS: COLLECT"));
        assert!(text.contains("| 10% LOW: COLLECT "));
        assert!(text.contains("| FT: 16.67 MS"));
        assert_eq!(char_column(&text, "| 10% LOW:"), FIRST_COLUMN);
        assert_eq!(char_column(&text, "| FT:"), SECOND_COLUMN);
        assert_eq!(text.chars().last(), Some('|'));
        assert_eq!(text.chars().count() - 1, THIRD_COLUMN);
    }

    #[test]
    fn frametime_shows_collection_for_invalid_values() {
        let now = Instant::now();
        for frametime in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let metrics = MetricsSnapshot::new_with_fps(
                frametime,
                60.0,
                0.5,
                55.0,
                GpuMeasurement {
                    value: 0.0,
                    timestamp: now,
                    status: TelemetryStatus::Unsupported,
                    device_id: None,
                },
                now,
            );
            let text = Renderer::text(&metrics);
            assert!(text.contains("| FT:  COLLECT"));
            assert_eq!(char_column(&text, "| FT:"), SECOND_COLUMN);
            assert_eq!(text.chars().count() - 1, THIRD_COLUMN);
        }
        let metrics = MetricsSnapshot::new(
            f32::MAX,
            -1.0,
            -1.0,
            GpuMeasurement {
                value: 0.0,
                timestamp: now,
                status: TelemetryStatus::Unsupported,
                device_id: None,
            },
            now,
        );
        let text = Renderer::text(&metrics);
        assert!(text.contains("| FT: 3.40E38 MS"));
        assert_eq!(text.chars().count() - 1, THIRD_COLUMN);
    }

    #[test]
    fn capture_highlight_targets_frametime_even_when_other_values_are_collecting() {
        let now = Instant::now();
        let metrics = MetricsSnapshot::new_with_fps(
            0.0,
            -1.0,
            -1.0,
            -1.0,
            GpuMeasurement {
                value: 0.0,
                timestamp: now,
                status: TelemetryStatus::Unsupported,
                device_id: None,
            },
            now,
        );
        let text = Renderer::text(&metrics);
        let rects = Renderer::frametime_value_rects(&text, &metrics, [800, 600], 1, 8);
        let expected = 8 + text.rfind("COLLECT").unwrap() as u32 * 6;
        assert!(!rects.is_empty());
        assert!(rects.iter().all(|rect| rect.x >= expected));
        assert!(rects
            .iter()
            .all(|rect| rect.x < 8 + THIRD_COLUMN as u32 * 6));
    }

    #[test]
    /// Checks visible labels for every GPU telemetry status.
    fn overlay_text_labels_all_gpu_telemetry_states() {
        let now = Instant::now();
        let text_for = |status| {
            let metrics = MetricsSnapshot::new_with_fps(
                16.67,
                60.0,
                0.5,
                55.0,
                GpuMeasurement {
                    value: 92.0,
                    timestamp: now,
                    status,
                    device_id: Some(0),
                },
                now,
            );
            Renderer::gpu_sensor_text(&metrics)
        };
        assert!(text_for(TelemetryStatus::Available).contains("Load:   92 %"));
        assert!(text_for(TelemetryStatus::Expired).contains("Load:  STALE"));
        assert!(text_for(TelemetryStatus::Unsupported).contains("Load:    N/A"));
        assert!(text_for(TelemetryStatus::Error).contains("Load:    ERR"));
    }

    #[test]
    /// Checks fixed text widths for metric and target columns.
    fn metric_and_target_columns_keep_fixed_text_widths() {
        let now = Instant::now();
        let gpu = GpuMeasurement {
            value: 95.0,
            timestamp: now,
            status: TelemetryStatus::Available,
            device_id: Some(0),
        };
        let fast = MetricsSnapshot::new_with_fps(10.0, 100.0, 0.1, 100.0, gpu.clone(), now);
        let slow = MetricsSnapshot::new_with_fps(35.0, 28.0, 8.5, 24.0, gpu, now);
        assert_eq!(Renderer::text(&fast).len(), Renderer::text(&slow).len());

        let manual_low = target_text(Some(60.0), None);
        let manual_high = target_text(Some(240.0), None);
        assert_eq!(manual_low.len(), manual_high.len());
        let status_low = Renderer::status_text(Some(1.0), None, None, None, None);
        let status_high = Renderer::status_text(Some(240.0), None, None, None, None);
        assert_eq!(status_low.len(), status_high.len());
    }

    #[test]
    /// Checks clipping of generated text rectangles to the swapchain extent.
    fn text_rects_clip_to_swapchain_extent() {
        let rects = Renderer::text_rects("FRAMECAP", [64, 32], 2);
        assert!(!rects.is_empty());
        assert!(rects.iter().all(|rect| rect.x + rect.width <= 64));
        assert!(rects.iter().all(|rect| rect.y + rect.height <= 32));
        let bottom_clipped = Renderer::text_rects_at("FRAMECAP", [64, 20], 2, 18);
        assert!(!bottom_clipped.is_empty());
        assert!(bottom_clipped.iter().all(|rect| rect.y + rect.height <= 20));
        assert!(Renderer::text_rects("FRAMECAP", [8, 8], 2).is_empty());
    }

    #[test]
    /// Checks that the ascii overlay sections and separators are strictly ordered vertically.
    fn ascii_overlay_sections_and_separators_do_not_overlap() {
        for scale in [1, 2] {
            let metrics_top = 8;
            let line_height = scale;
            let (
                status_top,
                gpu_sensor_top,
                cpu_sensor_top,
                memory_sensor_top,
                target_top,
                separators,
            ) = ascii_row_layout(scale, true);
            let [sep1_top, sep2_top, sep3_top] = separators.as_slice() else {
                panic!("expected three separators")
            };
            let background_height = target_top + 7 * scale + 8;

            assert!(8 + GRAPH_HEIGHT * scale < *sep1_top);
            assert!(metrics_top + 7 * scale < *sep1_top);
            assert!(*sep1_top + line_height < status_top);
            assert!(status_top + 7 * scale < *sep2_top);
            assert!(*sep2_top + line_height < gpu_sensor_top);
            assert!(gpu_sensor_top + 7 * scale < cpu_sensor_top);
            assert!(cpu_sensor_top + 7 * scale < memory_sensor_top);
            assert!(memory_sensor_top + 7 * scale < *sep3_top);
            assert!(*sep3_top + line_height < target_top);
            assert!(target_top + 7 * scale < background_height);
        }
    }
}
