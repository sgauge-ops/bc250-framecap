//! Text and rectangle batching shared by the models drawn with the SDF background shader.
//!
//! A [`Canvas`] collects glyph spans and boxes in logical units, grouped by color. Once prepared
//! for a swapchain extent, the groups are drawn with clears and the antialiased edges with the
//! SDF pipeline.

use super::{OverlayFeedback, OverlayFrame, OverlayMode, OverlayRect};
use crate::layer::config::{AutoMetric, AutoTarget, Hotkey};
use ash::vk;
use fontdue::{Font, FontSettings, Metrics};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub(super) const COVERAGE_LEVELS: usize = 8;
pub(super) const RECTANGLES_PER_DRAW: usize = 6;

/// Everything that changes the layout of a frame; a canvas is rebuilt only when it differs.
#[derive(Clone, PartialEq)]
pub(super) struct FrameKey {
    timestamp: Instant,
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

impl FrameKey {
    /// Captures the layout-relevant fields of a frame, including the metrics timestamp.
    pub(super) fn new(frame: &OverlayFrame<'_>) -> Self {
        Self {
            timestamp: frame.metrics.timestamp(),
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

    /// Compares everything except the timestamp, i.e. whether the same controls are displayed.
    pub(super) fn same_controls(&self, other: &Self) -> bool {
        self.mode == other.mode
            && self.target_fps == other.target_fps
            && self.auto_algorithm_index == other.auto_algorithm_index
            && self.auto_algorithm_label == other.auto_algorithm_label
            && self.hysteresis == other.hysteresis
            && self.auto_target == other.auto_target
            && self.hotkeys == other.hotkeys
            && self.manual_step_fps == other.manual_step_fps
            && self.feedback == other.feedback
    }
}

/// Bundled font family used to rasterize overlay text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FontFamily {
    Sans,
    Mono,
}

/// Rasterized glyph: font metrics plus horizontal spans `(x, y, width, coverage level)`.
pub(super) struct Glyph {
    pub(super) metrics: Metrics,
    pub(super) spans: Vec<(u32, u32, u32, usize)>,
}

/// Regular and bold fonts of one family with a cache of rasterized glyph spans.
pub(super) struct FontCache {
    pub(super) regular: Font,
    pub(super) bold: Font,
    glyphs: HashMap<(char, u32, bool), Glyph>,
}

impl FontCache {
    /// Loads the bundled regular and bold fonts for the family.
    pub(super) fn new(family: FontFamily) -> Self {
        let (regular, bold) = match family {
            FontFamily::Sans => (
                include_bytes!("../fonts/FiraSans-Regular.ttf") as &[u8],
                include_bytes!("../fonts/FiraSans-Bold.ttf") as &[u8],
            ),
            FontFamily::Mono => (
                include_bytes!("../fonts/FiraMono-Regular.ttf") as &[u8],
                include_bytes!("../fonts/FiraMono-Bold.ttf") as &[u8],
            ),
        };
        Self {
            regular: Font::from_bytes(regular, FontSettings::default())
                .expect("bundled regular font must be valid"),
            bold: Font::from_bytes(bold, FontSettings::default())
                .expect("bundled bold font must be valid"),
            glyphs: HashMap::new(),
        }
    }

    /// Returns the cached glyph, rasterizing it into quantized-coverage row spans on first use.
    pub(super) fn glyph(&mut self, character: char, size: u32, bold: bool) -> &Glyph {
        self.glyphs
            .entry((character, size, bold))
            .or_insert_with(|| {
                let font = if bold { &self.bold } else { &self.regular };
                let (metrics, bitmap) = font.rasterize(character, size as f32);
                let mut spans = Vec::new();
                for y in 0..metrics.height {
                    let mut x = 0;
                    while x < metrics.width {
                        let coverage =
                            (usize::from(bitmap[y * metrics.width + x]) * COVERAGE_LEVELS + 127)
                                / 255;
                        if coverage == 0 {
                            x += 1;
                            continue;
                        }
                        let start = x;
                        while x < metrics.width
                            && (usize::from(bitmap[y * metrics.width + x]) * COVERAGE_LEVELS + 127)
                                / 255
                                == coverage
                        {
                            x += 1;
                        }
                        spans.push((start as u32, y as u32, (x - start) as u32, coverage));
                    }
                }
                Glyph { metrics, spans }
            })
    }
}

/// Returns the process-wide lazily created font cache for the family.
fn fonts(family: FontFamily) -> &'static Mutex<FontCache> {
    static SANS: OnceLock<Mutex<FontCache>> = OnceLock::new();
    static MONO: OnceLock<Mutex<FontCache>> = OnceLock::new();
    let cell = match family {
        FontFamily::Sans => &SANS,
        FontFamily::Mono => &MONO,
    };
    cell.get_or_init(|| Mutex::new(FontCache::new(family)))
}

/// A rectangle tagged with a model-specific style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CanvasBox<S> {
    pub(super) rect: OverlayRect,
    pub(super) style: S,
}

/// Rectangles and glyph spans of one overlay frame, in `G` color groups, with boxes styled as `S`.
pub(super) struct Canvas<const G: usize, S> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) scale: u32,
    family: FontFamily,
    pub(super) groups: [Vec<OverlayRect>; G],
    pub(super) edges: [[Vec<OverlayRect>; COVERAGE_LEVELS - 1]; G],
    pub(super) boxes: Vec<CanvasBox<S>>,
}

/// A [`Canvas`] converted to swapchain-clipped Vulkan rectangles.
pub(super) struct PreparedCanvas<const G: usize, S> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) scale: u32,
    pub(super) groups: [Vec<vk::ClearRect>; G],
    pub(super) edges: [[Vec<OverlayRect>; COVERAGE_LEVELS - 1]; G],
    pub(super) boxes: Vec<CanvasBox<S>>,
}

/// Clips a rectangle to the extent, returning `None` if nothing remains visible.
pub(super) fn clip_rect(rect: OverlayRect, extent: vk::Extent2D) -> Option<OverlayRect> {
    if rect.x >= extent.width || rect.y >= extent.height {
        return None;
    }
    let width = rect.width.min(extent.width - rect.x);
    let height = rect.height.min(extent.height - rect.y);
    if width == 0 || height == 0 {
        return None;
    }
    Some(OverlayRect {
        width,
        height,
        ..rect
    })
}

impl<const G: usize, S: Copy> Canvas<G, S> {
    /// Creates an empty canvas of logical `width`×`height`, scaled to pixels by `scale`.
    pub(super) fn new(width: u32, height: u32, scale: u32, family: FontFamily) -> Self {
        Self {
            width,
            height,
            scale,
            family,
            groups: std::array::from_fn(|_| Vec::new()),
            edges: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            boxes: Vec::new(),
        }
    }

    /// Converts the canvas for drawing at the extent's origin.
    pub(super) fn prepare(self, extent: vk::Extent2D) -> PreparedCanvas<G, S> {
        self.prepare_at(extent, 0, 0)
    }

    /// Translates the canvas by `(x, y)` and clips everything to the extent.
    pub(super) fn prepare_at(self, extent: vk::Extent2D, x: u32, y: u32) -> PreparedCanvas<G, S> {
        let translate = |mut rect: OverlayRect| {
            rect.x = rect.x.saturating_add(x);
            rect.y = rect.y.saturating_add(y);
            rect
        };
        let convert = |rectangles: Vec<OverlayRect>| {
            rectangles
                .into_iter()
                .map(translate)
                .filter_map(|rect| clip_rect(rect, extent))
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
                .collect()
        };
        let groups = self.groups.map(convert);
        let edges = self.edges.map(|levels| {
            levels.map(|rectangles| {
                rectangles
                    .into_iter()
                    .map(translate)
                    .filter_map(|rect| clip_rect(rect, extent))
                    .collect()
            })
        });
        PreparedCanvas {
            width: self.width,
            height: self.height,
            scale: self.scale,
            groups,
            edges,
            boxes: self
                .boxes
                .into_iter()
                .map(|item| CanvasBox {
                    rect: translate(item.rect),
                    ..item
                })
                .collect(),
        }
    }

    /// Adds a solid rectangle, given in logical units, to a color group.
    pub(super) fn rect(&mut self, group: usize, x: u32, y: u32, width: u32, height: u32) {
        self.groups[group].push(OverlayRect {
            x: x * self.scale,
            y: y * self.scale,
            width: width * self.scale,
            height: height * self.scale,
        });
    }

    /// Adds a styled box, given in logical units, for the SDF pipeline.
    pub(super) fn styled_box(&mut self, x: u32, y: u32, width: u32, height: u32, style: S) {
        self.boxes.push(CanvasBox {
            rect: OverlayRect {
                x: x * self.scale,
                y: y * self.scale,
                width: width * self.scale,
                height: height * self.scale,
            },
            style,
        });
    }

    /// Draws left-aligned, top-anchored text fitted to `width`; returns its right edge.
    pub(super) fn text(
        &mut self,
        text: &str,
        x: u32,
        y: u32,
        size: u32,
        width: u32,
        bold: bool,
        group: usize,
    ) -> u32 {
        self.draw_text(text, x, y, size, width, bold, group, false, false)
    }

    /// Draws text shrunk (down to 8 px) or ellipsized to fit `width`; returns its right edge.
    pub(super) fn draw_text(
        &mut self,
        text: &str,
        x: u32,
        y: u32,
        size: u32,
        width: u32,
        bold: bool,
        group: usize,
        align_baseline: bool,
        align_right: bool,
    ) -> u32 {
        let mut cache = fonts(self.family).lock().expect("font cache poisoned");
        let x = x * self.scale;
        let y = y * self.scale;
        let width = width * self.scale;
        let mut size = size * self.scale;
        let baseline = y + size;
        let min_size = 8 * self.scale;
        let text_width = |cache: &mut FontCache, text: &str, size| {
            text.chars()
                .map(|c| cache.glyph(c, size, bold).metrics.advance_width)
                .sum::<f32>()
        };
        while size > min_size && text_width(&mut cache, text, size) > width as f32 {
            size -= self.scale;
        }
        let mut visible = text.to_owned();
        if text_width(&mut cache, &visible, size) > width as f32 {
            while !visible.is_empty()
                && text_width(&mut cache, &format!("{visible}…"), size) > width as f32
            {
                visible.pop();
            }
            visible.push('…');
        }
        let limit = x.saturating_add(width);
        let mut cursor = if align_right {
            (limit as f32 - text_width(&mut cache, &visible, size)).max(x as f32)
        } else {
            x as f32
        };
        let mut right = x;
        let baseline = if align_baseline { baseline } else { y + size };
        for character in visible.chars() {
            let glyph = cache.glyph(character, size, bold);
            let gx = cursor.round() as i32 + glyph.metrics.xmin;
            let gy = baseline as i32 - glyph.metrics.height as i32 - glyph.metrics.ymin;
            for &(dx, dy, span_width, coverage) in &glyph.spans {
                let left = gx + dx as i32;
                let top = gy + dy as i32;
                if left >= x as i32 && top >= 0 && left < limit as i32 {
                    let rect = OverlayRect {
                        x: left as u32,
                        y: top as u32,
                        width: span_width.min(limit - left as u32),
                        height: 1,
                    };
                    right = right.max(rect.x + rect.width);
                    if coverage == COVERAGE_LEVELS {
                        self.groups[group].push(rect);
                    } else {
                        self.edges[group][coverage - 1].push(rect);
                    }
                }
            }
            cursor += glyph.metrics.advance_width;
            if cursor >= limit as f32 {
                break;
            }
        }
        right.div_ceil(self.scale)
    }
}

impl<const G: usize, S> PreparedCanvas<G, S> {
    /// Draws every color group: solid spans with clears, antialiased edges with the SDF pipeline.
    pub(super) fn draw_groups(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        pipeline_layout: vk::PipelineLayout,
        colors: [[f32; 4]; G],
    ) {
        for (group, (rectangles, color)) in self.groups.iter().zip(colors).enumerate() {
            if !rectangles.is_empty() {
                let attachment = [vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(0)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: color },
                    })];
                unsafe { device.cmd_clear_attachments(command_buffer, &attachment, rectangles) };
            }
            for (index, edges) in self.edges[group].iter().enumerate() {
                let mut color = color;
                color[3] = (index + 1) as f32 / COVERAGE_LEVELS as f32;
                for chunk in edges.chunks(RECTANGLES_PER_DRAW) {
                    draw_rectangles(
                        device,
                        command_buffer,
                        pipeline_layout,
                        extent,
                        chunk,
                        color,
                        0.0,
                        0.0,
                    );
                }
            }
        }
    }
}

/// Draws up to [`RECTANGLES_PER_DRAW`] instanced rectangles with the SDF shader.
///
/// A positive `radius` rounds the corners, `border > 0` draws an outline and `border < 0` adds a
/// vertical gradient; a zero `radius` fills the rectangles with a flat color.
pub(super) fn draw_rectangles(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    extent: vk::Extent2D,
    rectangles: &[OverlayRect],
    color: [f32; 4],
    radius: f32,
    border: f32,
) {
    debug_assert!(!rectangles.is_empty() && rectangles.len() <= RECTANGLES_PER_DRAW);
    let mut constants = [0.0f32; 32];
    for (index, rect) in rectangles.iter().enumerate() {
        let x = rect.x as f32;
        let y = rect.y as f32;
        constants[index * 4..index * 4 + 4].copy_from_slice(&[
            2.0 * x / extent.width as f32 - 1.0,
            2.0 * y / extent.height as f32 - 1.0,
            2.0 * (x + rect.width as f32) / extent.width as f32 - 1.0,
            2.0 * (y + rect.height as f32) / extent.height as f32 - 1.0,
        ]);
    }
    constants[24..28].copy_from_slice(&color);
    constants[28..32].copy_from_slice(&[extent.width as f32, extent.height as f32, radius, border]);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            constants.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&constants),
        )
    };
    unsafe {
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytes,
        );
        device.cmd_draw(command_buffer, 6, rectangles.len() as u32, 0, 0);
    }
}
