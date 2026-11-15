//! Overlay rendering.
//!
//! Each visual model selectable through `[overlay].renderer` lives in its own submodule
//! (`ascii`, `modern`, `helix`, ...) and implements [`OverlayModel`]. `swapchain` owns the Vulkan objects
//! shared by all models and delegates drawing to the active one.
//!
//! To add a model: add a variant to `settings::OverlayRenderer`, create `renderer/<name>.rs`
//! implementing [`OverlayModel`], then register it in [`OverlayModels`] and in
//! `OverlayRendererState` (`state.rs`).

pub mod ascii;
mod background;
mod canvas;
mod format;
mod helix;
mod modern;
mod pipeline;
mod state;
mod swapchain;

use crate::layer::config::{AutoMetric, AutoTarget, Hotkey};
use crate::layer::lib::MetricsSnapshot;
use crate::layer::settings::OverlayRenderer;
use ash::vk;

pub use state::{
    OverlayFeedback, OverlayFeedbackEvent, OverlayFeedbackState, OverlayMode, OverlayModeState,
    OverlayRendererState,
};
pub use swapchain::VulkanSwapchainRenderer;

/// Axis-aligned rectangle in framebuffer pixels used to draw overlay shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Swapchain-wide display options handed to every model when it is created.
#[derive(Clone, Copy)]
struct OverlayStyle {
    extent: vk::Extent2D,
    background_opacity: f32,
    fps_color: [u8; 3],
    scale_2x_min_width_px: u32,
}

/// Everything a model needs to draw one overlay frame.
struct OverlayFrame<'a> {
    metrics: &'a MetricsSnapshot,
    mode: OverlayMode,
    target_fps: Option<f32>,
    auto_algorithm_index: Option<usize>,
    auto_algorithm_label: Option<&'a str>,
    hysteresis: Option<f32>,
    auto_target: Option<(AutoMetric, AutoTarget)>,
    hotkeys: [Hotkey; 4],
    manual_step_fps: f32,
    feedback: OverlayFeedback,
}

/// A visual model drawing the overlay inside an already begun render pass.
trait OverlayModel: Sized {
    /// Creates the model's Vulkan resources for a render pass and swapchain extent.
    fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        style: OverlayStyle,
    ) -> Result<Self, vk::Result>;

    /// Records the draw commands for one frame.
    fn record(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame: &OverlayFrame<'_>,
    );

    /// Destroys the model's Vulkan resources.
    fn destroy(&self, device: &ash::Device);
}

/// Every available visual model; the single place where a new model is instantiated and dispatched.
struct OverlayModels {
    ascii: ascii::AsciiModel,
    modern: modern::ModernModel,
    helix: helix::HelixModel,
}

impl OverlayModels {
    /// Creates every model, destroying those already built if a later one fails.
    fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        style: OverlayStyle,
    ) -> Result<Self, vk::Result> {
        let ascii = ascii::AsciiModel::new(device, render_pass, style)?;
        let modern = match modern::ModernModel::new(device, render_pass, style) {
            Ok(modern) => modern,
            Err(error) => {
                ascii.destroy(device);
                return Err(error);
            }
        };
        let helix = match helix::HelixModel::new(device, render_pass, style) {
            Ok(helix) => helix,
            Err(error) => {
                modern.destroy(device);
                ascii.destroy(device);
                return Err(error);
            }
        };
        Ok(Self {
            ascii,
            modern,
            helix,
        })
    }

    /// Records the frame with the model selected by `renderer`.
    fn record(
        &mut self,
        renderer: OverlayRenderer,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame: &OverlayFrame<'_>,
    ) {
        match renderer {
            OverlayRenderer::Ascii => self.ascii.record(device, command_buffer, frame),
            OverlayRenderer::Modern => self.modern.record(device, command_buffer, frame),
            OverlayRenderer::Helix => self.helix.record(device, command_buffer, frame),
        }
    }

    /// Destroys the resources of every model.
    fn destroy(&self, device: &ash::Device) {
        self.ascii.destroy(device);
        self.modern.destroy(device);
        self.helix.destroy(device);
    }
}
