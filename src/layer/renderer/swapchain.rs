use super::{
    OverlayFeedback, OverlayFrame, OverlayMode, OverlayModels, OverlayRendererState, OverlayStyle,
};
use crate::layer::config::{AutoMetric, AutoTarget, Hotkey};
use crate::layer::lib::{GpuMeasurement, HardwareMetrics, MetricsSnapshot};
use ash::vk;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Per-swapchain-image Vulkan objects used to draw the overlay.
struct ImageResources {
    image: vk::Image,
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    signal: vk::Semaphore,
}

/// Owns the Vulkan objects shared by all overlay models and draws the active one onto swapchain images.
pub struct VulkanSwapchainRenderer {
    extent: vk::Extent2D,
    renderer: Arc<OverlayRendererState>,
    overlay_refresh_rate: u64,
    overlay_poll_count: u64,
    displayed_metrics: Option<MetricsSnapshot>,
    render_pass: vk::RenderPass,
    models: OverlayModels,
    images: Vec<ImageResources>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    queue_family: Option<u32>,
    metrics_history: PresentationMetricsHistory,
}

/// Presentation timing state used to derive FPS metrics per limiter generation.
struct PresentationMetricsHistory {
    generation: u64,
    generation_started: Option<Instant>,
    last_present: Option<Instant>,
    fps_smoothing_window: Duration,
    minimum_collection_interval: Duration,
    fps_samples: VecDeque<(Instant, f32)>,
}

impl Default for PresentationMetricsHistory {
    /// Creates metric histories with their built-in smoothing windows.
    fn default() -> Self {
        Self {
            generation: 0,
            generation_started: None,
            last_present: None,
            fps_smoothing_window: Duration::from_secs(1),
            minimum_collection_interval: Duration::ZERO,
            fps_samples: VecDeque::new(),
        }
    }
}

impl PresentationMetricsHistory {
    /// Resets cap-dependent presentation samples.
    fn reset(&mut self, generation: u64, now: Instant) {
        self.generation = generation;
        self.generation_started = Some(now);
        self.last_present = Some(now);
        self.fps_samples.clear();
    }

    /// Records a presentation and calculates current FPS, jitter, and low-10 FPS.
    fn record(&mut self, now: Instant, generation: u64) -> (f32, f32, f32, f32) {
        if self.generation != generation || self.generation_started.is_none() {
            self.reset(generation, now);
        }

        let frametime_ms = self
            .last_present
            .replace(now)
            .map(|last| now.saturating_duration_since(last).as_secs_f32() * 1000.0)
            .unwrap_or(0.0);
        if frametime_ms.is_finite() && frametime_ms > 0.0 {
            let fps = 1000.0 / frametime_ms;
            if fps.is_finite() && fps > 0.0 {
                self.fps_samples.push_back((now, fps));
            }
        }
        while self.fps_samples.front().is_some_and(|(timestamp, _)| {
            now.saturating_duration_since(*timestamp) > self.fps_smoothing_window
        }) {
            self.fps_samples.pop_front();
        }
        let collection_ready = !self.fps_samples.is_empty()
            && self.generation_started.is_some_and(|started| {
                now.saturating_duration_since(started) >= self.minimum_collection_interval
            });
        let fps = if !collection_ready {
            -1.0
        } else {
            self.fps_samples.iter().map(|(_, fps)| fps).sum::<f32>() / self.fps_samples.len() as f32
        };
        let jitter_ms = if collection_ready {
            calculate_jitter_from_fps_samples(&self.fps_samples)
        } else {
            -1.0
        };
        let low_10_fps = if collection_ready {
            calculate_low_10_fps(&self.fps_samples)
        } else {
            -1.0
        };
        (frametime_ms, fps, jitter_ms, low_10_fps)
    }
}

/// Computes frametime jitter from the same time-windowed FPS samples as the overlay metrics.
fn calculate_jitter_from_fps_samples(samples: &VecDeque<(Instant, f32)>) -> f32 {
    if samples.is_empty() {
        return -1.0;
    }
    let mean = samples.iter().map(|(_, fps)| 1000.0 / fps).sum::<f32>() / samples.len() as f32;
    (samples
        .iter()
        .map(|(_, fps)| {
            let frametime_ms = 1000.0 / fps;
            (frametime_ms - mean).powi(2)
        })
        .sum::<f32>()
        / samples.len() as f32)
        .sqrt()
}

/// Computes the tenth-percentile FPS from time-windowed samples.
fn calculate_low_10_fps(samples: &VecDeque<(Instant, f32)>) -> f32 {
    if samples.is_empty() {
        return -1.0;
    }
    let mut fps_values = samples.iter().map(|(_, fps)| *fps).collect::<Vec<_>>();
    fps_values.sort_by(|a, b| a.total_cmp(b));
    let index = ((fps_values.len() as f64 * 0.10).ceil() as usize)
        .max(1)
        .min(fps_values.len())
        .saturating_sub(1);
    fps_values[index]
}

impl VulkanSwapchainRenderer {
    /// Returns the metrics snapshot currently shown by the overlay, if any.
    pub(crate) fn displayed_metrics(&self) -> Option<MetricsSnapshot> {
        self.displayed_metrics.clone()
    }

    /// Feeds a presentation timestamp into the frame-time graph.
    pub fn record_graph_present(&mut self, now: Instant) {
        self.models.ascii.record_graph_present(now);
    }

    /// Clears the frame-time graph history.
    pub fn reset_graph(&mut self) {
        self.models.ascii.reset_graph();
    }

    /// Creates render resources for the images owned by a Vulkan swapchain.
    pub fn new(
        device: &ash::Device,
        format: vk::Format,
        extent: vk::Extent2D,
        swapchain_images: &[vk::Image],
        fps_smoothing_window: Duration,
        minimum_collection_interval: Duration,
        overlay_refresh_rate: u64,
        background_opacity: f32,
        renderer: Arc<OverlayRendererState>,
        fps_color: [u8; 3],
        scale_2x_min_width_px: u32,
    ) -> Result<Self, vk::Result> {
        let attachment = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_reference = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_reference)];
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(&subpass);
        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };
        let style = OverlayStyle {
            extent,
            background_opacity,
            fps_color,
            scale_2x_min_width_px,
        };
        let models = match OverlayModels::new(device, render_pass, style) {
            Ok(models) => models,
            Err(error) => {
                unsafe { device.destroy_render_pass(render_pass, None) };
                return Err(error);
            }
        };
        let mut images = Vec::with_capacity(swapchain_images.len());

        for &image in swapchain_images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            let view = match unsafe { device.create_image_view(&view_info, None) } {
                Ok(view) => view,
                Err(error) => {
                    Self::destroy_partial(device, render_pass, &models, &images);
                    return Err(error);
                }
            };
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(std::slice::from_ref(&view))
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            let framebuffer = match unsafe { device.create_framebuffer(&framebuffer_info, None) } {
                Ok(framebuffer) => framebuffer,
                Err(error) => {
                    unsafe { device.destroy_image_view(view, None) };
                    Self::destroy_partial(device, render_pass, &models, &images);
                    return Err(error);
                }
            };
            let signal =
                match unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                {
                    Ok(signal) => signal,
                    Err(error) => {
                        unsafe {
                            device.destroy_framebuffer(framebuffer, None);
                            device.destroy_image_view(view, None);
                        }
                        Self::destroy_partial(device, render_pass, &models, &images);
                        return Err(error);
                    }
                };
            images.push(ImageResources {
                image,
                view,
                framebuffer,
                signal,
            });
        }

        Ok(Self {
            extent,
            renderer,
            overlay_refresh_rate: overlay_refresh_rate.max(1),
            overlay_poll_count: 0,
            displayed_metrics: None,
            render_pass,
            models,
            images,
            command_pool: vk::CommandPool::null(),
            command_buffers: Vec::new(),
            queue_family: None,
            metrics_history: PresentationMetricsHistory {
                fps_smoothing_window,
                minimum_collection_interval,
                ..PresentationMetricsHistory::default()
            },
        })
    }

    /// Releases resources created before a swapchain renderer initialization failed.
    fn destroy_partial(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        models: &OverlayModels,
        images: &[ImageResources],
    ) {
        models.destroy(device);
        unsafe {
            for image in images {
                device.destroy_semaphore(image.signal, None);
                device.destroy_framebuffer(image.framebuffer, None);
                device.destroy_image_view(image.view, None);
            }
            device.destroy_render_pass(render_pass, None);
        }
    }

    /// Destroys all Vulkan objects owned by this swapchain renderer.
    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if self.command_pool != vk::CommandPool::null() {
                device.destroy_command_pool(self.command_pool, None);
            }
        }
        self.models.destroy(device);
        unsafe {
            for image in &self.images {
                device.destroy_semaphore(image.signal, None);
                device.destroy_framebuffer(image.framebuffer, None);
                device.destroy_image_view(image.view, None);
            }
            device.destroy_render_pass(self.render_pass, None);
        }
    }

    /// Updates per-present history and returns current and refresh-cadenced metrics.
    pub fn record_present(
        &mut self,
        now: Instant,
        gpu_measurement: GpuMeasurement,
        hardware_metrics: HardwareMetrics,
        hardware_timestamp: Instant,
        cap_generation: u64,
    ) -> (MetricsSnapshot, MetricsSnapshot, bool) {
        let refresh_overlay =
            overlay_refresh_due(self.overlay_poll_count, self.overlay_refresh_rate);
        self.overlay_poll_count = self.overlay_poll_count.wrapping_add(1);
        let (frametime_ms, fps, jitter_ms, low_10_fps) =
            self.metrics_history.record(now, cap_generation);
        let current_metrics = MetricsSnapshot::new_with_hardware_metrics_at(
            frametime_ms,
            fps,
            jitter_ms,
            low_10_fps,
            gpu_measurement,
            hardware_metrics,
            hardware_timestamp,
            now,
        );
        if refresh_overlay {
            self.displayed_metrics = Some(current_metrics.clone());
        }
        let displayed_metrics = self
            .displayed_metrics
            .as_ref()
            .unwrap_or(&current_metrics)
            .clone();
        (current_metrics, displayed_metrics, refresh_overlay)
    }

    /// Records commands that draw the overlay into an acquired swapchain image.
    pub fn record_commands(
        &mut self,
        device: &ash::Device,
        queue_family: u32,
        image_index: u32,
        metrics: &MetricsSnapshot,
        overlay_mode: OverlayMode,
        target_fps: Option<f32>,
        auto_algorithm_index: Option<usize>,
        auto_algorithm_label: Option<&str>,
        hysteresis: Option<f32>,
        auto_target: Option<(AutoMetric, AutoTarget)>,
        hotkeys: [Hotkey; 4],
        manual_step_fps: f32,
        feedback: OverlayFeedback,
    ) -> Result<(vk::CommandBuffer, vk::Semaphore), vk::Result> {
        if image_index as usize >= self.images.len() {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        if self.queue_family != Some(queue_family) {
            if self.command_pool != vk::CommandPool::null() {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            self.command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(self.images.len() as u32);
            self.command_buffers = match unsafe { device.allocate_command_buffers(&allocate_info) }
            {
                Ok(buffers) => buffers,
                Err(error) => {
                    unsafe { device.destroy_command_pool(self.command_pool, None) };
                    self.command_pool = vk::CommandPool::null();
                    return Err(error);
                }
            };
            self.queue_family = Some(queue_family);
        }

        let index = image_index as usize;
        let command_buffer = self.command_buffers[index];
        unsafe {
            device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

        let image = self.images[index].image;
        let to_color = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )];
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_color,
            );
        }

        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.extent);
        let render_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.images[index].framebuffer)
            .render_area(render_area);
        unsafe {
            device.cmd_begin_render_pass(command_buffer, &render_begin, vk::SubpassContents::INLINE)
        };

        let frame = OverlayFrame {
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
        };
        self.models
            .record(self.renderer.current(), device, command_buffer, &frame);
        unsafe { device.cmd_end_render_pass(command_buffer) };

        let to_present = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )];
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_present,
            );
            device.end_command_buffer(command_buffer)?;
        }
        Ok((command_buffer, self.images[index].signal))
    }

    /// Returns the signal semaphore associated with an acquired swapchain image.
    pub fn signal_for_image(&self, image_index: u32) -> Option<vk::Semaphore> {
        self.images
            .get(image_index as usize)
            .map(|image| image.signal)
    }
}

/// Determines whether displayed overlay metrics should refresh on this poll.
fn overlay_refresh_due(poll_count: u64, refresh_rate: u64) -> bool {
    poll_count % refresh_rate.max(1) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Checks that metrics refresh immediately and then at the configured cadence.
    fn overlay_metrics_refresh_on_first_and_each_configured_poll() {
        let refreshed = (0..21)
            .filter(|poll| overlay_refresh_due(*poll, 10))
            .collect::<Vec<_>>();
        assert_eq!(refreshed, [0, 10, 20]);
        assert!(overlay_refresh_due(0, 0));
    }

    #[test]
    /// Checks that cap changes restart presentation metric collection.
    fn presentation_metrics_restart_collection_after_cap_changes() {
        let start = Instant::now();
        let mut history = PresentationMetricsHistory {
            minimum_collection_interval: Duration::from_millis(100),
            ..PresentationMetricsHistory::default()
        };

        for sample in 0..=6 {
            let metrics = history.record(start + std::time::Duration::from_millis(sample * 16), 0);
            assert_eq!(metrics.1, -1.0);
            assert_eq!(metrics.2, -1.0);
            assert_eq!(metrics.3, -1.0);
        }
        let metrics = history.record(start + Duration::from_millis(112), 0);
        assert!(metrics.1 >= 0.0);
        assert!(metrics.2 >= 0.0);
        assert!(metrics.3 > 0.0);

        let reset_time = start + Duration::from_millis(128);
        let metrics = history.record(reset_time, 1);
        assert_eq!(metrics, (0.0, -1.0, -1.0, -1.0));

        for sample in 1..=6 {
            let metrics = history.record(
                reset_time + std::time::Duration::from_millis(sample * 16),
                1,
            );
            assert_eq!(metrics.1, -1.0);
            assert_eq!(metrics.2, -1.0);
            assert_eq!(metrics.3, -1.0);
        }
        let metrics = history.record(reset_time + Duration::from_millis(112), 1);
        assert!(metrics.1 > 0.0);
        assert!(metrics.2 >= 0.0);
        assert!(metrics.3 > 0.0);
    }

    #[test]
    /// Checks average FPS over the configured elapsed-time window.
    fn displayed_fps_are_averaged_over_the_configured_time_window() {
        let start = Instant::now();
        let mut history = PresentationMetricsHistory {
            fps_smoothing_window: Duration::from_secs(1),
            ..PresentationMetricsHistory::default()
        };

        assert_eq!(history.record(start, 0).1, -1.0);
        let fps = history.record(start + Duration::from_millis(10), 0).1;
        assert!((fps - 100.0).abs() < 0.1);
        let fps = history.record(start + Duration::from_millis(30), 0).1;
        assert!((fps - 75.0).abs() < 0.1);

        let fps = history.record(start + Duration::from_millis(1040), 0).1;
        assert!((fps - (1000.0 / 1010.0)).abs() < 0.01);
    }

    #[test]
    /// Checks that low-10 FPS uses the configured elapsed-time window.
    fn displayed_10_percent_low_uses_the_configured_time_window() {
        let start = Instant::now();
        let mut history = PresentationMetricsHistory {
            fps_smoothing_window: Duration::from_millis(100),
            ..PresentationMetricsHistory::default()
        };

        assert_eq!(history.record(start, 0).3, -1.0);
        let low_10 = history.record(start + Duration::from_millis(10), 0).3;
        assert!((low_10 - 100.0).abs() < 0.1);
        let low_10 = history.record(start + Duration::from_millis(30), 0).3;
        assert!((low_10 - 50.0).abs() < 0.1);

        let low_10 = history.record(start + Duration::from_millis(150), 0).3;
        assert!((low_10 - (1000.0 / 120.0)).abs() < 0.1);

        let low_10 = history.record(start + Duration::from_millis(251), 0).3;
        assert!((low_10 - (1000.0 / 101.0)).abs() < 0.1);
    }

    #[test]
    /// Checks that FPS, 10% low, and jitter share samples after the minimum collection interval.
    fn presentation_metrics_share_samples_after_the_minimum_collection_interval() {
        let start = Instant::now();
        let mut history = PresentationMetricsHistory {
            fps_smoothing_window: Duration::from_millis(100),
            minimum_collection_interval: Duration::from_millis(30),
            ..PresentationMetricsHistory::default()
        };

        assert_eq!(history.record(start, 0), (0.0, -1.0, -1.0, -1.0));
        let metrics = history.record(start + Duration::from_millis(10), 0);
        assert_eq!(metrics.1, -1.0);
        assert_eq!(metrics.2, -1.0);
        assert_eq!(metrics.3, -1.0);

        let metrics = history.record(start + Duration::from_millis(30), 0);
        assert!((metrics.1 - 75.0).abs() < 0.1);
        assert!((metrics.2 - 5.0).abs() < 0.1);
        assert!((metrics.3 - 50.0).abs() < 0.1);

        let metrics = history.record(start + Duration::from_millis(150), 0);
        assert!((metrics.1 - (1000.0 / 120.0)).abs() < 0.1);
        assert_eq!(metrics.2, 0.0);
        assert!((metrics.3 - (1000.0 / 120.0)).abs() < 0.1);
    }
}
