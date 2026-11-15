use ash::vk;

/// Alpha-blended graphics pipeline whose triangles are positioned through push constants.
pub(super) struct OverlayPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl OverlayPipeline {
    /// Builds a pipeline from SPIR-V shaders whose push constants span `push_constant_size` bytes.
    pub fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        vertex_bytes: &[u8],
        fragment_bytes: &[u8],
        push_constant_size: u32,
    ) -> Result<Self, vk::Result> {
        let vertex_code = ash::util::read_spv(&mut std::io::Cursor::new(vertex_bytes))
            .map_err(|_| vk::Result::ERROR_UNKNOWN)?;
        let fragment_code = ash::util::read_spv(&mut std::io::Cursor::new(fragment_bytes))
            .map_err(|_| vk::Result::ERROR_UNKNOWN)?;
        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(push_constant_size)];
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constants);
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let vertex_info = vk::ShaderModuleCreateInfo::default().code(&vertex_code);
        let vertex_module = match unsafe { device.create_shader_module(&vertex_info, None) } {
            Ok(module) => module,
            Err(error) => {
                unsafe { device.destroy_pipeline_layout(layout, None) };
                return Err(error);
            }
        };
        let fragment_info = vk::ShaderModuleCreateInfo::default().code(&fragment_code);
        let fragment_module = match unsafe { device.create_shader_module(&fragment_info, None) } {
            Ok(module) => module,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_pipeline_layout(layout, None);
                }
                return Err(error);
            }
        };
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module)
                .name(entry),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D::default().extent(extent)];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewport)
            .scissors(&scissor);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
        let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0)];
        let pipeline = match unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        } {
            Ok(mut pipelines) => pipelines.remove(0),
            Err((pipelines, error)) => {
                unsafe {
                    for pipeline in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_shader_module(fragment_module, None);
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_pipeline_layout(layout, None);
                }
                return Err(error);
            }
        };
        unsafe {
            device.destroy_shader_module(fragment_module, None);
            device.destroy_shader_module(vertex_module, None);
        }
        Ok(Self { layout, pipeline })
    }

    /// Destroys the pipeline and its layout.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
