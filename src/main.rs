use anyhow::{anyhow, Context, Result};
use ash::vk;
use core::str;
use std::{ffi::CString, fmt::Debug};
use vk_shader_macros::include_glsl;
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle},
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const REQUIRED_EXTENSIONS: [&str; 1] = ["VK_KHR_SWAPCHAIN"];

#[derive(Default, Debug, PartialEq, Eq)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_completed(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct Surface {
    surface: vk::SurfaceKHR,
    instance: ash::khr::surface::Instance,
}

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl Surface {
    pub fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> Result<Self> {
        let surface_instance = ash::khr::surface::Instance::new(entry, instance);
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }?;

        Ok(Self {
            instance: surface_instance,
            surface,
        })
    }
}

struct SwapChain {
    device: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_format: vk::Format,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
}
struct VulkanApp {
    entry: ash::Entry,
    graphics_queue: vk::Queue,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    present_queue: vk::Queue,
    surface: Surface,
    swapchain: SwapChain,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    window: winit::window::Window,
}

impl VulkanApp {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        // Loading the vulkan dll
        let entry = unsafe { ash::Entry::load() }?;
        let window = Self::init_window(event_loop)?;
        // Creating vulkan instance
        let instance = Self::create_instance(&entry, window.display_handle()?.as_raw())?;
        // The surface is the connection between the OS window and vulkan
        let surface = Surface::new(&entry, &instance, &window)?;
        // getting the actual GPU
        let physical_device = Self::pick_physical_device(&instance, &surface)?;
        // making a logical device and queue to send commands to
        let (device, family_indices) =
            Self::create_logical_device(&instance, physical_device, &surface)?;

        let graphics_queue =
            unsafe { device.get_device_queue(family_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(family_indices.present_family.unwrap(), 0) };

        let swapchain = Self::create_swapchain(&instance, &device, physical_device, &surface)?;

        let render_pass = Self::create_render_pass(&device, &swapchain)?;
        let pipeline_layout = Self::create_graphics_pipeline(&device, &swapchain)?;
        Ok(VulkanApp {
            entry,
            graphics_queue,
            instance,
            device,
            physical_device,
            present_queue,
            surface,
            swapchain,
            pipeline_layout,
            window,
            render_pass,
        })
    }

    fn query_swapchain_support(
        device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> Result<SwapChainSupportDetails> {
        let capabilities = unsafe {
            surface
                .instance
                .get_physical_device_surface_capabilities(device, surface.surface)?
        };

        let formats = unsafe {
            surface
                .instance
                .get_physical_device_surface_formats(device, surface.surface)?
        };

        let present_modes = unsafe {
            surface
                .instance
                .get_physical_device_surface_present_modes(device, surface.surface)?
        };

        Ok(SwapChainSupportDetails {
            capabilities,
            formats,
            present_modes,
        })
    }

    fn create_render_pass(device: &ash::Device, swapchain: &SwapChain) -> Result<vk::RenderPass> {
        let color_attachments = [vk::AttachmentDescription::default()
            .format(swapchain.image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
        let color_attachment_ref = [
            vk::AttachmentReference::default().layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        ];

        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_ref)];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&color_attachments)
            .subpasses(&subpass);
        unsafe { Ok(device.create_render_pass(&render_pass_info, None)?) }
    }
    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain: &SwapChain,
    ) -> Result<vk::PipelineLayout> {
        let vert_shader_module =
            Self::create_shader_module(device, include_glsl!("shaders/triangle.vert"))?;
        let frag_shader_module =
            Self::create_shader_module(device, include_glsl!("shaders/triangle.frag"))?;

        let main_function_name = CString::new("main")?;
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(&main_function_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .name(&main_function_name),
        ];
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&[color_blend_attachment]);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None)
        };

        Ok(pipeline_layout)
    }

    fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        unsafe {
            Ok(device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)?)
        }
    }

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> Result<SwapChain> {
        let swap_chain_support = Self::query_swapchain_support(physical_device, surface)?;
        let surface_format = Self::choose_swap_surface_format(&swap_chain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swap_chain_support.present_modes);
        let extent = Self::choose_swap_extent(&swap_chain_support.capabilities);
        let mut image_count = swap_chain_support.capabilities.min_image_count + 1;
        if swap_chain_support.capabilities.max_image_count > 0
            && image_count > swap_chain_support.capabilities.max_image_count
        {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        let indices = Self::find_queue_families(instance, physical_device, surface)?;
        let queue_family_indices = [
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swap_chain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let create_info = match indices.graphics_family != indices.present_family {
            true => create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices),
            false => create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE),
        };
        let swapchain_device = ash::khr::swapchain::Device::new(instance, device);
        let swapchain = unsafe { swapchain_device.create_swapchain(&create_info, None) }?;
        let swapchain_images = unsafe { swapchain_device.get_swapchain_images(swapchain) }?;
        let image_views =
            Self::create_image_views(device, &swapchain_images, surface_format.format);
        Ok(SwapChain {
            device: swapchain_device,
            swapchain,
            images: swapchain_images,
            image_format: surface_format.format,
            extent,
            image_views,
        })
    }

    fn create_image_views(
        device: &ash::Device,
        images: &Vec<vk::Image>,
        format: vk::Format,
    ) -> Vec<vk::ImageView> {
        images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(
                        vk::ComponentMapping::default()
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY)
                            .a(vk::ComponentSwizzle::IDENTITY),
                    )
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .base_array_layer(0)
                            .level_count(1)
                            .layer_count(1),
                    );
                unsafe { device.create_image_view(&create_info, None).unwrap() }
            })
            .collect()
    }
    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> Result<(ash::Device, QueueFamilyIndices)> {
        let indices = Self::find_queue_families(instance, physical_device, surface)?;
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(indices.graphics_family.context(format!(
                "The graphics family of {:?} is None",
                &physical_device
            ))?)
            .queue_priorities(&queue_priorities);

        // Going with the device features that the device has
        let required_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_features = unsafe { instance.get_physical_device_features(physical_device) };
        let binding = [queue_create_info];
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&binding)
            .enabled_features(&device_features)
            .enabled_extension_names(&required_extensions);
        let device = unsafe { instance.create_device(physical_device, &create_info, None) }?;
        // Going with the first suitable queue family of the device
        Ok((device, indices))
    }
    fn find_queue_families(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> Result<QueueFamilyIndices> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(device) };
        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
        };
        for (i, queue_family) in queue_family_properties.iter().enumerate() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(i as u32);
            }

            if unsafe {
                surface.instance.get_physical_device_surface_support(
                    device,
                    i as u32,
                    surface.surface,
                )?
            } {
                queue_family_indices.present_family = Some(i as u32);
            }
            if queue_family_indices.is_completed() {
                break;
            }
        }

        Ok(queue_family_indices)
    }
    fn pick_physical_device(
        instance: &ash::Instance,
        surface: &Surface,
    ) -> Result<vk::PhysicalDevice> {
        let physical_devices: Vec<_> = unsafe { instance.enumerate_physical_devices() }?;
        let devices: Vec<_> = physical_devices
            .iter()
            .filter(|device| Self::is_device_suitable(instance, **device, surface))
            .collect();
        match devices.len() {
            0 => Err(anyhow!(
                "There are {} physical devices but None of them are suitable for our use case",
                physical_devices.len()
            )),
            _ => Ok(*devices[0]),
        }
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> bool {
        let indices = Self::find_queue_families(instance, device, surface).unwrap();
        let is_extensions_supported = Self::check_device_extension_support(instance, device);
        let swap_chain_adequate = if is_extensions_supported {
            let swap_chain_support_details =
                Self::query_swapchain_support(device, surface).unwrap();
            !swap_chain_support_details.formats.is_empty()
                && !swap_chain_support_details.present_modes.is_empty()
        } else {
            false
        };

        indices.is_completed() && is_extensions_supported && swap_chain_adequate
    }

    fn choose_swap_surface_format(
        available_formats: &Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        for f in available_formats {
            if f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *f;
            }
        }
        available_formats[0]
    }

    fn choose_swap_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        for m in available_present_modes {
            if *m == vk::PresentModeKHR::MAILBOX {
                return *m;
            }
        }
        vk::PresentModeKHR::FIFO
    }

    fn choose_swap_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        vk::Extent2D {
            width: u32::clamp(
                WINDOW_WIDTH,
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: u32::clamp(
                WINDOW_HEIGHT,
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
    fn check_device_extension_support(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> bool {
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .expect("Failed to get device extension properties.")
        };

        let available_extension_names: Vec<_> = available_extensions
            .iter()
            .map(|s| {
                s.extension_name_as_c_str()
                    .unwrap()
                    .to_str()
                    .to_owned()
                    .unwrap()
            })
            .collect();

        let are_required_extensions_present: Vec<_> = REQUIRED_EXTENSIONS
            .iter()
            .filter(|e| available_extension_names.contains(e))
            .collect();
        are_required_extensions_present.is_empty()
    }
    fn create_instance(entry: &ash::Entry, handle: RawDisplayHandle) -> Result<ash::Instance> {
        let app_name = CString::new("Vulkan")?;
        let engine_name = CString::new("No Engine")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_0);
        let extension_names = ash_window::enumerate_required_extensions(handle)?;
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extension_names);
        Ok(unsafe { entry.create_instance(&create_info, None) }?)
    }
    fn init_window(event_loop: &EventLoop<()>) -> Result<winit::window::Window> {
        Ok(winit::window::WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title("Vulkan")
            .with_resizable(false)
            .build(event_loop)?)
    }

    pub fn main_loop(self, event_loop: EventLoop<()>) -> Result<()> {
        Ok(event_loop.run(move |event, control_flow| match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => control_flow.exit(),
                _ => {}
            },
            _ => {}
        })?)
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            for image_view in &self.swapchain.image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.swapchain
                .device
                .destroy_swapchain(self.swapchain.swapchain, None);
            self.surface
                .instance
                .destroy_surface(self.surface.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let app = VulkanApp::new(&event_loop)?;
    app.main_loop(event_loop)?;
    Ok(())
}
