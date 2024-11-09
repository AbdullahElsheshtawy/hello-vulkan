use anyhow::{anyhow, Context, Result};
use ash::vk::{self};
use std::ffi::CString;
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle},
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

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

struct VulkanApp {
    _entry: ash::Entry,
    _graphics_queue: vk::Queue,
    _physical_device: vk::PhysicalDevice,
    _window: winit::window::Window,
    instance: ash::Instance,
    logical_device: ash::Device,
    present_queue: vk::Queue,
    surface: Surface,
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
        let (logical_device, family_indices) =
            Self::create_logical_device(&instance, physical_device, &surface)?;

        let graphics_queue =
            unsafe { logical_device.get_device_queue(family_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { logical_device.get_device_queue(family_indices.present_family.unwrap(), 0) };
        Ok(VulkanApp {
            _entry: entry,
            _graphics_queue: graphics_queue,
            _physical_device: physical_device,
            surface,
            present_queue,
            _window: window,
            instance,
            logical_device,
        })
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
        let device_features = unsafe { instance.get_physical_device_features(physical_device) };
        let binding = [queue_create_info];
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&binding)
            .enabled_features(&device_features);
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
        Self::find_queue_families(instance, device, surface)
            .unwrap()
            .is_completed()
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
            self.surface
                .instance
                .destroy_surface(self.surface.surface, None);
            self.logical_device.destroy_device(None);
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
