use anyhow::{anyhow, Context, Result};
use ash::vk;
use std::ffi::CString;
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, RawDisplayHandle},
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

#[derive(Default, Debug, PartialEq, Eq)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
}
impl QueueFamilyIndices {
    fn is_completed(&self) -> bool {
        self.graphics_family.is_some()
    }
}

struct VulkanApp {
    _entry: ash::Entry,
    instance: ash::Instance,
    _window: winit::window::Window,
    _physical_device: vk::PhysicalDevice,
    logical_device: ash::Device,
    graphics_queue: vk::Queue,
}

impl VulkanApp {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }?;
        let window = Self::init_window(event_loop)?;
        window.set_resizable(false);
        let instance = Self::create_instance(&entry, window.display_handle()?.as_raw())?;
        let physical_device = Self::pick_physical_device(&instance)?;
        let (logical_device, graphics_queue) =
            Self::create_logical_device(&instance, physical_device)?;
        Ok(VulkanApp {
            _entry: entry,
            instance,
            _window: window,
            _physical_device: physical_device,
            graphics_queue,
            logical_device,
        })
    }
    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(ash::Device, vk::Queue)> {
        let indices = Self::find_queue_families(instance, physical_device);

        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(indices.graphics_family.context(format!(
                "The graphics family of {:?} is None",
                &physical_device
            ))?)
            .queue_priorities(&queue_priorities);

        let device_features = unsafe { instance.get_physical_device_features(physical_device) };
        let binding = [queue_create_info];
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&binding)
            .enabled_features(&device_features);
        let device = unsafe { instance.create_device(physical_device, &create_info, None) }?;
        let queue = unsafe { device.get_device_queue(indices.graphics_family.unwrap(), 0) };
        Ok((device, queue))
    }
    fn find_queue_families(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> QueueFamilyIndices {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(device) };
        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
        };
        for (i, queue_family) in queue_family_properties.iter().enumerate() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(i as u32);
            }

            if queue_family_indices.is_completed() {
                break;
            }
        }

        queue_family_indices
    }
    fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
        let physical_devices: Vec<_> = unsafe { instance.enumerate_physical_devices() }?;
        let devices: Vec<_> = physical_devices
            .iter()
            .filter(|device| Self::is_device_suitable(instance, **device))
            .collect();

        match devices.len() {
            0 => Err(anyhow!(
                "There are {} physical devices but None of them are suitable for our use case",
                physical_devices.len()
            )),
            _ => Ok(*devices[0]),
        }
    }

    fn is_device_suitable(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
        Self::find_queue_families(instance, device).is_completed()
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
            .enabled_extension_names(&extension_names);
        Ok(unsafe { entry.create_instance(&create_info, None) }?)
    }
    fn init_window(event_loop: &EventLoop<()>) -> Result<winit::window::Window> {
        Ok(winit::window::WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title("Vulkan")
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
