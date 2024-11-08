use anyhow::Result;
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
const VALIDATION: [&str; 1] = ["VK_LAYER_KHRONOS_VALIDATION"];

#[cfg(debug_assertions)]
const ENABLEVALIDATIONLAYERS: bool = true;

#[cfg(not(debug_assertions))]
const ENABLEVALIDATIONLAYERS: bool = false;

struct VulkanApp {
    entry: ash::Entry,
    instance: ash::Instance,
    _window: winit::window::Window,
}

impl VulkanApp {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }?;
        let window = Self::init_window(&event_loop)?;
        window.set_resizable(false);
        let instance = Self::create_instance(&entry, window.display_handle()?.as_raw())?;
        Ok(VulkanApp {
            entry,
            instance,
            _window: window,
        })
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
            winit::event::Event::WindowEvent { ref event, .. } => match event {
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

    fn check_validation_layer_support(&self) -> Result<Vec<vk::LayerProperties>> {
        unsafe {
            Ok(ash::Entry::enumerate_instance_layer_properties(
                &self.entry,
            )?)
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let app = VulkanApp::new(&event_loop)?;
    app.main_loop(event_loop)?;
    Ok(())
}
