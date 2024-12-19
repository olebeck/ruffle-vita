use ruffle_core::backend::navigator::NullNavigatorBackend;
use ruffle_core::backend::storage::MemoryStorageBackend;
use ruffle_core::config::Letterbox;
use ruffle_core::external::NullFsCommandProvider;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::LoadBehavior;
use ruffle_core::PlayerBuilder;
use ruffle_render::quality::StageQuality;
use ruffle_render_gles2::GLES2RenderBackend;
use vitasdk_sys::{_sceKernelLoadModule, sceKernelStartModule};
use std::ffi::CString;
use std::time::Duration;

mod egl;
mod pvr;

#[link(name = "SceLibKernel_stub", kind = "static")]
extern "C" {}

#[used]
#[no_mangle]
#[export_name = "sceLibcHeapSize"]
pub static SCE_LIBC_HEAP_SIZE: u32 = 50 * 1024 * 1024; // 10 MiB

fn sce_kernel_load_start_module(name: &str) {
    let c_name = CString::new(name).expect("CString::new failed");
    let mod_id = unsafe {
        _sceKernelLoadModule(c_name.as_ptr(), 0, std::ptr::null())
    };
    if mod_id < 0 {
        panic!("_sceKernelLoadModule {name} 0x{mod_id:x}")
    }
    let ret = unsafe {
        let mut res: i32 = 0;
        sceKernelStartModule(mod_id, 0, std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut res as *mut i32)
    };
    if ret < 0 {
        panic!("sceKernelStartModule {name} 0x{ret:x}")
    }
    println!("sce_kernel_load_start_module {name} success!");
}


fn egl_init() -> (egl::EGLDisplay, egl::EGLSurface) {
    sce_kernel_load_start_module("app0:sce_module/libfios2.suprx");
    sce_kernel_load_start_module("app0:sce_module/libc.suprx");
    sce_kernel_load_start_module("app0:module/libgpu_es4_ext.suprx");
    sce_kernel_load_start_module("app0:module/libIMGEGL.suprx");

    unsafe {
        let mut hint: pvr::PVRSRV_PSP2_APPHINT = std::mem::zeroed();
        pvr::PVRSRVInitializeAppHint(&mut hint);
        hint.ui32DriverMemorySize = 0x1000000;
        pvr::PVRSRVCreateVirtualAppHint(&mut hint);

        let dpy = egl::eglGetDisplay(std::ptr::null_mut());

        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        if !egl::eglInitialize(dpy, &mut major, &mut minor) {
            panic!("eglInitialize {:x}", egl::eglGetError());
        }

        let mut num_config: i32 = 0;
        let mut configs: [egl::EGLConfig; 2] = std::mem::zeroed();

        if !egl::eglGetConfigs(dpy, configs.as_mut_ptr(), configs.len() as i32, &mut num_config) {
            panic!("eglGetConfigs {:x}", egl::eglGetError());
        }

        let cfg_attribs: [u32; 13] = [
            egl::EGL_BUFFER_SIZE,       egl::EGL_DONT_CARE,
            egl::EGL_DEPTH_SIZE,        16,
            egl::EGL_RED_SIZE,          8,
            egl::EGL_GREEN_SIZE,        8,
            egl::EGL_BLUE_SIZE,         8,
            egl::EGL_RENDERABLE_TYPE,   egl::EGL_OPENGL_ES2_BIT,
            egl::EGL_NONE
        ];

        if !egl::eglChooseConfig(dpy, cfg_attribs.as_ptr().cast(), configs.as_mut_ptr(), 2, &mut num_config) {
            panic!("eglChooseConfig {:x}", egl::eglGetError());
        }

        let win = 0;
        let surface = egl::eglCreateWindowSurface(dpy, configs[0], win, std::ptr::null_mut());

        if !egl::eglBindAPI(egl::EGL_OPENGL_ES_API) {
            panic!("eglCreateWindowSurface {:x}", egl::eglGetError());
        }

        let context_attribs: [u32; 3] = [
            egl::EGL_CONTEXT_CLIENT_VERSION, 2,
            egl::EGL_NONE
        ];
        
        let context = egl::eglCreateContext(
            dpy,
            configs[0],
            std::ptr::null_mut(),
            context_attribs.as_ptr().cast()
        );
        if std::ptr::eq(context, std::ptr::null()) {
            panic!("eglCreateContext {:x}", egl::eglGetError());
        }

        if !egl::eglMakeCurrent(dpy, surface, surface, context) {
            panic!("eglMakeCurrent {:x}", egl::eglGetError());
        }

        return (dpy, surface);
    }
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "full");
    // need to have reference so compiler doesnt remove it
    println!("{:x}", SCE_LIBC_HEAP_SIZE);


    let (dpy, surface) = egl_init();

    let mut width: i32 = 0;
    let mut height: i32 = 0;
    unsafe {
        egl::eglQuerySurface(dpy, surface, egl::EGL_WIDTH as i32, &mut width);
        egl::eglQuerySurface(dpy, surface, egl::EGL_HEIGHT as i32, &mut height);
        egl::eglSwapInterval(dpy, 1); 
    }
    let _gl = gl::load_with(|s| unsafe {
        let ret = egl::eglGetProcAddress(CString::new(s).unwrap().as_ptr());
        if !ret.is_null() {
            println!("{s}: 0x{:x}", ret as u32);
        }
        ret
    } );

    let swf_path = "app0:/test.swf";
    let data = std::fs::read(swf_path).unwrap();
    let movie = SwfMovie::from_data(&data, "app0:".into(), None).unwrap();
    
    let navigator = NullNavigatorBackend::new();
    let renderer = GLES2RenderBackend::new(
        false,
        StageQuality::Best,
        width,
        height,
    )
        .expect("Couldn't create gles2 rendering backend");

    let builder = PlayerBuilder::new();
    let player = builder
        .with_video(ruffle_video_software::backend::SoftwareVideoBackend::new())
        .with_navigator(navigator)
        .with_renderer(renderer)
        .with_storage(Box::new(MemoryStorageBackend::new()))
        .with_fs_commands(Box::new(NullFsCommandProvider{}))
        .with_autoplay(true)
        .with_letterbox(Letterbox::Fullscreen)
        .with_max_execution_duration(Duration::MAX)
        .with_quality(StageQuality::High)
        .with_fullscreen(true)
        .with_load_behavior(LoadBehavior::Streaming)
        .with_frame_rate(Some(30.0))
        .with_avm2_optimizer_enabled(true)
        .with_movie(movie)
        .with_viewport_dimensions(width as u32, height as u32, 1.0)
        .build();

    let mut player_locked = player.lock().unwrap();
    loop {
        player_locked.run_frame();
        player_locked.render();
        unsafe {
            egl::eglSwapBuffers(dpy, surface);
        }
    }
}