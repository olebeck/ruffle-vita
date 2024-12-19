use ruffle_core::backend::navigator::NullNavigatorBackend;
use ruffle_core::backend::storage::MemoryStorageBackend;
use ruffle_core::config::Letterbox;
use ruffle_core::external::NullFsCommandProvider;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::LoadBehavior;
use ruffle_core::PlayerBuilder;
use ruffle_render::quality::StageQuality;
use ruffle_render_gles2::GLES2RenderBackend;
use std::ffi::CString;
use std::time::Duration;


fn main() {
    let sdl = beryllium::Sdl::init(beryllium::init::InitFlags::VIDEO);
    sdl.set_gl_profile(beryllium::video::GlProfile::Core).unwrap();
    sdl.set_gl_context_major_version(3).unwrap();
    sdl.set_gl_context_major_version(1).unwrap();

    let win_args = beryllium::video::CreateWinArgs {
        title: "Ruffle",
        width: 960,
        height: 544,
        allow_high_dpi: true,
        borderless: false,
        resizable: false,
    };

    let width = win_args.width;
    let height = win_args.height;

    let _win = sdl
        .create_gl_window(win_args)
        .expect("couldn't make a window and context");
    _win.set_swap_interval(beryllium::video::GlSwapInterval::Vsync).unwrap();
    
    let _gl = gl::load_with(|s| unsafe {
        _win.get_proc_address(CString::new(s).unwrap().as_ptr().cast())
    });

    let swf_path = "test.swf";
    let data = std::fs::read(swf_path).unwrap();
    let movie = SwfMovie::from_data(&data, "".into(), None).unwrap();
    
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
    'main_loop: loop {
        while let Some(event) = sdl.poll_events() {
            match event {
                (beryllium::events::Event::Quit, _) => break 'main_loop,
                _ => (),
            }
        }
        player_locked.run_frame();
        player_locked.render();
        _win.swap_window();
    }
}