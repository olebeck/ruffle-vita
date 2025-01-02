use ruffle_core::backend::navigator::NullNavigatorBackend;
use ruffle_core::backend::storage::MemoryStorageBackend;
use ruffle_core::config::Letterbox;
use ruffle_core::external::NullFsCommandProvider;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::LoadBehavior;
use ruffle_core::PlayerBuilder;
use ruffle_render::quality::StageQuality;
use ruffle_render_gxm::GxmRenderBackend;
use std::time::Duration;

mod audio;
use audio::VitaAudioBackend;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "full");

    let swf_path = "app0:/test.swf";
    let data = std::fs::read(swf_path).unwrap();
    let movie = SwfMovie::from_data(&data, "app0:".into(), None).unwrap();
    
    let navigator = NullNavigatorBackend::new();
    let renderer = GxmRenderBackend::new(
        false,
        StageQuality::Best,
    ).expect("Couldn't create gxm rendering backend");

    let width = renderer.width();
    let height = renderer.height();

    let builder = PlayerBuilder::new();
    let player = builder
        .with_video(ruffle_video_software::backend::SoftwareVideoBackend::new())
        .with_audio(VitaAudioBackend::new().unwrap())
        .with_navigator(navigator)
        .with_renderer(renderer)
        .with_storage(Box::new(MemoryStorageBackend::new()))
        .with_fs_commands(Box::new(NullFsCommandProvider{}))
        .with_autoplay(true)
        .with_letterbox(Letterbox::On)
        .with_max_execution_duration(Duration::MAX)
        .with_quality(StageQuality::High)
        .with_fullscreen(true)
        .with_load_behavior(LoadBehavior::Streaming)
        .with_frame_rate(Some(30.0))
        .with_avm2_optimizer_enabled(true)
        .with_movie(movie)
        .with_viewport_dimensions(width, height, 1.0)
        .build();

    let mut player_locked = player.lock().unwrap();
    let mut last_time = std::time::Instant::now();
    loop {
        let new_time = std::time::Instant::now();
        let dt = new_time.duration_since(last_time).as_micros() as f64 / 1000.;
        if dt > 0. {
            last_time = new_time;
            player_locked.tick(dt);
            player_locked.render();

            let renderer: &mut GxmRenderBackend = player_locked.renderer_mut().downcast_mut().unwrap();
            renderer.swap();
            let nf = player_locked.time_til_next_frame();
            std::thread::sleep(nf)
        }
        
    }
}