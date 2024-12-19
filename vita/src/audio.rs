use ruffle_core::backend::audio::{
    swf, AudioBackend, AudioMixer, AudioMixerProxy, DecodeError, RegisterError, SoundHandle,
    SoundInstanceHandle, SoundStreamInfo, SoundTransform,
};
use ruffle_core::impl_audio_mixer_backend;
use std::time::Duration;
use vitasdk_sys::*;

#[macro_export]
macro_rules! sce_err {
    ($func:expr, $($arg:expr),*) => {{
        //println!(stringify!($func));
        let err = unsafe { $func($($arg),*) };
        if err < 0 {
            panic!("{} failed with error: 0x{:x}", stringify!($func), err);
        }
        err
    }};
}

#[allow(dead_code)]
pub struct VitaAudioBackend {
    port: i32,
    mixer: AudioMixer,
    sample_rate: u32,
}

impl VitaAudioBackend {
    /// These govern the adaptive buffer size algorithm, all are in number of frames (pairs of samples).
    /// They must all be integer powers of 2 (due to how the algorithm works).
    const INITIAL_BUFFER_SIZE: u32 = 2048; // 46.44 ms at 44.1 kHz

    pub fn new() -> Result<Self, String> {
        const SAMPLE_RATE: u32 = 48000;
        let port = sce_err!(sceAudioOutOpenPort,
            SCE_AUDIO_OUT_PORT_TYPE_MAIN,
            VitaAudioBackend::INITIAL_BUFFER_SIZE as i32 / 2,
            SAMPLE_RATE as i32,
            SCE_AUDIO_OUT_PARAM_FORMAT_S16_STEREO
        );

        let audio = Self {
            port,
            mixer: AudioMixer::new(2, SAMPLE_RATE as u32),
            sample_rate: SAMPLE_RATE,
        };

        let mixer_proxy: AudioMixerProxy = audio.mixer.proxy();
        std::thread::Builder::new()
            .name("audio".into())
            .spawn(move || {
                let mut output_buffer = [0i16; VitaAudioBackend::INITIAL_BUFFER_SIZE as usize];
                loop {
                    mixer_proxy.mix(&mut output_buffer);
                    sce_err!(sceAudioOutOutput, port, output_buffer.as_ptr().cast());
                }
            }).expect("failed to launch audio thread");

        Ok(audio)
    }
}

impl AudioBackend for VitaAudioBackend {
    impl_audio_mixer_backend!(mixer);

    fn play(&mut self) {
    }

    fn pause(&mut self) {
    }

    fn position_resolution(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            f64::from(VitaAudioBackend::INITIAL_BUFFER_SIZE) / f64::from(self.sample_rate),
        ))
    }
}

impl Drop for VitaAudioBackend {
    fn drop(&mut self) {
        sce_err!(sceAudioOutReleasePort, self.port);
    }
}

