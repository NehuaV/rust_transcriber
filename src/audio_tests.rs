use std::env;
use std::path::PathBuf;

use super::*;

/// A temporary file that is removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str) -> Self {
        Self(env::temp_dir().join(format!("transcriber-{}-{name}", std::process::id())))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Writes a 16-bit PCM WAV file holding a sine tone.
fn write_wav(path: &Path, rate: u32, channels: u16, seconds: f32) {
    let frames = (rate as f32 * seconds) as usize;
    let mut data = Vec::with_capacity(frames * usize::from(channels) * 2);
    for frame in 0..frames {
        let value = (8000.0 * (frame as f32 * 0.05).sin()) as i16;
        for _ in 0..channels {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }

    let block_align = channels * 2;
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);
    fs::write(path, wav).unwrap();
}

#[test]
fn a_16khz_mono_wav_decodes_to_the_same_samples() {
    // WAV frames carry no channel layout; this once failed with "Input changed".
    let file = TempFile::new("mono16k.wav");
    write_wav(&file.0, 16_000, 1, 2.0);

    let audio = decode(&file.0).unwrap();
    assert_eq!(audio.samples.len(), 32_000);
    assert!((audio.duration_secs - 2.0).abs() < 0.01);
    assert_eq!(audio.codec_name, "pcm_s16le");
    assert!(audio.size_bytes > 64_000);
    assert!(
        audio.samples.iter().any(|&s| s.abs() > 0.1),
        "the tone is audible"
    );
    assert!(audio.samples.iter().all(|&s| s.abs() <= 1.0));
}

#[test]
fn a_stereo_44khz_wav_is_resampled_to_16khz_mono() {
    let file = TempFile::new("stereo44k.wav");
    write_wav(&file.0, 44_100, 2, 3.0);

    let audio = decode(&file.0).unwrap();
    let expected = 3.0 * SAMPLE_RATE as f32;
    assert!(
        (audio.samples.len() as f32 - expected).abs() < 0.01 * expected,
        "got {} samples, expected about {expected}",
        audio.samples.len()
    );
}

#[test]
fn a_missing_file_is_an_error() {
    assert!(decode(Path::new("/does/not/exist.mp3")).is_err());
}

#[test]
fn a_file_without_audio_is_an_error() {
    let file = TempFile::new("not-audio.txt");
    fs::write(&file.0, "this is not audio").unwrap();
    assert!(decode(&file.0).is_err());
}
