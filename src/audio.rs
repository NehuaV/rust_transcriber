//! Decoding any audio or video file to 16 kHz mono samples.
//!
//! FFmpeg is statically linked into the binary (`ffmpeg-sys-next`), so no system
//! `ffmpeg` is needed at runtime. The whole recording is decoded into memory.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ffmpeg::channel_layout::ChannelLayout;
use ffmpeg::format::sample::Type::Packed;
use ffmpeg::format::Sample;
use ffmpeg::frame::Audio as Frame;
use ffmpeg::software::resampling::Context as Resampler;
use ffmpeg::util::log::Level;
use ffmpeg_next as ffmpeg;

/// Sample rate the model expects, in hertz.
pub const SAMPLE_RATE: u32 = 16_000;

/// A decoded recording and facts about its source file.
#[derive(Debug)]
pub struct DecodedAudio {
    /// Mono samples at [`SAMPLE_RATE`], in the range -1 to 1.
    pub samples: Vec<f32>,
    /// Duration in seconds, from the container when it knows, otherwise
    /// counted from the decoded samples.
    pub duration_secs: f64,
    /// Name of the audio codec.
    pub codec_name: String,
    /// Size of the source file in bytes.
    pub size_bytes: u64,
}

/// Decodes the best audio stream of `path` (M4A, MP3, WAV, FLAC, OGG, OPUS,
/// AAC, MP4 and anything else FFmpeg reads) to 16 kHz mono `f32` samples.
///
/// Packets or frames FFmpeg cannot decode are skipped, so a file with a damaged
/// stretch still yields the rest.
pub fn decode(path: &Path) -> Result<DecodedAudio> {
    let size_bytes = fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);

    ffmpeg::init().context("Failed to initialize statically linked FFmpeg")?;
    ffmpeg::util::log::set_level(Level::Error);

    let mut input = ffmpeg::format::input(&path)
        .with_context(|| format!("Failed to open media file: {}", path.display()))?;
    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .context("Could not find audio stream in file")?;
    let stream_index = stream.index();
    let stream_duration = stream.duration();
    let time_base = f64::from(stream.time_base());

    let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .context("Failed to get codec context from stream parameters")?
        .decoder()
        .audio()
        .context("Failed to instantiate FFmpeg audio decoder")?;
    let codec_name = decoder
        .codec()
        .map_or_else(|| "Unknown".to_owned(), |codec| codec.name().to_owned());

    let input_layout = if decoder.channel_layout().is_empty() {
        ChannelLayout::default(i32::from(decoder.channels()))
    } else {
        decoder.channel_layout()
    };
    let mut resampler = Resampler::get(
        decoder.format(),
        input_layout,
        decoder.rate(),
        Sample::F32(Packed),
        ChannelLayout::MONO,
        SAMPLE_RATE,
    )
    .context("Failed to create FFmpeg audio resampler")?;

    let container_duration = if stream_duration > 0 {
        stream_duration as f64 * time_base
    } else if input.duration() > 0 {
        input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
    } else {
        0.0
    };

    let mut samples = Vec::new();
    for (stream, packet) in input.packets() {
        if stream.index() == stream_index && decoder.send_packet(&packet).is_ok() {
            drain(&mut decoder, &mut resampler, input_layout, &mut samples);
        }
    }
    // Flush the decoder, then the samples the resampler still holds.
    let _ = decoder.send_eof();
    drain(&mut decoder, &mut resampler, input_layout, &mut samples);
    flush(&mut resampler, &mut samples);

    if samples.is_empty() {
        bail!("Could not decode any audio samples from file");
    }

    let duration_secs = if container_duration > 0.0 {
        container_duration
    } else {
        samples.len() as f64 / f64::from(SAMPLE_RATE)
    };
    Ok(DecodedAudio {
        samples,
        duration_secs,
        codec_name,
        size_bytes,
    })
}

/// Pulls every frame the decoder has ready, resamples it and appends the
/// samples to `out`.
///
/// Some decoders (PCM in WAV files, for one) return frames with no channel
/// layout even though the stream has one. The resampler was set up with
/// `layout` and refuses frames that disagree ("Input changed"), so such frames
/// are given that layout first.
fn drain(
    decoder: &mut ffmpeg::decoder::Audio,
    resampler: &mut Resampler,
    layout: ChannelLayout,
    out: &mut Vec<f32>,
) {
    let mut decoded = Frame::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        if decoded.channel_layout().is_empty() {
            decoded.set_channel_layout(layout);
        }
        let mut resampled = Frame::empty();
        if resampler.run(&decoded, &mut resampled).is_ok() {
            append(&resampled, out);
        }
    }
}

/// Appends the samples the resampler still buffers once the stream has ended.
fn flush(resampler: &mut Resampler, out: &mut Vec<f32>) {
    let mut tail = Frame::empty();
    if resampler.flush(&mut tail).is_ok() {
        append(&tail, out);
    }
}

/// Appends the mono `f32` samples of a resampled frame to `out`.
fn append(frame: &Frame, out: &mut Vec<f32>) {
    if frame.samples() > 0 {
        out.extend_from_slice(frame.plane::<f32>(0));
    }
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod tests;
