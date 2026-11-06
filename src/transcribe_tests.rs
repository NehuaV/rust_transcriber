use super::*;

#[test]
fn encoder_limit_matches_positional_embedding() {
    assert_eq!(encoder_steps(MAX_SAMPLES), MAX_ENCODER_STEPS);
    assert_eq!(
        encoder_steps(MAX_SAMPLES + HOP_LENGTH),
        MAX_ENCODER_STEPS + 1
    );
    assert!(encoder_steps(CHUNK_SAMPLES) < MAX_ENCODER_STEPS);
    assert!(CHUNK_SAMPLES < MAX_SAMPLES);
    assert!(OVERLAP_SAMPLES < CHUNK_SAMPLES);
    assert_eq!(OVERLAP_SAMPLES % 2, 0);
}

#[test]
fn long_audio_chunks_stay_inside_encoder_limit() {
    // 2 * 18062 - 1 = 36123, the attention axis from the reported failure.
    let steps = 18_062usize;
    let mel_frames = (steps - 1) * 8 + 1;
    let samples = mel_frames * HOP_LENGTH;
    assert_eq!(encoder_steps(samples), steps as i64);

    let spans = chunk_spans(samples, CHUNK_SAMPLES, OVERLAP_SAMPLES);
    assert!(spans.len() > 1);
    assert_eq!(spans.first().copied(), Some((0, CHUNK_SAMPLES)));
    assert_eq!(spans.last().map(|span| span.1), Some(samples));

    let mut covered_until = 0.0f32;
    for (i, &(start, end)) in spans.iter().enumerate() {
        assert!(start < end);
        assert!(encoder_steps(end - start) <= MAX_ENCODER_STEPS);
        let is_last = i + 1 == spans.len();
        if is_last {
            assert!(end - start > OVERLAP_SAMPLES);
        } else {
            assert_eq!(end - start, CHUNK_SAMPLES);
            assert_eq!(spans[i + 1].0, start + CHUNK_SAMPLES - OVERLAP_SAMPLES);
        }

        let (from, to) = absolute_keep_window(start, end, i == 0, is_last);
        if i == 0 {
            assert_eq!(from, 0.0);
        } else {
            assert!((from - covered_until).abs() < 1e-3);
        }
        let logical_to = if is_last {
            end as f32 / SAMPLE_RATE as f32
        } else {
            to
        };
        assert!(logical_to > from);
        covered_until = logical_to;
    }

    let duration = samples as f32 / SAMPLE_RATE as f32;
    assert!((covered_until - duration).abs() < 1e-2);
}

#[test]
fn short_audio_is_one_chunk() {
    let spans = chunk_spans(SAMPLE_RATE as usize, CHUNK_SAMPLES, OVERLAP_SAMPLES);
    assert_eq!(spans, vec![(0, SAMPLE_RATE as usize)]);
}

#[test]
fn trailing_punctuation_keeps_the_word_timestamp() {
    let folded = fold_trailing_punctuation(vec![token("Hello", 1.0, 1.4), token(".", 1.4, 1.45)]);
    assert_eq!(folded.len(), 1);
    assert_eq!(folded[0].text, "Hello.");
    assert_eq!((folded[0].start, folded[0].end), (1.0, 1.4));
}

#[test]
fn join_words_attaches_punctuation() {
    let words = [
        token("Hello", 0.0, 0.4),
        token(",", 0.4, 0.4),
        token("world", 0.5, 0.9),
        token(".", 0.9, 0.9),
    ];
    assert_eq!(join_words(&words), "Hello, world.");
}

fn token(text: &str, start: f32, end: f32) -> TimedToken {
    TimedToken {
        text: text.to_string(),
        start,
        end,
    }
}
