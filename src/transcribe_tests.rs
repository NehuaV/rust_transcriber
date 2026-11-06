use super::*;

#[test]
fn coreml_windows_cover_the_audio_with_overlap() {
    // 2 minutes 7 seconds of audio.
    let samples = 127 * SAMPLE_RATE as usize;
    let spans = chunk_spans(samples, COREML_WINDOW_SAMPLES, COREML_OVERLAP_SAMPLES);
    assert!(spans.len() > 1);
    assert_eq!(spans.first().copied(), Some((0, COREML_WINDOW_SAMPLES)));
    assert_eq!(spans.last().map(|span| span.1), Some(samples));

    let mut covered_until = 0.0f32;
    for (i, &(start, end)) in spans.iter().enumerate() {
        assert!(start < end);
        assert!(end - start <= COREML_WINDOW_SAMPLES);
        let is_last = i + 1 == spans.len();
        if is_last {
            assert!(end - start > COREML_OVERLAP_SAMPLES);
        } else {
            assert_eq!(end - start, COREML_WINDOW_SAMPLES);
            assert_eq!(
                spans[i + 1].0,
                start + COREML_WINDOW_SAMPLES - COREML_OVERLAP_SAMPLES
            );
        }

        // The windows' keep ranges tile the recording with no gaps or overlaps.
        let (from, to) = absolute_keep_window(start, end, COREML_OVERLAP_SAMPLES, i == 0, is_last);
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
fn short_audio_is_one_window() {
    let spans = chunk_spans(
        SAMPLE_RATE as usize,
        COREML_WINDOW_SAMPLES,
        COREML_OVERLAP_SAMPLES,
    );
    assert_eq!(spans, vec![(0, SAMPLE_RATE as usize)]);
    // A single window keeps everything.
    assert_eq!(
        absolute_keep_window(0, SAMPLE_RATE as usize, COREML_OVERLAP_SAMPLES, true, true),
        (0.0, f32::MAX)
    );
}

#[test]
fn empty_audio_has_no_windows() {
    assert!(chunk_spans(0, COREML_WINDOW_SAMPLES, COREML_OVERLAP_SAMPLES).is_empty());
}

#[test]
fn words_are_owned_by_exactly_one_window() {
    // Two windows over 20 s: the cut is at 4 s overlap middle = 13 s.
    let samples = 20 * SAMPLE_RATE as usize;
    let spans = chunk_spans(samples, COREML_WINDOW_SAMPLES, COREML_OVERLAP_SAMPLES);
    assert_eq!(spans.len(), 2);

    // A word at 12 s absolute is heard by both windows.
    let first = own_words(
        vec![token("hello", 11.9, 12.1)],
        spans[0].0,
        spans[0].0,
        spans[0].1,
        COREML_OVERLAP_SAMPLES,
        0,
        2,
    );
    let second = own_words(
        vec![token("hello", 11.9 - 11.0, 12.1 - 11.0)],
        spans[1].0,
        spans[1].0,
        spans[1].1,
        COREML_OVERLAP_SAMPLES,
        1,
        2,
    );
    assert_eq!(first.len() + second.len(), 1);
    assert_eq!(first.len(), 1, "12 s is before the 13 s cut");

    // Kept words carry whole-recording timestamps.
    let late = own_words(
        vec![token("late", 5.0, 5.4)],
        spans[1].0,
        spans[1].0,
        spans[1].1,
        COREML_OVERLAP_SAMPLES,
        1,
        2,
    );
    assert_eq!(late.len(), 1);
    assert!((late[0].start - 16.0).abs() < 1e-3);
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
