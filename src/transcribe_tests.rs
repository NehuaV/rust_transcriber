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

#[test]
fn a_word_kept_by_two_windows_is_merged_once() {
    let merged = merge_windows(vec![
        vec![token("count", 10.0, 10.3), token("The", 12.8, 13.0)],
        vec![token("The", 12.9, 13.1), token("counties", 13.2, 13.7)],
    ]);
    let text: Vec<&str> = merged.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(text, vec!["count", "The", "counties"]);
}

#[test]
fn a_real_repetition_across_a_seam_is_kept() {
    // "the the": the second one is spoken 0.4 s later.
    let merged = merge_windows(vec![
        vec![token("the", 12.5, 12.7)],
        vec![token("the", 12.9, 13.1), token("voice", 13.2, 13.5)],
    ]);
    assert_eq!(merged.len(), 3);
}

#[test]
fn repetition_inside_one_window_is_never_merged() {
    let merged = merge_windows(vec![vec![token("I", 1.0, 1.1), token("I", 1.1, 1.2)]]);
    assert_eq!(merged.len(), 2);
}

#[test]
fn same_word_ignores_case_and_punctuation() {
    assert!(same_word("river.", "River"));
    assert!(!same_word("river", "rivers"));
    assert!(!same_word(".", ","));
}

#[test]
fn variants_start_with_the_plain_decode_and_are_distinct() {
    assert_eq!(VARIANTS[0], (0, coreml::LengthPolicy::ACTUAL));
    for (i, a) in VARIANTS.iter().enumerate() {
        assert!(!VARIANTS[i + 1..].contains(a), "duplicate variant {a:?}");
    }
    assert!(DEFAULT_VARIANTS as usize <= VARIANTS.len());
    // The cut shifts never exceed the overlap's context on either side.
    for (shift, _) in VARIANTS {
        assert!(shift.abs() <= 3);
    }
}

#[test]
fn window_threads_are_bounded() {
    assert_eq!(window_threads(1), 1);
    assert!(window_threads(100) >= 1);
    assert!(window_threads(100) <= MAX_WINDOW_THREADS);
    assert_eq!(window_threads(0), 1);
}

fn token(text: &str, start: f32, end: f32) -> TimedToken {
    TimedToken {
        text: text.to_string(),
        start,
        end,
    }
}
