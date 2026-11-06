use super::*;

/// The fixed CoreML window: 15 s at 16 kHz.
const WINDOW: usize = 240_000;
/// Overlap between neighbouring windows: 4 s.
const OVERLAP: usize = 64_000;
const RATE: usize = SAMPLE_RATE as usize;

fn word(text: &str, start: f32, end: f32) -> Word {
    Word {
        text: text.to_string(),
        start,
        end,
    }
}

#[test]
fn windows_cover_the_audio_with_overlap() {
    // 2 minutes 7 seconds of audio.
    let samples = 127 * RATE;
    let plan = Plan::new(samples, WINDOW, OVERLAP);
    let spans = plan.spans();
    assert!(plan.len() > 1);
    assert_eq!(spans.first(), Some(&(0..WINDOW)));
    assert_eq!(spans.last().map(|span| span.end), Some(samples));

    let mut covered_until = 0.0f32;
    for (i, span) in spans.iter().enumerate() {
        assert!(span.start < span.end);
        assert!(span.len() <= WINDOW);
        let is_last = i + 1 == spans.len();
        if is_last {
            assert!(span.len() > OVERLAP, "a final sliver would own nothing");
        } else {
            assert_eq!(span.len(), WINDOW);
            assert_eq!(spans[i + 1].start, span.start + WINDOW - OVERLAP);
        }

        // The keep ranges tile the recording with no gaps and no overlaps.
        let keep = plan.keep_range(i);
        if i == 0 {
            assert_eq!(keep.start, 0.0);
        } else {
            assert!((keep.start - covered_until).abs() < 1e-3);
        }
        let logical_end = if is_last {
            span.end as f32 / SAMPLE_RATE as f32
        } else {
            keep.end
        };
        assert!(logical_end > keep.start);
        covered_until = logical_end;
    }
    assert!((covered_until - samples as f32 / SAMPLE_RATE as f32).abs() < 1e-2);
}

#[test]
fn short_audio_is_one_window_that_keeps_everything() {
    let plan = Plan::new(RATE, WINDOW, OVERLAP);
    assert_eq!(plan.len(), 1);
    assert_eq!(plan.spans()[0], 0..RATE);
    assert_eq!(plan.keep_range(0), 0.0..f32::MAX);
}

#[test]
fn audio_exactly_one_window_long_is_one_window() {
    assert_eq!(Plan::new(WINDOW, WINDOW, OVERLAP).len(), 1);
    assert_eq!(Plan::new(WINDOW + 1, WINDOW, OVERLAP).len(), 2);
}

#[test]
fn empty_audio_has_no_windows() {
    let plan = Plan::new(0, WINDOW, OVERLAP);
    assert_eq!(plan.len(), 0);
    assert!(plan.spans().is_empty());
}

#[test]
#[should_panic(expected = "longer than the overlap")]
fn a_window_must_be_longer_than_its_overlap() {
    let _ = Plan::new(10, 5, 5);
}

#[test]
fn a_word_is_owned_by_exactly_one_window() {
    // Two windows over 20 s. The cut is the middle of the overlap: 13 s.
    let plan = Plan::new(20 * RATE, WINDOW, OVERLAP);
    assert_eq!(plan.len(), 2);

    // A word at 12 s absolute is heard by both windows.
    let first = plan.own_words(0, vec![word("hello", 11.9, 12.1)], plan.spans()[0].start);
    let second = plan.own_words(
        1,
        vec![word("hello", 11.9 - 11.0, 12.1 - 11.0)],
        plan.spans()[1].start,
    );
    assert_eq!(first.len() + second.len(), 1);
    assert_eq!(first.len(), 1, "12 s is before the 13 s cut");
}

#[test]
fn owned_words_carry_whole_recording_timestamps() {
    let plan = Plan::new(20 * RATE, WINDOW, OVERLAP);
    let late = plan.own_words(1, vec![word("late", 5.0, 5.4)], plan.spans()[1].start);
    assert_eq!(late.len(), 1);
    assert!((late[0].start - 16.0).abs() < 1e-3);
    assert!((late[0].end - 16.4).abs() < 1e-3);
}

#[test]
fn a_shifted_decode_is_placed_by_where_it_started() {
    // Decoded from a cut 1 s earlier than the window start (11 s).
    let plan = Plan::new(20 * RATE, WINDOW, OVERLAP);
    let from = plan.spans()[1].start - RATE;
    let owned = plan.own_words(1, vec![word("x", 6.0, 6.4)], from);
    assert_eq!(owned.len(), 1);
    assert!((owned[0].start - 16.0).abs() < 1e-3, "10 s offset + 6 s");
}

#[test]
fn owning_folds_punctuation_and_drops_blank_words() {
    let plan = Plan::new(RATE, WINDOW, OVERLAP);
    let owned = plan.own_words(
        0,
        vec![
            word("Hi", 0.1, 0.3),
            word(".", 0.3, 0.35),
            word("  ", 0.4, 0.5),
        ],
        0,
    );
    assert_eq!(owned, vec![word("Hi.", 0.1, 0.3)]);
}

#[test]
fn a_word_kept_by_two_windows_is_merged_once() {
    let merged = merge(vec![
        vec![word("count", 10.0, 10.3), word("The", 12.8, 13.0)],
        vec![word("The", 12.9, 13.1), word("counties", 13.2, 13.7)],
    ]);
    let text: Vec<&str> = merged.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(text, ["count", "The", "counties"]);
}

#[test]
fn a_real_repetition_across_a_seam_is_kept() {
    // "the the": the second one is spoken 0.4 s later.
    let merged = merge(vec![
        vec![word("the", 12.5, 12.7)],
        vec![word("the", 12.9, 13.1), word("voice", 13.2, 13.5)],
    ]);
    assert_eq!(merged.len(), 3);
}

#[test]
fn repetition_inside_one_window_is_never_merged() {
    let merged = merge(vec![vec![word("I", 1.0, 1.1), word("I", 1.1, 1.2)]]);
    assert_eq!(merged.len(), 2);
}

#[test]
fn only_the_first_word_of_a_window_can_be_a_seam_duplicate() {
    let merged = merge(vec![
        vec![word("a", 1.0, 1.1)],
        vec![word("b", 1.2, 1.3), word("a", 1.0, 1.1)],
    ]);
    assert_eq!(merged.len(), 3);
}
