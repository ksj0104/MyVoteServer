import unittest

from app.streaming.stability import StabilityTracker
from app.utils.text import common_prefix_length, token_char_end, tokenize


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class TextTests(unittest.TestCase):
    def test_lossless_unicode_token_offsets(self):
        source = "  café\t가나다 你好。 👩🏽‍💻 e\u0301  12.50 kg!\n"
        tokens = tokenize(source)
        self.assertEqual([t.text for t in tokens], ["café", "가나다", "你", "好", "。", "👩🏽‍💻", "e\u0301", "12.50", "kg", "!"])
        self.assertTrue(all(source[token.start:token.end] == token.text for token in tokens))
        self.assertEqual(source[:token_char_end(tokens, 4)], "  café\t가나다 你好")
        self.assertEqual(token_char_end(tokens, 0), 0)
        with self.assertRaises(ValueError):
            token_char_end(tokens, len(tokens) + 1)

    def test_exact_lcp_does_not_normalize_changed_words(self):
        self.assertEqual(common_prefix_length((("I", "see"), ("I", "saw"), ("I", "see"))), 1)
        self.assertEqual(common_prefix_length(()), 0)


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.tracker = StabilityTracker(clock=self.clock)

    def observe_four(self, text, **kwargs):
        for sequence in range(4):
            result = self.tracker.observe(text, sequence, **kwargs)
        return result

    def test_four_hypotheses_required_even_with_age_and_pause(self):
        self.tracker.observe("Stop.", 0, confidences=1, pause_ms=500)
        self.clock.value = 1000
        for sequence in (1, 2):
            result = self.tracker.observe("Stop.", sequence, confidences=1, pause_ms=500)
            self.assertEqual(result.stable_char_end, 0)
        result = self.tracker.observe("Stop.", 3, confidences=1, pause_ms=500)
        self.assertEqual(result.stable_char_end, len("Stop."))

    def test_missing_confidence_is_omitted_not_zero(self):
        missing = self.observe_four("We arrived.")
        self.assertIsNone(missing.confidence)
        self.assertAlmostEqual(missing.score, 0.35 / 0.55)
        self.clock.value = 0.4
        refreshed = self.tracker.refresh()
        self.assertEqual(refreshed.stable_char_end, len("We arrived."))
        self.assertEqual(refreshed.sequence, 3)
        self.assertEqual(len(self.tracker._history), 4)
        low = StabilityTracker(clock=self.clock)
        for seq in range(4):
            known_zero = low.observe("We arrived.", seq, confidences=0)
        self.assertAlmostEqual(known_zero.score, 0.35 / 0.85)

    def test_available_weights_and_supplied_pause_are_normalized(self):
        result = self.observe_four("Stop.", confidences=0.8, pause_ms=250)
        self.assertAlmostEqual(result.score, 0.35 + 0.30 * 0.8 + 0.15 * 0.5)
        custom = StabilityTracker(clock=self.clock, agreement_weight=1, confidence_weight=0,
                                  age_weight=0, pause_weight=0)
        for seq in range(4):
            result = custom.observe("Stop.", seq)
        self.assertEqual(result.score, 1)

    def test_only_common_prefix_survives_hypothesis_changes(self):
        for seq, source in enumerate(("We have twenty", "We have twenty kilos", "We have thirty kilos", "We have thirty")):
            self.clock.value = seq
            result = self.tracker.observe(source, seq)
        self.assertEqual(result.stable_token_count, 2)
        self.assertEqual(result.text[:result.stable_char_end], "We have")

    def test_refresh_does_not_create_agreement_and_clock_cannot_move_backwards(self):
        self.tracker.observe("Stop.", 0)
        self.clock.value = 5
        self.assertEqual(self.tracker.refresh().stable_char_end, 0)
        self.clock.value = -10
        self.assertEqual(self.tracker.refresh().age_ms, 5000)

    def test_conflicting_partial_waits_and_final_needs_explicit_reconciliation(self):
        initial = self.tracker.observe("I bought apples.", 0, is_final=True)
        self.tracker.commit(initial.stable_char_end)
        partial = self.tracker.observe("I bought oranges.", 1)
        self.assertTrue(partial.prefix_conflict)
        self.assertFalse(partial.reconcile_required)
        self.assertEqual(partial.stable_char_end, 0)
        with self.assertRaises(ValueError):
            self.tracker.commit(0)
        final = self.tracker.observe("I bought oranges.", 2, is_final=True)
        self.assertTrue(final.reconcile_required)
        with self.assertRaises(ValueError):
            self.tracker.commit(final.stable_char_end)
        accepted = self.tracker.commit(final.stable_char_end, allow_final_reconcile=True)
        self.assertFalse(accepted.prefix_conflict)

    def test_word_extension_is_also_a_committed_prefix_conflict(self):
        source = self.tracker.observe("20", 0, is_final=True)
        self.tracker.commit(source.stable_char_end)
        result = self.tracker.observe("200 kilograms", 1)
        self.assertTrue(result.prefix_conflict)
        self.assertEqual(result.stable_char_end, 0)

    def test_new_suffix_does_not_retract_committed_exact_prefix(self):
        source = self.tracker.observe("Stop.", 0, is_final=True)
        self.tracker.commit(source.stable_char_end)
        result = self.tracker.observe("Stop. Wait.", 1)
        self.assertFalse(result.prefix_conflict)
        self.assertEqual(result.stable_char_end, len("Stop."))

    def test_invalid_confidence_ignored_and_input_limits_enforced(self):
        result = self.observe_four("Stop.", confidences=[float("nan"), -1])
        self.assertIsNone(result.confidence)
        with self.assertRaises(ValueError):
            self.tracker.observe("Duplicate", 3)
        with self.assertRaises(ValueError):
            self.tracker.observe("x" * 8001, 4)
        with self.assertRaises(ValueError):
            self.tracker.commit(2)

    def test_optional_open_lexical_tail_hold_ignores_age_confidence_and_pause(self):
        tracker = StabilityTracker(clock=self.clock, hold_last_token=True)
        source = "She wants to go to the hosp"
        for seq in range(4):
            self.clock.value = seq
            result = tracker.observe(source, seq, confidences=1, pause_ms=60000)
        self.assertEqual(result.stable_char_end, len("She wants to go to the"))
        self.clock.value = 1000
        self.assertEqual(tracker.refresh().stable_char_end, result.stable_char_end)
        final = tracker.observe(source, 4, is_final=True)
        self.assertEqual(final.stable_char_end, len(source))

    def test_open_tail_is_released_by_whitespace_or_punctuation(self):
        for source in ("Go ", "Stop!", "你好。", "보고서를 보내세요."):
            with self.subTest(source=source):
                tracker = StabilityTracker(clock=self.clock, hold_last_token=True)
                for seq in range(4):
                    result = tracker.observe(source, seq, confidences=1, pause_ms=500)
                self.assertEqual(result.stable_char_end, len(source.rstrip()))
        tracker = StabilityTracker(clock=self.clock, hold_last_token=True)
        for seq in range(4):
            result = tracker.observe("你好", seq, confidences=1, pause_ms=500)
        self.assertEqual(result.stable_char_end, 1)
