import unittest

from app.streaming.revision import RevisionSegment, plan_revision


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.old = "We arrived. I bought apples. Stop!"
        self.spans = [RevisionSegment(0, 11), RevisionSegment(11, 28), RevisionSegment(28, 34)]

    def test_correction_rolls_back_whole_source_segment_not_target_chars(self):
        changed = "We arrived. I bought oranges. Stop!"
        plan = plan_revision(self.old, changed, self.spans)
        self.assertEqual(plan.rollback_segment_index, 1)
        self.assertEqual(plan.rollback_char_end, 11)
        self.assertEqual(plan.replacement_text, " I bought oranges. Stop!")

    def test_truncation_inside_segment_removes_it_and_all_later(self):
        plan = plan_revision(self.old, "We arrived. I bought", self.spans)
        self.assertEqual(plan.rollback_segment_index, 1)
        self.assertEqual(plan.rollback_char_end, 11)

    def test_append_after_sentence_keeps_all_old_segments(self):
        plan = plan_revision(self.old, self.old + " Go!", self.spans)
        self.assertEqual(plan.rollback_segment_index, 3)
        self.assertEqual(plan.replacement_text, " Go!")

    def test_lexical_extension_invalidates_whole_previous_segment(self):
        plan = plan_revision("20", "200 kg", [RevisionSegment(0, 2)])
        self.assertEqual(plan.rollback_segment_index, 0)
        self.assertEqual(plan.replacement_text, "200 kg")

    def test_unchanged_and_empty_segments(self):
        plan = plan_revision(self.old, self.old, self.spans)
        self.assertFalse(plan.changed)
        self.assertEqual(plan.rollback_segment_index, 3)
        self.assertEqual(plan_revision("", "Stop!", []).replacement_text, "Stop!")

    def test_unordered_or_overlapping_source_spans_are_rejected(self):
        with self.assertRaises(ValueError):
            plan_revision(self.old, "Stop!", [RevisionSegment(0, 12), RevisionSegment(11, 28)])
