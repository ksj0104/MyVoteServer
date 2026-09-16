import unittest

from app.streaming.commit import DraftCommitter


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.committer = DraftCommitter(clock=lambda: self.now)

    def propose(self, target="멈추세요!", **kwargs):
        return self.committer.propose(4, 10, " Stop!", target, **kwargs)

    def test_whole_draft_commits_after_identical_count(self):
        first = self.propose()
        self.assertEqual(first.observations, 1)
        self.assertFalse(self.committer.ready())
        second = self.propose()
        self.assertEqual(second.observations, 2)
        self.assertTrue(self.committer.ready())
        committed = self.committer.mark_committed()
        self.assertEqual(committed.target_text, "멈추세요!")
        self.assertTrue(committed.committed)
        self.assertFalse(self.committer.ready())

    def test_timer_commits_only_unchanged_whole_draft(self):
        self.propose()
        self.now = 0.6
        self.assertFalse(self.committer.ready())
        self.propose("정지하세요!")
        self.now = 0.8
        self.assertFalse(self.committer.ready())
        self.now = 1.3
        self.assertTrue(self.committer.ready())

    def test_source_revision_resets_observation_count_and_age(self):
        self.propose(source_revision=1)
        self.now = 0.5
        revised = self.propose(source_revision=2)
        self.assertEqual(revised.observations, 1)
        self.now = 0.8
        self.assertFalse(self.committer.ready())

    def test_final_is_not_a_commit_override(self):
        self.propose(is_final=True)
        self.assertFalse(self.committer.ready())
        with self.assertRaises(ValueError):
            self.committer.mark_committed()
        self.now = 0.7
        self.assertTrue(self.committer.ready())

    def test_committed_draft_needs_explicit_reset_and_exact_source_span(self):
        self.propose()
        self.propose()
        self.committer.mark_committed()
        with self.assertRaises(ValueError):
            self.propose("정지!")
        self.committer.reset()
        self.assertFalse(self.committer.ready())
        with self.assertRaises(ValueError):
            self.committer.propose(0, 3, "Stop!", "정지!")
