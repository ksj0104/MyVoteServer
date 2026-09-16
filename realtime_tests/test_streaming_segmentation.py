import unittest

from app.streaming.policy import ReadWritePolicy
from app.streaming.segmentation import SemanticSegmenter, segment_prefix
from app.utils.text import tokenize


class SegmentationTests(unittest.TestCase):
    def candidate(self, text, **kwargs):
        return segment_prefix(text, len(text), **kwargs)

    def test_dangling_english_clauses_and_amounts_always_wait(self):
        for text in ("The shipment weighs about twenty", "The shipment weighs about twenty.",
                     "If it rains.", "Because we arrived.", "Although she agrees.",
                     "We need to", "I will", "Send the", "We arrived and", "I think he",
                     "The price is approximately", "She waited 20.", "The cost is", "Please send.",
                     "The shipment weighs.", "The quick brown fox.", "The talented engineer."):
            with self.subTest(text=text):
                self.assertIsNone(self.candidate(text))

    def test_completed_amount_and_independent_clauses(self):
        for text in ("The shipment weighs about twenty kilograms.", "The price is 20 dollars.",
                     "If it rains, take a taxi.", "Because we arrived early, we waited outside.",
                     "I bought three books.", "The answer is twenty."):
            with self.subTest(text=text):
                candidate = self.candidate(text)
                self.assertIsNotNone(candidate)
                self.assertEqual(candidate.text, text)

    def test_short_complete_commands_and_questions(self):
        for text in ("Stop.", "Go!", "Wait", "Please wait.", "Help!", "Ready?", "Are you ready?",
                     "What is it?", "What do you want?", "What do you think?", "How much does it weigh?", "Thank you."):
            with self.subTest(text=text):
                candidate = self.candidate(text)
                self.assertIsNotNone(candidate)
                self.assertTrue(candidate.strong_boundary)

    def test_explicit_complement_examples_wait_for_their_own_predicate(self):
        for text in ("I think that this solution", "I think that this solution.",
                     "He said that the shipment", "He said that the shipment.",
                     "I believe this solution", "I believe this solution.",
                     "The computer works but it", "The computer works but the monitor",
                     "What do you?", "Can you please?", "Is the cat?"):
            with self.subTest(text=text):
                self.assertIsNone(self.candidate(text))
        for text in ("I think that this solution works.", "He said that the shipment arrived.",
                     "I believe this solution will work.", "The computer works but it crashes.",
                     "I believe in this solution.", "I think about this solution.", "I said yes.",
                     "I bought apples and a pear."):
            with self.subTest(text=text):
                self.assertIsNotNone(self.candidate(text))

    def test_truncated_asr_word_outside_stable_prefix_cannot_complete_a_clause(self):
        source = "She wants to go to the hosp"
        stable_end = len("She wants to go to the")
        self.assertIsNone(segment_prefix(source, stable_end))
        full = "She wants to go to the hospital."
        self.assertIsNotNone(self.candidate(full))
        amount = "The shipment weighs about twenty kilograms."
        self.assertIsNone(segment_prefix(amount, len("The shipment weighs about twenty")))
        self.assertIsNotNone(self.candidate(amount))

    def test_nearest_explicit_question_allows_only_appropriate_short_answers(self):
        self.assertIsNone(self.candidate("Twenty."))
        self.assertIsNotNone(self.candidate("Twenty.", context="How many tickets do you need?"))
        self.assertIsNone(self.candidate("Twenty.", context="How long did you wait?"))
        self.assertIsNotNone(self.candidate("Twenty minutes.", context="How long did you wait?"))
        self.assertIsNone(self.candidate("Yes.", context="What is your name?"))
        self.assertIsNotNone(self.candidate("Yes.", context="Are you ready?"))
        self.assertIsNone(self.candidate("Twenty.", context="How many?\nWe arrived."))

    def test_korean_connectives_do_not_become_complete_from_punctuation(self):
        for text in ("배송물의 무게는 약 20", "배송물의 무게는 스물.", "오늘은 비가 오지만.",
                     "내일 시간이 있으면", "보고서를 보냈는데.", "아직 끝나지 않아서",
                     "결과를 확인하기 위해", "음료가 아니라", "20킬로그램보다.", "푸른 바다."):
            with self.subTest(text=text):
                self.assertIsNone(self.candidate(text, language="ko"))
        for text in ("배송물의 무게는 약 20킬로그램입니다.", "보고서를 보내 주세요.", "감사합니다.", "멈춰!", "준비됐나요?"):
            with self.subTest(text=text):
                self.assertIsNotNone(self.candidate(text, language="ko"))

    def test_exact_prefix_and_whitespace_are_preserved(self):
        text = "  Stop!\n  We arrived.\tThe shipment weighs about twenty"
        candidate = self.candidate(text)
        self.assertEqual(candidate.text, "  Stop!\n  We arrived.")
        self.assertEqual(text[candidate.start:candidate.end], candidate.text)
        self.assertEqual(candidate.end, len("  Stop!\n  We arrived."))
        next_candidate = self.candidate("Stop!\tGo!", start=len("Stop!"))
        self.assertEqual(next_candidate.text, "\tGo!")

    def test_never_skips_an_incomplete_leading_sentence(self):
        self.assertIsNone(self.candidate("Because he left. Stop!"))

    def test_stability_cannot_cut_words_or_expose_unstable_suffix(self):
        text = "We arrived. Stop!"
        candidate = segment_prefix(text, len("We arrived."))
        self.assertEqual(candidate.text, "We arrived.")
        with self.assertRaises(ValueError):
            segment_prefix(text, 3)
        self.assertIsNone(segment_prefix("The shipment weighs 20 kg.", len("The shipment weighs 20")))

    def test_unicode_cjk_and_decimal_offsets(self):
        candidate = self.candidate(" 你好。次の", language="zh")
        self.assertEqual(candidate.text, " 你好。")
        candidate = self.candidate("It weighs 12.50 kilograms.")
        self.assertEqual(candidate.text, "It weighs 12.50 kilograms.")
        self.assertEqual(candidate.end, tokenize(candidate.text)[-1].end)

    def test_budget_never_cuts_meaning_and_unclosed_source_waits(self):
        self.assertIsNone(self.candidate("The shipment weighs twenty kilograms.", max_tokens=3))
        self.assertEqual(self.candidate("Stop! Go!", max_tokens=2).text, "Stop!")
        for text in ("We arrived...", "We arrived…", "We arrived (early."):
            with self.subTest(text=text):
                self.assertIsNone(self.candidate(text))

    def test_oversize_complete_opt_in_keeps_whole_first_sentence(self):
        text = "The shipment weighs twenty kilograms."
        self.assertIsNone(self.candidate(text, max_tokens=3))
        self.assertEqual(self.candidate(text, max_tokens=3, allow_oversize_complete=True).text, text)
        self.assertEqual(self.candidate("Stop! " + text, max_tokens=3, allow_oversize_complete=True).text, "Stop!")
        self.assertIsNone(self.candidate("The shipment weighs twenty", max_tokens=3, allow_oversize_complete=True))
        self.assertIsNone(self.candidate(text[:-1], max_tokens=3, allow_oversize_complete=True))

    def test_gateway_segmenter_explicitly_advertises_only_supported_languages(self):
        segmenter = SemanticSegmenter()
        for language in ("en", "en-US", "EN_us", "ko", "ko-KR"):
            self.assertTrue(segmenter.supports(language))
        for language in (None, "zh", "ja", "fr", "english", "korean"):
            self.assertFalse(segmenter.supports(language))
        self.assertEqual(segmenter.select("Stop!", 5, language="en").text, "Stop!")
        with self.assertRaisesRegex(ValueError, "UNKNOWN_SOURCE_LANGUAGE"):
            segmenter.select("你好。", 3, language="zh")


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_time_pause_and_final_cannot_force_incomplete_source(self):
        policy = ReadWritePolicy()
        for text in ("If we arrive.", "The shipment weighs about twenty", "날씨가 좋으면."):
            candidate = segment_prefix(text, len(text))
            decision = await policy.decide(candidate, stability_score=1, stable_token_count=100,
                                           age_ms=1_000_000, pause_ms=60000, is_final=True)
            self.assertEqual(decision.action, "READ")
            self.assertEqual(decision.reason, "incomplete_source")

    async def test_strong_short_complete_units_bypass_length_preferences_only(self):
        policy = ReadWritePolicy()
        source = "Stop!"
        candidate = segment_prefix(source, len(source))
        ready = await policy.decide(candidate, stability_score=1, stable_token_count=2)
        self.assertEqual(ready.action, "WRITE")
        for kwargs in ({"stability_score": 0.1}, {"stability_score": 1, "prefix_conflict": True}):
            decision = await policy.decide(candidate, stable_token_count=2, **kwargs)
            self.assertEqual(decision.action, "READ")

    async def test_complete_unpunctuated_context_obeys_configurable_latency(self):
        policy = ReadWritePolicy(min_segment_tokens=3, min_stable_tokens=2, target_segment_tokens=10,
                                 max_latency_ms=1500)
        source = "We arrived home early"
        candidate = segment_prefix(source, len(source))
        self.assertIsNotNone(candidate)
        for age, action in ((0, "READ"), (1499, "READ"), (1500, "WRITE")):
            result = await policy.decide(candidate, stability_score=1, stable_token_count=4, age_ms=age)
            self.assertEqual(result.action, action)
