"""Scheduler concurrency regressions with event-gated in-process backends."""

import asyncio
from dataclasses import replace
import unittest

from app.translation.backend import TranslationError, TranslationResult
from app.translation.scheduler import (
    QueueCapacityError, SchedulerClosedError, StaleJobError, TranslationJob, TranslationScheduler,
)


class Backend:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Queue()
        self.cancelled = asyncio.Queue()
        self.gates = {}
        self.resistant = set()
        self.errors = set()
        self.active = self.max_active = 0
        self.active_sessions = {}
        self.max_session_active = 0

    def hold(self, source, *, resistant=False):
        self.gates[source] = asyncio.Event()
        if resistant:
            self.resistant.add(source)

    async def translate(self, source, *args, **kwargs):
        self.calls.append(source)
        session = source.split(":")[0]
        self.active += 1
        self.active_sessions[session] = self.active_sessions.get(session, 0) + 1
        self.max_active = max(self.max_active, self.active)
        self.max_session_active = max(self.max_session_active, self.active_sessions[session])
        self.started.put_nowait(source)
        try:
            if source in self.gates:
                try:
                    await self.gates[source].wait()
                except asyncio.CancelledError:
                    self.cancelled.put_nowait(source)
                    if source not in self.resistant:
                        raise
                    await self.gates[source].wait()
            if source in self.errors:
                raise RuntimeError("PRIVATE upstream details")
            return TranslationResult(source.upper(), 12.5)
        finally:
            self.active -= 1
            self.active_sessions[session] -= 1


class TranslationSchedulerTests(unittest.IsolatedAsyncioTestCase):
    def make(self, **kwargs):
        self.backend = Backend()
        self.finished = asyncio.Queue()
        self.metrics = []
        self.scheduler = TranslationScheduler(self.backend, metrics=lambda name, value: self.metrics.append((name, value)), **kwargs)

        async def cleanup():
            for gate in self.backend.gates.values():
                gate.set()
            await self.scheduler.close(timeout_s=1)

        self.addAsyncCleanup(cleanup)

    def job(self, session, number, *, segment=None, revision=1, final=True, generation=0):
        return TranslationJob(session, generation, number, revision, segment or f"segment-{number}",
                              f"{session}:{number}", "en", "ko", final=final)

    async def callback(self, job, result, error):
        self.finished.put_nowait((job, result, error))

    async def get(self, queue):
        return await asyncio.wait_for(queue.get(), 1)

    async def test_four_workers_bound_total_concurrency(self):
        self.make(workers=4)
        for index in range(5):
            session = f"s{index}"
            self.backend.hold(f"{session}:0")
            await self.scheduler.submit(self.job(session, 0), self.callback)
        for _ in range(4):
            await self.get(self.backend.started)
        self.assertEqual(self.backend.active, 4)
        self.assertEqual(self.scheduler.pending_count, 1)
        self.assertTrue(self.backend.started.empty())
        for gate in self.backend.gates.values():
            gate.set()
        for _ in range(5):
            await self.get(self.finished)
        self.assertEqual(self.backend.max_active, 4)
        self.assertEqual(self.backend.max_session_active, 1)

    async def test_same_session_is_serial_while_other_sessions_run(self):
        self.make(workers=2)
        for source in ("a:1", "b:1"):
            self.backend.hold(source)
        for job in (self.job("a", 1), self.job("a", 2), self.job("b", 1)):
            await self.scheduler.submit(job, self.callback)
        self.assertEqual({await self.get(self.backend.started), await self.get(self.backend.started)}, {"a:1", "b:1"})
        self.assertNotIn("a:2", self.backend.calls)
        self.backend.gates["a:1"].set()
        self.assertEqual(await self.get(self.backend.started), "a:2")
        self.assertEqual(self.backend.max_session_active, 1)

    async def test_round_robin_does_not_let_hot_session_starve_other_sessions(self):
        self.make(workers=1)
        self.backend.hold("a:1")
        await self.scheduler.submit(self.job("a", 1), self.callback)
        await self.get(self.backend.started)
        for job in (self.job("a", 2), self.job("a", 3), self.job("b", 1), self.job("c", 1)):
            await self.scheduler.submit(job, self.callback)
        self.backend.gates["a:1"].set()
        for _ in range(5):
            await self.get(self.finished)
        self.assertEqual(self.backend.calls, ["a:1", "b:1", "c:1", "a:2", "a:3"])

    async def test_only_pending_partial_for_same_segment_is_coalesced(self):
        self.make(workers=1, max_pending=2)
        self.backend.hold("a:0")
        await self.scheduler.submit(self.job("a", 0), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 1, segment="partial", final=False), self.callback)
        await self.scheduler.submit(self.job("a", 2, segment="other"), self.callback)
        await self.scheduler.submit(self.job("a", 3, segment="partial", revision=2, final=False), self.callback)
        self.assertEqual(self.scheduler.pending_count, 2)
        self.backend.gates["a:0"].set()
        for _ in range(3):
            await self.get(self.finished)
        self.assertEqual(self.backend.calls, ["a:0", "a:3", "a:2"])
        self.assertIn(("translation_partial_coalesced", 1), self.metrics)

    async def test_full_queue_explicitly_rejects_without_evicting_accepted_finals(self):
        self.make(workers=1, max_pending=2)
        self.backend.hold("a:0")
        await self.scheduler.submit(self.job("a", 0), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 1, segment="accepted"), self.callback)
        await self.scheduler.submit(self.job("b", 1), self.callback)
        for job in (self.job("c", 1), self.job("a", 2, segment="accepted", revision=2, final=False)):
            with self.assertRaises(QueueCapacityError) as caught:
                await self.scheduler.submit(job, self.callback)
            self.assertTrue(caught.exception.retryable)
        self.assertEqual(self.scheduler.pending_count, 2)
        self.backend.gates["a:0"].set()
        delivered = [await self.get(self.finished) for _ in range(3)]
        self.assertEqual({item[0].source for item in delivered}, {"a:0", "a:1", "b:1"})

    async def test_final_can_replace_its_pending_partial_at_capacity_without_dropping_other_final(self):
        self.make(workers=1, max_pending=2)
        self.backend.hold("a:0")
        await self.scheduler.submit(self.job("a", 0), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 1, segment="partial", final=False), self.callback)
        await self.scheduler.submit(self.job("b", 1), self.callback)
        await self.scheduler.submit(self.job("a", 2, segment="partial", revision=2, final=True), self.callback)
        self.backend.gates["a:0"].set()
        delivered = [await self.get(self.finished) for _ in range(3)]
        self.assertEqual({item[0].source for item in delivered}, {"a:0", "a:2", "b:1"})
        self.assertTrue(all(item[0].final for item in delivered))

    async def test_per_session_capacity_rejection_keeps_other_sessions_admissible(self):
        self.make(workers=1, max_pending=8, max_pending_per_session=1)
        self.backend.hold("a:0")
        await self.scheduler.submit(self.job("a", 0), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 1), self.callback)
        with self.assertRaises(QueueCapacityError):
            await self.scheduler.submit(self.job("a", 2), self.callback)
        await self.scheduler.submit(self.job("b", 1), self.callback)
        self.backend.gates["a:0"].set()
        for _ in range(3):
            await self.get(self.finished)
        self.assertEqual(self.backend.calls, ["a:0", "b:1", "a:1"])

    async def test_new_revision_cancels_child_not_worker_and_retains_resistant_slot(self):
        self.make(workers=2)
        self.backend.hold("a:1", resistant=True)
        await self.scheduler.submit(self.job("a", 1, segment="same", final=False), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 2, segment="same", revision=2), self.callback)
        self.assertEqual(await self.get(self.backend.cancelled), "a:1")
        await self.scheduler.submit(self.job("b", 1), self.callback)
        self.assertEqual(await self.get(self.backend.started), "b:1")
        self.assertEqual((await self.get(self.finished))[0].source, "b:1")
        self.assertNotIn("a:2", self.backend.calls)
        self.backend.gates["a:1"].set()
        self.assertEqual((await self.get(self.finished))[0].source, "a:2")
        self.assertTrue(self.finished.empty())
        self.assertEqual(self.backend.max_session_active, 1)
        self.assertTrue(all(not task.done() for task in self.scheduler._workers))

    async def test_cancel_session_suppresses_old_generation_and_pending_finals(self):
        self.make(workers=2)
        self.backend.hold("a:1", resistant=True)
        await self.scheduler.submit(self.job("a", 1), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 2), self.callback)
        await self.scheduler.cancel_session("a")
        await self.get(self.backend.cancelled)
        await self.scheduler.submit(self.job("a", 3, generation=1), self.callback)
        self.assertNotIn("a:3", self.backend.calls)
        self.backend.gates["a:1"].set()
        self.assertEqual((await self.get(self.finished))[0].source, "a:3")
        self.assertNotIn("a:2", self.backend.calls)
        self.assertTrue(self.finished.empty())
        self.assertEqual(self.backend.max_session_active, 1)

    async def test_explicit_invalidate_can_cancel_final_but_keeps_other_segments(self):
        self.make(workers=1)
        self.backend.hold("a:1", resistant=True)
        await self.scheduler.submit(self.job("a", 1, segment="corrected"), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 2, segment="corrected", revision=2), self.callback)
        await self.scheduler.submit(self.job("a", 3, segment="retained"), self.callback)
        await self.scheduler.invalidate("a", ["corrected"])
        await self.get(self.backend.cancelled)
        self.backend.gates["a:1"].set()
        self.assertEqual((await self.get(self.finished))[0].source, "a:3")
        self.assertEqual(self.backend.calls, ["a:1", "a:3"])

    async def test_backend_and_callback_errors_do_not_kill_worker(self):
        self.make(workers=1)
        self.backend.errors.add("a:1")

        async def failing_callback(job, result, error):
            await self.callback(job, result, error)
            raise RuntimeError("application callback failed")

        await self.scheduler.submit(self.job("a", 1), failing_callback)
        await self.scheduler.submit(self.job("a", 2), self.callback)
        first, second = await self.get(self.finished), await self.get(self.finished)
        self.assertIsNone(first[1])
        self.assertIsInstance(first[2], TranslationError)
        self.assertEqual(str(first[2]), "backend_error")
        self.assertIsNone(second[2])
        self.assertEqual(second[1].text, "A:2")
        self.assertFalse(self.scheduler._workers[0].done())

    async def test_backend_self_cancellation_is_reported_without_killing_worker(self):
        self.make(workers=1)
        original = self.backend.translate

        async def self_cancelling(source, *args, **kwargs):
            if source == "a:1":
                raise asyncio.CancelledError()
            return await original(source, *args, **kwargs)

        self.backend.translate = self_cancelling
        await self.scheduler.submit(self.job("a", 1), self.callback)
        await self.scheduler.submit(self.job("a", 2), self.callback)
        first, second = await self.get(self.finished), await self.get(self.finished)
        self.assertEqual(first[2].code, "backend_cancelled")
        self.assertEqual(second[1].text, "A:2")
        self.assertFalse(self.scheduler._workers[0].done())

    async def test_shutdown_is_bounded_without_freeing_a_resistant_physical_slot(self):
        self.make(workers=1)
        self.backend.hold("a:1", resistant=True)
        await self.scheduler.submit(self.job("a", 1), self.callback)
        await self.get(self.backend.started)
        with self.assertRaises(TimeoutError):
            await self.scheduler.close(timeout_s=.01)
        self.assertEqual(self.scheduler.active_count, 1)
        with self.assertRaises(SchedulerClosedError):
            await self.scheduler.submit(self.job("b", 1), self.callback)
        self.backend.gates["a:1"].set()
        await self.scheduler.close(timeout_s=1)
        self.assertTrue(self.finished.empty())
        self.assertEqual(self.scheduler.active_count, 0)

    async def test_stale_revision_rejection_leaves_newer_pending_job_intact(self):
        self.make(workers=1)
        self.backend.hold("a:0")
        await self.scheduler.submit(self.job("a", 0), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 1, segment="same", revision=3, final=False), self.callback)
        with self.assertRaises(StaleJobError):
            await self.scheduler.submit(self.job("a", 2, segment="same", revision=2, final=False), self.callback)
        self.backend.gates["a:0"].set()
        for _ in range(2):
            await self.get(self.finished)
        self.assertEqual(self.backend.calls, ["a:0", "a:1"])

    async def test_active_final_is_not_cancelled_by_a_new_partial(self):
        self.make(workers=1)
        self.backend.hold("a:1")
        await self.scheduler.submit(self.job("a", 1, segment="same"), self.callback)
        await self.get(self.backend.started)
        await self.scheduler.submit(self.job("a", 2, segment="same", revision=2, final=False), self.callback)
        self.assertTrue(self.backend.cancelled.empty())
        self.backend.gates["a:1"].set()
        self.assertEqual([(await self.get(self.finished))[0].source for _ in range(2)], ["a:1", "a:2"])

    def test_job_snapshots_context_and_glossary_and_accepts_sequence_zero(self):
        context, glossary = ["previous"], {"term": "용어"}
        job = TranslationJob("session", 0, 0, 1, "segment", "source", "en", "ko",
                             source_context=context, glossary=glossary)
        context.append("changed")
        glossary["term"] = "changed"
        self.assertEqual(job.source_context, ("previous",))
        self.assertEqual(dict(job.glossary), {"term": "용어"})
        with self.assertRaises(TypeError):
            job.glossary["new"] = "value"
        with self.assertRaises(ValueError):
            replace(job, sequence=True)


if __name__ == "__main__":
    unittest.main()
