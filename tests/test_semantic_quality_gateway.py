"""Quality gateway metadata contracts without engines, protobuf or network."""

from copy import deepcopy
from dataclasses import dataclass
import asyncio
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts.semantic_quality_gateway import install_gateway_metadata


class ServerMessage:
    def __init__(self, *, kind="", data_json="", **fields):
        self.kind, self.data_json = kind, data_json
        self.__dict__.update(fields)

    def CopyFrom(self, original):
        self.__dict__.clear()
        self.__dict__.update(deepcopy(original.__dict__))


class FakeService:
    def __init__(self, messages, *, config=None):
        self.messages = messages
        self.config = config or SimpleNamespace(max_event_bytes=65536)

    async def StreamSession(self, request_iterator, context):
        self.seen_iterator, self.seen_context = request_iterator, context
        self.seen_auth = context.auth_context()
        for message in self.messages:
            await context.write(message)
        return "original-result"


class RpcAborted(Exception):
    pass


@dataclass(frozen=True)
class FakeConfig:
    max_event_bytes: int = 65536
    finish_timeout_s: float = 30


class QualityGatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.output = io.StringIO()
        capture = patch("sys.stdout", self.output)
        capture.start()
        self.addCleanup(capture.stop)
        self.module = SimpleNamespace(
            GatewayService=FakeService, GatewayConfig=FakeConfig, pb=SimpleNamespace(ServerMessage=ServerMessage),
            grpc=SimpleNamespace(StatusCode=SimpleNamespace(RESOURCE_EXHAUSTED="resource_exhausted")))
        self.auth = {"transport_security_type": [b"ssl"], "x509_pem_cert": [b"test-certificate"]}
        self.context = SimpleNamespace(
            auth_context=Mock(return_value=self.auth), peer_identities=Mock(return_value=[b"client"]),
            peer=Mock(return_value="test-peer"), invocation_metadata=Mock(return_value=("test",)),
            add_done_callback=Mock(), time_remaining=Mock(return_value=23),
            abort=AsyncMock(side_effect=RpcAborted), write=AsyncMock(return_value="written"))

    def started(self, **extra):
        payload = {
            "semantic_translation_status": "enabled",
            "semantic_min_request_interval_ms": 800,
            "semantic_max_hold_ms": 4000,
            "semantic_total_budget_ms": 8000,
            "semantic_request_timeout_ms": 3500,
            "capabilities": ["semantic_translation_v1"],
            **extra,
        }
        return ServerMessage(kind="session.started", data_json=json.dumps(payload, ensure_ascii=False),
                             session_id="session", segment_id="", event_sequence=1,
                             server_elapsed_ms=12.25, unknown_fields={"retained": [1, 2]})

    async def stream(self, messages, *, limit=65536):
        service = self.module.GatewayService(messages, config=FakeConfig(max_event_bytes=limit))
        iterator = object()
        result = await service.StreamSession(iterator, self.context)
        self.assertEqual(result, "original-result")
        self.assertIs(service.seen_iterator, iterator)
        return service

    async def test_enabled_started_copies_message_and_changes_only_quality_metadata(self):
        install_gateway_metadata(self.module)
        original = self.started(engine_revision="verified", message="원문 유지")
        snapshot = deepcopy(original.__dict__)
        await self.stream([original])
        sent = self.context.write.await_args.args[0]
        self.assertIsNot(sent, original)
        expected = json.loads(original.data_json)
        expected.update(semantic_min_request_interval_ms=1500, semantic_max_hold_ms=12000,
                        semantic_total_budget_ms=45000, semantic_quality_first=True,
                        semantic_request_timeout_ms=20000, semantic_orchestrator_budget_ms=5000,
                        semantic_translation_budget_ms=15000,
                        semantic_finish_timeout_ms=30000,
                        semantic_incomplete_policy="source_only")
        self.assertEqual(json.loads(sent.data_json), expected)
        self.assertEqual({key: value for key, value in sent.__dict__.items() if key != "data_json"},
                         {key: value for key, value in snapshot.items() if key != "data_json"})
        self.assertEqual(original.__dict__, snapshot)
        self.context.abort.assert_not_awaited()

    async def test_other_events_and_nonsemantic_or_malformed_started_pass_through_unchanged(self):
        install_gateway_metadata(self.module)
        messages = [
            ServerMessage(kind="caption.source", data_json='{"semantic_translation_status":"enabled"}'),
            ServerMessage(kind="translation.completed", data_json="not-json"),
            *(ServerMessage(kind="session.started", data_json=raw) for raw in (
                "not-json", "null", "[]", "42", "{}",
                '{"semantic_translation_status":"client_not_supported"}',
                '{"semantic_translation_status":"provider_not_supported"}',
                '{"semantic_translation_status":true}',
            )),
        ]
        await self.stream(messages, limit=1)  # Unmodified events keep upstream guards.
        self.assertEqual(self.context.write.await_count, len(messages))
        for message, call in zip(messages, self.context.write.await_args_list):
            self.assertIs(call.args[0], message)
        self.context.abort.assert_not_awaited()

    async def test_inactivity_advertises_flush_not_a_two_second_display_sla(self):
        metadata = install_gateway_metadata(self.module, low_latency=True,
            min_request_interval_s=.25, max_hold_s=2, inactivity_flush_s=2)
        self.assertEqual(metadata["semantic_inactivity_flush_ms"], 2000)
        self.assertNotIn("semantic_latency_target_ms", metadata)
        self.assertNotIn("semantic_latency_scope", metadata)
        await self.stream([self.started()])
        data = json.loads(self.context.write.await_args.args[0].data_json)
        self.assertEqual(data["semantic_incomplete_policy"], "translate_residual_after_inactivity")
        self.assertEqual(data["semantic_inactivity_flush_ms"], 2000)
        diagnostic = json.loads(self.output.getvalue().strip())
        self.assertEqual(diagnostic["semantic_dispatch_policy"], "complete-prefix-or-transcript-inactivity-v1")
        self.assertEqual(diagnostic["semantic_inactivity_flush_ms"], 2000)

    async def test_context_authentication_abort_cancellation_and_metadata_are_delegated(self):
        install_gateway_metadata(self.module)
        service = await self.stream([])
        proxy = service.seen_context
        self.assertIsNot(proxy, self.context)
        self.assertIs(service.seen_auth, self.auth)
        self.context.auth_context.assert_called_once_with()
        for name in ("auth_context", "peer_identities", "peer", "invocation_metadata",
                     "add_done_callback", "time_remaining", "abort"):
            self.assertIs(getattr(proxy, name), getattr(self.context, name))
        callback = object()
        proxy.add_done_callback(callback)
        self.context.add_done_callback.assert_called_once_with(callback)
        with self.assertRaises(RpcAborted):
            await proxy.abort("unauthenticated", "client_certificate_required")
        self.context.abort.assert_awaited_once_with("unauthenticated", "client_certificate_required")
        with self.assertRaises(AttributeError):
            _ = proxy.unknown_attribute

    async def test_rewritten_utf8_payload_at_exact_limit_is_accepted_but_one_byte_over_aborts(self):
        install_gateway_metadata(self.module)
        original = self.started(message="한글" * 30)
        snapshot = original.data_json
        await self.stream([original])
        payload = self.context.write.await_args.args[0].data_json
        byte_count = len(payload.encode("utf-8"))
        self.assertGreater(byte_count, len(payload))
        self.context.write.reset_mock()
        await self.stream([original], limit=byte_count)
        self.context.write.assert_awaited_once()
        self.context.write.reset_mock()
        with self.assertRaises(RpcAborted):
            await self.stream([original], limit=byte_count - 1)
        self.context.write.assert_not_awaited()
        self.context.abort.assert_awaited_once_with("resource_exhausted", "event_payload_too_large")
        self.assertEqual(original.data_json, snapshot)

    async def test_oversize_never_writes_even_if_context_abort_returns(self):
        install_gateway_metadata(self.module)
        self.context.abort.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "abort unexpectedly returned"):
            await self.stream([self.started()], limit=1)
        self.context.write.assert_not_awaited()

    async def test_context_write_failure_propagates_without_mutating_source(self):
        install_gateway_metadata(self.module)
        original = self.started()
        snapshot = original.data_json
        failure = RuntimeError("write failed")
        self.context.write.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            await self.stream([original])
        self.assertIs(caught.exception, failure)
        self.assertEqual(original.data_json, snapshot)

    async def test_install_is_idempotent_returns_fresh_metadata_and_keeps_existing_instances(self):
        original = self.started()
        existing = FakeService([original])
        metadata = install_gateway_metadata(self.module)
        installed = self.module.GatewayService
        metadata["semantic_max_hold_ms"] = -1
        self.assertEqual(install_gateway_metadata(self.module)["semantic_max_hold_ms"], 12000)
        self.assertIs(self.module.GatewayService, installed)
        self.assertEqual(installed.__bases__, (FakeService,))
        self.assertIsNot(installed.__init__, FakeService.__init__)
        self.assertIs(installed._myvote_quality_gateway_original, FakeService)
        await existing.StreamSession(object(), self.context)
        self.assertIs(self.context.write.await_args.args[0], original)
        with self.assertRaisesRegex(ValueError, "different timings"):
            install_gateway_metadata(self.module, max_hold_s=13)
        self.assertIs(self.module.GatewayService, installed)

    async def test_default_drain_budget_follows_source_age_but_explicit_limit_is_preserved(self):
        metadata = install_gateway_metadata(self.module)
        self.assertEqual(metadata["semantic_finish_timeout_ms"], 60000)
        default_service = self.module.GatewayService([self.started()])
        self.assertEqual(default_service.config.finish_timeout_s, 60)
        await default_service.StreamSession(object(), self.context)
        sent = json.loads(self.context.write.await_args.args[0].data_json)
        self.assertEqual(sent["semantic_finish_timeout_ms"], 60000)
        explicit = FakeConfig(finish_timeout_s=7)
        service = self.module.GatewayService([self.started()], config=explicit)
        self.assertIs(service.config, explicit)
        await service.StreamSession(object(), self.context)
        self.assertEqual(json.loads(self.context.write.await_args.args[0].data_json)["semantic_finish_timeout_ms"], 7000)

    async def test_low_latency_target_and_observed_age_are_not_a_display_guarantee(self):
        metadata = install_gateway_metadata(self.module, low_latency=True, min_request_interval_s=.25,
                                            max_hold_s=.75, latency_target_s=2)
        self.assertEqual(metadata["semantic_latency_target_ms"], 2000)
        self.assertEqual(metadata["semantic_min_request_interval_ms"], 250)
        self.assertEqual(metadata["semantic_latency_scope"], "first_source_preview_to_server_translation_event")
        await self.stream([self.started(), ServerMessage(kind="translation.completed", data_json=json.dumps({
            "semantic_first_word_age_ms": 2134.5, "semantic_latency_target_ms": 2000,
            "semantic_latency_exceeded": True, "semantic_latency_anchor_fallback": False,
            "text": "PRIVATE", "semantic_latency_scope": "first_source_preview_to_server_translation_event"}))])
        diagnostics = [json.loads(line) for line in self.output.getvalue().splitlines()]
        self.assertTrue(diagnostics[-1]["semantic_latency_exceeded"])
        self.assertEqual(diagnostics[-1]["semantic_first_word_age_ms"], 2134.5)
        self.assertNotIn("PRIVATE", self.output.getvalue())

    async def test_explicit_timing_values_can_follow_policy_constants(self):
        metadata = install_gateway_metadata(self.module, min_request_interval_s=2,
                                            max_hold_s=15, max_total_age_s=25)
        await self.stream([self.started()])
        sent = json.loads(self.context.write.await_args.args[0].data_json)
        for key, expected in (("semantic_min_request_interval_ms", 2000),
                              ("semantic_max_hold_ms", 15000), ("semantic_total_budget_ms", 25000)):
            self.assertEqual(metadata[key], expected)
            self.assertEqual(sent[key], expected)

    async def test_invalid_timing_parameters_do_not_patch_service(self):
        for value in (True, 0, -1, .0001, 121, float("nan"), float("inf"), "12"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    install_gateway_metadata(self.module, max_hold_s=value)
                self.assertIs(self.module.GatewayService, FakeService)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            install_gateway_metadata(self.module, max_hold_s=46)
        self.assertIs(self.module.GatewayService, FakeService)

    async def test_diagnostics_include_only_successful_selected_events_and_per_proxy_counts(self):
        install_gateway_metadata(self.module)
        messages = [
            self.started(),
            ServerMessage(kind="transcript.preview", data_json='{"text":"private preview"}'),
            ServerMessage(kind="caption.source", data_json='{"text":"private source"}'),
            ServerMessage(kind="translation.preview", data_json='{"text":"private result"}'),
            ServerMessage(kind="translation.completed", data_json='{"semantic_model_ms":25.5}'),
            ServerMessage(kind="translation.failed", data_json='{"error":"semantic_incomplete_source"}'),
            ServerMessage(kind="translation.completed", data_json="{}"),
            ServerMessage(kind="session.completed", data_json="{}"),
        ]
        await self.stream(messages)
        diagnostics = [json.loads(line) for line in self.output.getvalue().splitlines()]
        self.assertEqual([item["event_kind"] for item in diagnostics], [
            "session.started", "translation.completed", "translation.failed",
            "translation.completed", "session.completed",
        ])
        self.assertEqual([(item["completed_count"], item["failed_count"]) for item in diagnostics],
                         [(0, 0), (1, 0), (1, 1), (2, 1), (2, 1)])
        self.assertTrue(all(item["kind"] == "gateway.semantic_diagnostic" for item in diagnostics))
        self.assertEqual(diagnostics[0]["semantic_translation_status"], "enabled")
        self.assertIs(diagnostics[0]["semantic_quality_first"], True)
        self.assertEqual(diagnostics[0]["semantic_incomplete_policy"], "source_only")
        self.assertEqual(diagnostics[0]["semantic_max_hold_ms"], 12000)
        self.assertEqual(diagnostics[1]["semantic_model_ms"], 25.5)
        self.assertEqual(diagnostics[2]["error_category"], "semantic_incomplete_source")
        self.assertNotIn("private", self.output.getvalue())
        await self.stream([ServerMessage(kind="translation.completed", data_json="{}")])
        latest = json.loads(self.output.getvalue().splitlines()[-1])
        self.assertEqual((latest["completed_count"], latest["failed_count"]), (1, 0))

    async def test_diagnostic_whitelists_never_log_text_ids_arbitrary_errors_or_invalid_timings(self):
        install_gateway_metadata(self.module)
        secret = "PRIVATE-SOURCE-RESULT-AND-ID"
        payload = {
            "text": secret, "source_text": secret, "translation": secret,
            "session_id": secret, "segment_id": secret, "request_id": secret,
            "error": {"message": secret}, "usage": {"prompt": secret},
            "semantic_selection_ms": 12, "semantic_buffer_wait_ms": 1.25,
            "semantic_model_ms": float("inf"), "semantic_translation_ms": float("nan"),
            "semantic_total_ms": True, "provider_elapsed_ms": secret,
            "queue_ms": -5, "semantic_total_budget_ms": 10 ** 400,
            "unapproved_timing_ms": 123,
        }
        original = ServerMessage(kind="translation.failed", data_json=json.dumps(payload),
                                 session_id=secret, segment_id=secret, server_elapsed_ms=float("nan"))
        snapshot = original.data_json
        await self.stream([original])
        diagnostic = json.loads(self.output.getvalue())
        self.assertEqual(diagnostic, {
            "kind": "gateway.semantic_diagnostic", "event_kind": "translation.failed",
            "completed_count": 0, "failed_count": 1, "error_category": "other_engine_error",
            "semantic_selection_ms": 12, "semantic_buffer_wait_ms": 1.25,
        })
        self.assertNotIn(secret, self.output.getvalue())
        self.assertEqual(original.data_json, snapshot)

    async def test_diagnostic_error_categories_are_exact_matches_not_arbitrary_prefixes(self):
        install_gateway_metadata(self.module)
        errors = ("semantic_model_timeout", "semantic_source_limit", "semantic_buffer_capacity",
                  "semantic_model_timeout PRIVATE-TEXT", "semantic_PRIVATE-TEXT", None)
        await self.stream([ServerMessage(kind="translation.failed", data_json=json.dumps({"error": error}))
                           for error in errors])
        diagnostics = [json.loads(line) for line in self.output.getvalue().splitlines()]
        self.assertEqual([item["error_category"] for item in diagnostics], [
            "semantic_model_timeout", "semantic_source_limit", "semantic_buffer_capacity",
            "other_engine_error", "other_engine_error", "other_engine_error",
        ])
        self.assertNotIn("PRIVATE-TEXT", self.output.getvalue())

    async def test_diagnostic_waits_for_successful_write_and_preserves_return_value(self):
        install_gateway_metadata(self.module)
        service = await self.stream([])
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_write(message):
            entered.set()
            await release.wait()
            return "underlying-write-result"

        self.context.write.side_effect = delayed_write
        task = asyncio.create_task(service.seen_context.write(
            ServerMessage(kind="translation.completed", data_json="{}")))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(self.output.getvalue(), "")
        finally:
            release.set()
        self.assertEqual(await asyncio.wait_for(task, 1), "underlying-write-result")
        self.assertEqual(json.loads(self.output.getvalue())["completed_count"], 1)

    async def test_failed_or_cancelled_write_has_no_success_diagnostic_or_counter_increment(self):
        install_gateway_metadata(self.module)
        for failure in (RuntimeError("write failed"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                self.output.seek(0)
                self.output.truncate()
                service = await self.stream([])
                self.context.write.side_effect = failure
                message = ServerMessage(kind="translation.completed", data_json="{}")
                with self.assertRaises(type(failure)) as caught:
                    await service.seen_context.write(message)
                self.assertIs(caught.exception, failure)
                self.assertEqual(self.output.getvalue(), "")
                self.context.write.side_effect = None
                await service.seen_context.write(message)
                self.assertEqual(json.loads(self.output.getvalue())["completed_count"], 1)

    async def test_diagnostic_sink_failure_does_not_fail_an_already_delivered_write(self):
        install_gateway_metadata(self.module)
        service = await self.stream([])
        with patch("builtins.print", side_effect=BrokenPipeError) as emit:
            result = await service.seen_context.write(
                ServerMessage(kind="translation.completed", data_json="{}"))
        self.assertEqual(result, "written")
        self.context.write.assert_awaited_once()
        emit.assert_called_once()
        self.assertIs(emit.call_args.kwargs["flush"], True)
        self.assertEqual(json.loads(emit.call_args.args[0])["event_kind"], "translation.completed")

    async def test_started_diagnostic_never_copies_unrecognized_policy_strings(self):
        install_gateway_metadata(self.module)
        await self.stream([self.started(semantic_translation_status="PRIVATE-STATUS",
                                       semantic_quality_first="PRIVATE-POLICY",
                                       semantic_incomplete_policy="PRIVATE-INCOMPLETE")])
        diagnostic = json.loads(self.output.getvalue())
        self.assertEqual(diagnostic["semantic_translation_status"], "unknown")
        self.assertNotIn("semantic_quality_first", diagnostic)
        self.assertNotIn("semantic_incomplete_policy", diagnostic)
        self.assertNotIn("PRIVATE", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
