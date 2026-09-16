"""Advertise opt-in quality timing without editing the verified gateway.

Only enabled semantic session.started payloads are copied and amended. The
default drain allowance follows the larger quality budget; explicit caller
configuration is retained. Authentication and session ownership stay upstream.
Successful delivery diagnostics contain closed-vocabulary metadata and timings,
never source/result text, request IDs or arbitrary engine error messages.
"""

import json
import math
from dataclasses import replace


_DIAGNOSTIC_EVENTS = frozenset({
    "session.started", "translation.completed", "translation.failed", "session.completed",
})
_SAFE_ERROR_CATEGORIES = frozenset({
    "semantic_incomplete_source", "semantic_model_timeout", "semantic_source_limit",
    "semantic_buffer_capacity", "semantic_source_too_large", "semantic_source_deadline",
    "semantic_invalid_model_response", "semantic_provider_error", "semantic_unsupported_language",
    "semantic_invalid_source", "semantic_unsupported_profile", "semantic_reset",
})
_SAFE_TIMINGS = (
    "semantic_min_request_interval_ms", "semantic_max_hold_ms", "semantic_total_budget_ms",
    "semantic_request_timeout_ms", "semantic_orchestrator_budget_ms",
    "semantic_translation_budget_ms",
    "semantic_finish_timeout_ms",
    "semantic_first_word_age_ms", "semantic_latency_target_ms", "semantic_inactivity_flush_ms",
    "semantic_buffer_wait_ms", "semantic_model_ms", "semantic_total_ms",
    "semantic_selection_ms", "semantic_translation_ms", "provider_elapsed_ms", "queue_ms",
)


def _finite_timing(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _event_data(message):
    try:
        data = json.loads(message.data_json)
    except (TypeError, ValueError, RecursionError):
        return {}
    return data if type(data) is dict else {}


def install_gateway_metadata(gateway_module, *, min_request_interval_s=1.5,
                             max_hold_s=12, max_total_age_s=45,
                             request_timeout_s=20, selection_timeout_s=5,
                             translation_timeout_s=15, low_latency=False, latency_target_s=2,
                             inactivity_flush_s=0):
    """Install once; callers may pass the corresponding quality-policy constants."""
    values = (min_request_interval_s, max_hold_s, max_total_age_s,
              request_timeout_s, selection_timeout_s, translation_timeout_s, latency_target_s)
    if type(low_latency) is not bool:
        raise ValueError("low_latency must be a bool")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not .001 <= value <= 120 for value in values):
        raise ValueError("Quality gateway timings must be finite positive milliseconds")
    if max_hold_s > max_total_age_s:
        raise ValueError("Quality hold must not exceed the total source budget")
    if (type(inactivity_flush_s) not in (int, float) or not math.isfinite(inactivity_flush_s)
            or not 0 <= inactivity_flush_s < max_total_age_s):
        raise ValueError("Inactivity flush must be zero or shorter than source lifetime")
    if (request_timeout_s > max_total_age_s
            or selection_timeout_s + translation_timeout_s > request_timeout_s):
        raise ValueError("Quality stage budgets must fit request and source budgets")
    updates = {
        "semantic_min_request_interval_ms": int(min_request_interval_s * 1000),
        "semantic_max_hold_ms": int(max_hold_s * 1000),
        "semantic_total_budget_ms": int(max_total_age_s * 1000),
        "semantic_request_timeout_ms": int(request_timeout_s * 1000),
        "semantic_orchestrator_budget_ms": int(selection_timeout_s * 1000),
        "semantic_translation_budget_ms": int(translation_timeout_s * 1000),
        "semantic_quality_first": True,
        "semantic_incomplete_policy": "source_only",
    }
    default_finish_s = max(gateway_module.GatewayConfig().finish_timeout_s,
                           max_total_age_s + translation_timeout_s)
    updates["semantic_finish_timeout_ms"] = int(default_finish_s * 1000)
    if low_latency:
        updates["semantic_low_latency"] = True
    if inactivity_flush_s:
        updates.update(semantic_inactivity_flush_ms=int(inactivity_flush_s * 1000),
                       semantic_incomplete_policy="translate_residual_after_inactivity",
                       semantic_dispatch_policy="complete-prefix-or-transcript-inactivity-v1")
    elif low_latency:
        updates.update(semantic_latency_target_ms=int(latency_target_s * 1000),
                       semantic_latency_scope="first_source_preview_to_server_translation_event")
    signature = tuple(updates.items())
    original = gateway_module.GatewayService
    if getattr(original, "_myvote_quality_gateway_v1", False):
        if original._myvote_quality_gateway_values != signature:
            raise ValueError("Quality gateway metadata is already installed with different timings")
        return {"semantic_quality_gateway": "complete-thought-metadata-v1", **updates}

    class ContextProxy:
        def __init__(self, context, max_event_bytes, finish_timeout_s):
            self._context = context
            self._max_event_bytes = max_event_bytes
            self._completed_count = 0
            self._failed_count = 0
            self._finish_timeout_s = finish_timeout_s

        def __getattr__(self, name):
            return getattr(self._context, name)

        async def _write(self, message):
            result = await self._context.write(message)
            # Failed/cancelled writes never advance counters or claim delivery.
            if message.kind not in _DIAGNOSTIC_EVENTS:
                return result
            self._completed_count += message.kind == "translation.completed"
            self._failed_count += message.kind == "translation.failed"
            diagnostic = {
                "kind": "gateway.semantic_diagnostic", "event_kind": message.kind,
                "completed_count": self._completed_count, "failed_count": self._failed_count,
            }
            data = _event_data(message)
            if message.kind == "session.started":
                status = data.get("semantic_translation_status")
                diagnostic["semantic_translation_status"] = (
                    status if type(status) is str and status in (
                        "enabled", "client_not_supported", "provider_not_supported") else "unknown")
                if type(data.get("semantic_quality_first")) is bool:
                    diagnostic["semantic_quality_first"] = data["semantic_quality_first"]
                if data.get("semantic_incomplete_policy") in ("source_only", "translate_residual_after_inactivity"):
                    diagnostic["semantic_incomplete_policy"] = data["semantic_incomplete_policy"]
                if data.get("semantic_dispatch_policy") == "complete-prefix-or-transcript-inactivity-v1":
                    diagnostic["semantic_dispatch_policy"] = data["semantic_dispatch_policy"]
            if message.kind == "translation.failed":
                error = data.get("error")
                diagnostic["error_category"] = (
                    error if type(error) is str and error in _SAFE_ERROR_CATEGORIES else "other_engine_error")
            for key in _SAFE_TIMINGS:
                value = data.get(key)
                if _finite_timing(value):
                    diagnostic[key] = value
            for key in ("semantic_low_latency", "semantic_latency_exceeded", "semantic_latency_anchor_fallback"):
                if type(data.get(key)) is bool:
                    diagnostic[key] = data[key]
            if data.get("semantic_latency_scope") == "first_source_preview_to_server_translation_event":
                diagnostic["semantic_latency_scope"] = data["semantic_latency_scope"]
            elapsed = getattr(message, "server_elapsed_ms", None)
            if _finite_timing(elapsed):
                diagnostic["server_elapsed_ms"] = elapsed
            try:
                print(json.dumps(diagnostic, allow_nan=False, separators=(",", ":")), flush=True)
            except (OSError, ValueError):
                # Losing a diagnostic sink must not change an already delivered
                # RPC result. Cancellation and underlying write errors are not caught.
                pass
            return result

        async def write(self, message):
            if message.kind != "session.started":
                return await self._write(message)
            try:
                data = json.loads(message.data_json)
            except (TypeError, ValueError):
                # Invalid/unrelated payloads remain the original service's
                # responsibility; the proxy does not reinterpret other events.
                return await self._write(message)
            if type(data) is not dict or data.get("semantic_translation_status") != "enabled":
                return await self._write(message)
            data.update(updates)
            # Explicit caller limits remain authoritative and must be reported
            # honestly rather than advertising the extended launcher default.
            data["semantic_finish_timeout_ms"] = int(self._finish_timeout_s * 1000)
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > self._max_event_bytes:
                # The original size guard ran before these fields were added.
                # Keep its resource-exhaustion status and never send oversize.
                await self._context.abort(gateway_module.grpc.StatusCode.RESOURCE_EXHAUSTED,
                                          "event_payload_too_large")
                raise RuntimeError("RPC abort unexpectedly returned")
            copied = gateway_module.pb.ServerMessage()
            copied.CopyFrom(message)
            copied.data_json = encoded
            return await self._write(copied)

    class QualityMetadataGateway(original):
        _myvote_quality_gateway_v1 = True
        _myvote_quality_gateway_original = original
        _myvote_quality_gateway_values = signature

        def __init__(self, engine_factory, *, config=None):
            if config is None:
                config = replace(gateway_module.GatewayConfig(), finish_timeout_s=default_finish_s)
            super().__init__(engine_factory, config=config)

        async def StreamSession(self, request_iterator, context):
            return await super().StreamSession(
                request_iterator, ContextProxy(context, self.config.max_event_bytes, self.config.finish_timeout_s))

    gateway_module.GatewayService = QualityMetadataGateway
    return {"semantic_quality_gateway": "complete-thought-metadata-v1", **updates}
