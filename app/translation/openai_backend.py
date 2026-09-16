"""Bounded, reusable async transport for generic OpenAI-compatible chat APIs."""

import asyncio
import json
import math
import time

import httpx

from .backend import TranslationError, TranslationResult
from .prompt import build_translation_messages


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response field")
        result[key] = value
    return result


class OpenAITranslationBackend:
    def __init__(self, base_url, default_model="google/gemma-4-26b-a4b", *, client=None,
                 api_key=None, timeout_s=30, max_input_chars=24000, max_output_chars=16000,
                 max_request_bytes=262144, max_response_bytes=262144,
                 max_tokens=2048, temperature=0):
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("Expected an HTTP(S) API base URL without credentials, query or fragment")
        path = url.path.rstrip("/")
        self.endpoint = str(url.copy_with(path=path + ("/chat/completions" if path.endswith("/v1")
                                                     else "/v1/chat/completions")))
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
            raise ValueError("timeout_s must be positive and at most 300 seconds")
        for value in (max_input_chars, max_output_chars, max_request_bytes, max_response_bytes, max_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("Translation capacity limits must be positive integers")
        self._validate_model(default_model)
        self._validate_temperature(temperature)
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()
                                    or any(character in api_key for character in "\r\n")):
            raise ValueError("Invalid API key")
        self.default_model, self.temperature = default_model, temperature
        self.timeout_s = timeout_s
        self.timeout = httpx.Timeout(timeout_s, connect=min(5, timeout_s))
        self.max_input_chars, self.max_output_chars = max_input_chars, max_output_chars
        self.max_request_bytes, self.max_response_bytes = max_request_bytes, max_response_bytes
        self.max_tokens = max_tokens
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = "Bearer " + api_key
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(
            trust_env=False, timeout=self.timeout, limits=httpx.Limits(max_connections=16,
                                                                       max_keepalive_connections=8))
        self._closed = False

    @staticmethod
    def _validate_model(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
            raise TranslationError("invalid_model")

    @staticmethod
    def _validate_temperature(value):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2:
            raise TranslationError("invalid_temperature")

    async def translate(self, source, source_language, target_language, source_context=(),
                        translation_context=(), glossary=None, *, model=None, temperature=None):
        if self._closed:
            raise TranslationError("backend_closed")
        model = self.default_model if model is None else model
        temperature = self.temperature if temperature is None else temperature
        self._validate_model(model)
        self._validate_temperature(temperature)
        messages = build_translation_messages(
            source, source_language, target_language, source_context, translation_context, glossary,
            max_input_chars=self.max_input_chars)
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": self.max_tokens, "stream": False}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(body) > self.max_request_bytes:
            raise TranslationError("input_too_large")
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout_s), self._client.stream(
                    "POST", self.endpoint, content=body, headers=self._headers, timeout=self.timeout) as response:
                if response.status_code != 200:
                    code = "rate_limited" if response.status_code == 429 else "upstream_http_error"
                    raise TranslationError(code, retryable=response.status_code in (408, 429)
                                           or response.status_code >= 500)
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise TranslationError("response_too_large")
                    chunks.append(chunk)
        except (httpx.TimeoutException, TimeoutError):
            raise TranslationError("timeout", retryable=True) from None
        except httpx.HTTPError:
            raise TranslationError("transport_error", retryable=True) from None
        try:
            packet = json.loads(b"".join(chunks).decode("utf-8"), object_pairs_hook=_unique_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            choices = packet["choices"]
            if not isinstance(choices, list) or len(choices) != 1 or type(choices[0]) is not dict:
                raise ValueError("invalid choices")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise TranslationError("incomplete_response")
            message = choice["message"]
            if type(message) is not dict or message.get("tool_calls") or message.get("refusal"):
                raise ValueError("invalid assistant response")
            result = message["content"]
            if not isinstance(result, str) or not result.strip():
                raise ValueError("missing translation")
            result.encode("utf-8")
        except TranslationError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, RecursionError):
            raise TranslationError("invalid_response") from None
        if len(result) > self.max_output_chars:
            raise TranslationError("output_too_large")
        return TranslationResult(result.strip(), (time.monotonic() - started) * 1000)

    async def close(self):
        if not self._closed:
            self._closed = True
            if self._owns_client:
                await self._client.aclose()

    async def aclose(self):
        await self.close()
