"""Serialize one provider's live and correction requests, preferring live work.

An already running backend request is not preempted. This admission rule limits
additional contention; physical inference latency still needs hardware testing.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class TranslationAdmission:
    def __init__(self, primary_pending=lambda: False, *, max_primary=1):
        if type(max_primary) is not int or not 1 <= max_primary <= 4:
            raise ValueError("Primary translation capacity must be 1..4")
        self.primary_pending = primary_pending
        self.max_primary = max_primary
        self._busy = 0
        self._secondary_busy = False
        self._primary_waiters = 0
        self._changed = asyncio.Event()
        self.secondary_calls = 0
        self.primary_calls = 0

    @asynccontextmanager
    async def acquire(self, *, secondary=False):
        acquired = False
        if not secondary:
            self._primary_waiters += 1
        try:
            while (self._secondary_busy or self._busy >= self.max_primary
                   or (secondary and (self._busy or self._primary_waiters or self.primary_pending()))):
                self._changed.clear()
                # The runner's completed-task callback may release the last
                # pending primary after the provider generator has returned.
                try:
                    await asyncio.wait_for(self._changed.wait(), .05)
                except TimeoutError:
                    pass
            acquired = True
            self._busy += 1
            self._secondary_busy = secondary
            if secondary:
                self.secondary_calls += 1
            else:
                self.primary_calls += 1
            yield
        finally:
            if not secondary:
                self._primary_waiters -= 1
            if acquired:
                self._busy -= 1
                if secondary:
                    self._secondary_busy = False
            self._changed.set()


class AdmittedTranslationProvider:
    def __init__(self, provider, admission: TranslationAdmission, *, secondary=False):
        self.provider, self.admission, self.secondary = provider, admission, secondary

    async def stream(self, request):
        async with self.admission.acquire(secondary=self.secondary):
            stream = self.provider.stream(request)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                closer = getattr(stream, "aclose", None)
                if closer is not None:
                    await closer()

    async def semantic_stream(self, request):
        # Boundary selection is part of the first live result, even when a
        # caller holds a wrapper otherwise configured for secondary corrections.
        async with self.admission.acquire(secondary=False):
            stream = self.provider.semantic_stream(request)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                closer = getattr(stream, "aclose", None)
                if closer is not None:
                    await closer()
