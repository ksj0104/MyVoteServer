"""Measure text-to-translation stages on the machine running this command.

This is NOT an audio-to-render latency benchmark. Reports keep failures in the
denominator and contain source/translation text from the supplied research set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import httpx

from .translation import (
    LlamaCppProvider, OllamaProvider, LMStudioProvider, LatestTranslationRunner,
    TranslationEvent, TranslationRequest,
)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]


def summarize(rows: list[dict]) -> dict:
    measured = [row for row in rows if not row["warmup"]]
    success = [row for row in measured if row["status"] == "completed"]
    complete_times = [row["elapsed_ms"] for row in success]
    first_times = [row["first_content_ms"] for row in measured
                   if row["first_content_ms"] is not None]
    return {
        "measurement_scope": "text_submission_to_provider_result; excludes audio, ASR and UI",
        "requests": len(measured), "completed": len(success),
        "failed": len(measured) - len(success),
        "completion_rate": len(success) / len(measured) if measured else None,
        "completed_only_p50_ms": percentile(complete_times, 0.50),
        "completed_only_p95_ms": percentile(complete_times, 0.95),
        "first_content_p95_ms": percentile(first_times, 0.95),
        "first_content_sample_count": len(first_times),
        "failed_statuses": {name: sum(row["status"] == name for row in measured)
                            for name in sorted({row["status"] for row in measured}
                                               - {"completed"})},
    }


async def benchmark(args: argparse.Namespace) -> dict:
    cases = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if not cases:
        raise ValueError("Input has no JSONL cases")
    # Validate before touching the report or contacting an inference server.
    for index, case in enumerate(cases):
        TranslationRequest(str(index), 1, case["text"], case["source_language"],
                           case.get("target_language", "ko"),
                           tuple(case.get("context", [])), args.deadline_ms)
    rows: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        # The runner owns the total deadline, including cold model load. A short
        # shared HTTP read timeout would incorrectly truncate the warm-up budget.
        timeout=httpx.Timeout(None, connect=3),
        follow_redirects=False, trust_env=False,
    ) as client:
        provider_type = {"llamacpp": LlamaCppProvider, "ollama": OllamaProvider, "lmstudio": LMStudioProvider}[args.backend]
        provider = provider_type(client, args.endpoint, args.model,
                                 max_tokens=args.max_tokens)
        with args.output.open("w", encoding="utf-8") as output:
            for iteration in range(args.warmup + args.repetitions):
                warmup = iteration < args.warmup
                selected = cases[:1] if warmup else cases
                for index, case in enumerate(selected):
                    events: list[TranslationEvent] = []

                    async def collect(event: TranslationEvent) -> None:
                        events.append(event)

                    runner = LatestTranslationRunner(provider, collect)
                    request = TranslationRequest(
                        f"{iteration}-{index}", 1, case["text"], case["source_language"],
                        case.get("target_language", "ko"), tuple(case.get("context", [])),
                        args.cold_deadline_ms if warmup else args.deadline_ms,
                    )
                    try:
                        await runner.submit(request)
                    finally:
                        await runner.close()
                    terminal = next((e for e in reversed(events)
                                     if e.kind in ("completed", "failed")), None)
                    first = next((e for e in events if e.kind == "preview" and e.text.strip()), None)
                    row = {
                        "case_id": case.get("id", str(index)), "iteration": iteration,
                        "warmup": warmup, "backend": args.backend, "model": args.model,
                        "status": (terminal.error or terminal.kind) if terminal else "missing_terminal",
                        "first_content_ms": first.elapsed_ms if first else None,
                        "elapsed_ms": terminal.elapsed_ms if terminal else None,
                        "queue_ms": terminal.queue_ms if terminal else None,
                        "source_text": request.text, "translation": terminal.text if terminal else "",
                        "usage": terminal.usage if terminal else {},
                        "events": [asdict(event) for event in events],
                    }
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{'warmup' if warmup else 'sample'} {row['case_id']}: {row['status']}")
    summary = summarize(rows)
    summary.update(
        backend=args.backend, model=args.model, endpoint=args.endpoint,
        platform=platform.platform(), machine=platform.machine(),
        python=platform.python_version(), budget_ms=args.deadline_ms,
        max_output_tokens=args.max_tokens, warmup_attempts=args.warmup,
        source_file=str(args.input), raw_report=str(args.output),
        limitations=["No audio capture, ASR, speaker model or UI included",
                     "First content is the first non-whitespace text, not necessarily a readable clause",
                     "Record model digest, server version, device, cache and memory separately",
                     "A preview may later fail; it is not a successful translation",
                     "Latency does not establish translation quality"],
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("llamacpp", "ollama", "lmstudio"), default="llamacpp")
    parser.add_argument("--endpoint", help="Server origin; default port 1234 for LM Studio, otherwise 8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/translation.jsonl"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--deadline-ms", type=float, default=2000)
    parser.add_argument("--cold-deadline-ms", type=float, default=60000)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.endpoint is None:
        args.endpoint = "http://127.0.0.1:1234" if args.backend == "lmstudio" else "http://127.0.0.1:8080"
    if args.repetitions < 1 or args.warmup < 0 or args.max_tokens < 1:
        parser.error("repetitions/max-tokens must be positive; warmup must be nonnegative")
    if not all(0 < budget <= 120000 for budget in (args.deadline_ms, args.cold_deadline_ms)):
        parser.error("deadlines must be in (0, 120000] ms")
    try:
        summary = asyncio.run(benchmark(args))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Benchmark setup failed: {exc}\n")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
