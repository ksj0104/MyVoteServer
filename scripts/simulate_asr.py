"""Send synthetic cumulative ASR partials, then await a matching final snapshot.

No audio is captured and no legacy 50051 server is contacted. Use --show-text
only when displaying translated text on this terminal is intended.
"""

import argparse
import asyncio
import uuid

from dotenv import load_dotenv

if __package__:
    from .client import StreamingClient, add_connection_options, positive_number, run_cli
else:
    from client import StreamingClient, add_connection_options, positive_number, run_cli


def cumulative_partials(text):
    """Keep original spacing while revealing one whitespace-delimited word at a time."""
    import re
    return [text[:match.end()] for match in re.finditer(r"\S+", text)]


async def simulate(args):
    partials = cumulative_partials(args.text)
    if not partials:
        raise ValueError("Text must contain at least one word")
    if args.disconnect_after and (args.reconnects < 1 or args.disconnect_after > len(partials)):
        raise ValueError("Disconnect position needs a reconnect attempt and an existing partial")
    async with StreamingClient(args) as client:
        utterance = "sim-" + uuid.uuid4().hex
        for number, partial in enumerate(partials, start=1):
            await client.send_asr(partial, utterance)
            if args.disconnect_after == number:
                await client.disconnect_and_recover()
            await asyncio.sleep(args.interval / 1000)
        sequence = await client.send_asr(args.text, utterance, final=True)
        await client.wait_final(sequence, utterance)
        print("simulation completed: matching final snapshot received (not an audio/E2E quality test)")


def main():
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_options(parser)
    parser.add_argument("--interval", type=positive_number, default=150,
                        help="milliseconds between cumulative ASR partials (default: 150)")
    parser.add_argument("--text", default="The shipment weighs about twenty kilograms.")
    parser.add_argument("--disconnect-after", type=int, default=0,
                        help="explicitly disconnect after this partial and recover from a snapshot")
    args = parser.parse_args()
    if args.disconnect_after < 0:
        parser.error("--disconnect-after must be zero or positive")
    return run_cli(simulate, args)


if __name__ == "__main__":
    raise SystemExit(main())
