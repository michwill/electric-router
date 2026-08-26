"""Freeze `_recalibrate`'s inputs and its answers, for a differential.

A ported model has a natural acceptance test -- it is wei-exact or it is not.
A ported *stage* does not: it mutates ladders, re-fits arcs and reassembles a
graph, and "did the Rust do the same thing" is a question about an object
graph rather than a number. So the stage is given one, by recording what
Python was asked and what it answered, field by field, and holding the port to
both.

This is the first of those: every arc `_recalibrate` re-fits, with the ladder
it read and the calibration it produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_quote import USDC, WETH, warm

from erouter.core import pipeline


def bits(value: float) -> str:
    """A float as its exact bit pattern, in a string.

    Decimal text does not survive the trip. `repr` round-trips in Python and
    `1542159.0319370825` came back one ULP out through serde_json -- which
    would not matter, except that `calibrate` runs divided differences and
    they cancel: one ULP in the delta moved `drift` by 2.7e-7 relative, an
    amplification of about a billion. A differential that cannot tell a real
    disagreement from its own transport is not a differential.
    """
    return str(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--from", dest="src", default=USDC)
    p.add_argument("--to", dest="dst", default=WETH)
    p.add_argument("--amount", type=int, default=100_000 * 10**6)
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--out", default="/tmp/recalibrate.json")
    args = p.parse_args()

    session = warm(args)

    rows: list[dict] = []
    original = pipeline._recalibrate

    def recording(arcs, ladders, nodes):
        from erouter.core.calibrate import CalibrationError, calibrate
        by_id = {lad.arc.id: lad for lad in ladders}
        for arc in arcs:
            ladder = by_id.get(arc.id)
            if ladder is None or len(ladder.deltas) < 3:
                continue
            deltas, quotes = ladder.as_float()
            quantum = pipeline._quantum(ladder.arc.decimals_out)
            try:
                fit = calibrate(deltas, quotes, quantum=quantum)
            except CalibrationError as exc:
                rows.append({"deltas": [bits(v) for v in deltas],
                             "quotes": [bits(v) for v in quotes],
                             "quantum": bits(quantum), "error": str(exc)[:60]})
                continue
            rows.append({
                "deltas": [bits(v) for v in deltas],
                "quotes": [bits(v) for v in quotes],
                "quantum": bits(quantum),
                "rate_in": bits(nodes.rate(arc.token_in)),
                "rate_out": bits(nodes.rate(arc.token_out)),
                "cap_before": bits(arc.cap),
                "fit": {
                    "a": bits(fit.a), "B": bits(fit.B), "cap": bits(fit.cap),
                    "clamped": bool(fit.clamped),
                    "convex_flag": bool(fit.convex_flag),
                    "flag_reason": str(fit.flag_reason.value),
                    "drift": bits(fit.drift),
                    "eta": bits(fit.eta),
                    "calib_delta": bits(fit.calib_delta),
                },
            })
        return original(arcs, ladders, nodes)

    pipeline._recalibrate = recording
    try:
        asyncio.run(session.set_pair(args.src, args.dst))
        session.quote(args.amount)
        rows.clear()
        session.quote(args.amount)
    finally:
        pipeline._recalibrate = original

    Path(args.out).write_text(json.dumps(rows))
    failed = sum(1 for r in rows if "error" in r)
    print(f"wrote {args.out}  ·  {len(rows)} arcs re-fitted "
          f"({failed} of which Python refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
