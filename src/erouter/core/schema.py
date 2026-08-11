"""JSON serialisation of a route (spec §15).

Token amounts are always **strings of integers in native units**; the `*_human`
fields are decorative.  `certificate` is non-optional and carries a non-null
`certificate_reason` whenever it is false -- §15's contract is that it "must be
surfaced, not swallowed", and pairing the two enforces that at the schema level
rather than by convention.
"""

from __future__ import annotations

from typing import Any

from .pipeline import RouteResult
from .rendermodel import format_units


def to_json(
    result: RouteResult,
    *,
    chain: str = "",
    chain_id: int = 0,
    block: int = 0,
    candidates: list[dict] | None = None,
    verified_out: int | None = None,
) -> dict[str, Any]:
    route = result.route
    nodes = result.nodes
    if route is None or nodes is None:
        raise ValueError("cannot serialise a route that was not produced")

    src_dec = nodes.decimals(result.src_token)
    dst_dec = nodes.decimals(result.dst_token)
    total_bp = result.fee_bp + result.impact_bp

    payload: dict[str, Any] = {
        "version": 1,
        "chain": {"name": chain, "chain_id": chain_id, "block": block},
        "request": {
            "src": result.src_token,
            "dst": result.dst_token,
            "src_symbol": nodes.symbol(result.src_token),
            "dst_symbol": nodes.symbol(result.dst_token),
            "amount_in": str(result.amount_in),
            "amount_in_human": format_units(result.amount_in, src_dec),
        },
        "result": {
            "amount_out": str(route.modelled_out),
            "amount_out_human": format_units(route.modelled_out, dst_dec),
            "verified_out": None if verified_out is None else str(verified_out),
            "verified": verified_out is not None,
            "certificate": result.certificate,
            "certificate_reason": result.certificate_reason,
            "loss_bp": {
                "total": round(total_bp, 4),
                "fee": round(result.fee_bp, 4),
                "impact": round(result.impact_bp, 4),
            },
        },
        "legs": [],
        "nodes": [],
        "paths": route.paths,
        "candidates": candidates or [],
        "diagnostics": {
            **dict(result.counters),
            "timings_ms": {k: round(v, 2) for k, v in result.timings.items()},
        },
        "warnings": result.warnings,
    }

    by_id = {arc.id: arc for arc in result.arcs}
    for index, realized in enumerate(route.legs):
        arc = by_id.get(realized.arc_id or "")
        entry: dict[str, Any] = {
            "index": index,
            "kind": realized.kind.name,
            "target": realized.target,
            "token_in": realized.token_in,
            "token_out": realized.token_out,
            "symbol_in": nodes.symbol(realized.token_in),
            "symbol_out": nodes.symbol(realized.token_out),
            "amount_in": str(realized.amount_in),
            "amount_out": str(realized.amount_out),
            "amount_in_human": format_units(
                realized.amount_in, nodes.decimals(realized.token_in)
            ),
            "amount_out_human": format_units(
                realized.amount_out, nodes.decimals(realized.token_out)
            ),
            "src_slot": realized.leg.src_slot,
            "dst_slot": realized.leg.dst_slot,
            "bps": realized.leg.bps,
            "share_of_node": round(realized.share_of_node, 6),
            "is_conversion": realized.is_conversion,
        }
        if arc is not None:
            entry["pool_name"] = arc.note
            entry["arc_id"] = arc.id
            entry["i"], entry["j"] = arc.i, arc.j
            entry["arc"] = {
                "a": arc.a,
                "B": arc.B,
                "G": arc.G,
                "R": arc.resistance,
                "eps": arc.eps,
                "cap": None if arc.cap == float("inf") else arc.cap,
                "theta": realized.theta,
                "drift": arc.drift,
                "eta": None if arc.eta != arc.eta else arc.eta,  # NaN -> null
                "convex_flag": arc.convex_flag,
                "clamped": arc.clamped,
                "flag_reason": arc.flag_reason.value,
                "calib_delta": arc.calib_delta,
                "tvl_usd": arc.tvl_usd,
            }
        payload["legs"].append(entry)

    for token, slot in sorted(route.slots.items(), key=lambda kv: kv[1]):
        node = route.node_of_slot.get(slot, 0)
        payload["nodes"].append(
            {
                "slot": slot,
                "node": node,
                "token": token,
                "symbol": nodes.symbol(token),
                "u": route.potentials.get(node),
                "merged": nodes.tokens_of[node],
            }
        )
    return payload
