"""Give a token a node when only a venue can reach it.

`X -> USDT -> crvUSD` is a perfectly ordinary route and the graph has always
been able to represent it.  What stopped it was that `X` had no node: the node
map is built from Curve's own pools, so a token Curve has never held is not
routable at all, however deep its Uniswap pair.

That was never a topology problem.  §4 fits `nu` from the arcs it is given, so
a node no frame arc reaches keeps the default 1.0 and an arc into it reads as
free money -- measured at `eps = -5.4e7` on a `WETH -> RSR` tick arc, which
emptied the relaxation into it and cost `FRAX -> WBTC` 4,805 bp.  Refusing such
tokens was the cheap fix.  The real one is to price them, which
`pipeline.admissible_late_arcs` now does from the venue arcs themselves.

So the remaining question is which tokens to admit, and that is what this
module answers.  On ethereum 6,379 tokens sit one bridge from routable, holding
$617.8M against a priced coin -- 88% of the v2 census's liquidity.  Admitting
all of them is 12,832 arcs, well past where this graph stops converging, and a
token priced by a single pool is a token routed at whatever that pool asserts.

So they are admitted **one at a time, by name**: the endpoints of the trade
actually being quoted.  A token nobody asked about does not belong in the
graph, and as an intermediate it would have to be worth two venue fees and the
impact on both sides to beat a coin Curve already prices -- which the deep,
well-connected tokens that could manage it already are.
"""

from __future__ import annotations

from ..core.codec import decode_uint, encode_call

#: Never introduce a token whose bridge pair is thinner than this.  A pool that
#: holds nothing prices nothing, and its price is the only one the token gets.
MIN_BRIDGE_USD = 10_000.0


def bridges_for(token: str, nodes, venues, min_usd: float = MIN_BRIDGE_USD):
    """`(venue, pool, tvl)` for every census pair joining `token` to a priced coin.

    The other side has to be priced *by the frame*, not merely present: this is
    what makes one Gauss-Seidel step enough to price `token` in
    `admissible_late_arcs`, and what stops a chain of unpriced tokens.
    """
    token = token.lower()
    out = []
    for venue in venues:
        census = getattr(venue, "census", None) or {}
        floor = max(min_usd, float(getattr(venue, "floor_usd", 0.0) or 0.0))
        for pool, row in census.items():
            token0, token1 = row[0].lower(), row[1].lower()
            if token not in (token0, token1):
                continue
            if not nodes.has(token1 if token == token0 else token0):
                continue
            tvl = float(row[3]) if len(row) > 3 else 0.0
            if tvl < floor:
                continue
            out.append((venue, pool.lower(), tvl))
    out.sort(key=lambda r: -r[2])
    return out


def token_meta(transport, token: str, block: int) -> tuple[str, int] | None:
    """`(symbol, decimals)`, or `None` if the address will not answer.

    Decimals are load-bearing -- every amount in and out of the pair is scaled
    by them -- so a token that will not say is not introduced.  The symbol is
    cosmetic and a short address stands in: plenty of tokens predate the string
    return and answer `symbol()` in `bytes32`.
    """
    at = hex(block)
    calls = [("eth_call", [{"to": token, "data": "0x" + encode_call(sig).hex()}, at])
             for sig in ("decimals()", "symbol()")]
    try:
        got = transport.fetch_multi(calls, concurrent=True)
    except Exception:
        return None
    raw_decimals, raw_symbol = [*list(got), None, None][:2]
    if not isinstance(raw_decimals, str) or len(raw_decimals) < 4:
        return None
    try:
        decimals = decode_uint(bytes.fromhex(raw_decimals[2:]))
    except (ValueError, TypeError):
        return None
    if not 0 <= decimals <= 36:
        # Not a token, or one whose arithmetic this router cannot carry.
        return None
    return _symbol(raw_symbol) or token[:10], decimals


def _symbol(raw) -> str:
    if not isinstance(raw, str) or len(raw) < 4:
        return ""
    try:
        data = bytes.fromhex(raw[2:])
    except ValueError:
        return ""
    if len(data) >= 64:                       # the ABI string encoding
        try:
            size = decode_uint(data[32:64])
            if 0 < size <= 32 and len(data) >= 64 + size:
                return data[64:64 + size].decode("utf-8", "ignore").strip()
        except (ValueError, TypeError):
            pass
    return data[:32].rstrip(b"\x00").decode("utf-8", "ignore").strip()


def introduce(token: str, nodes, venues, transport, block: int) -> str | None:
    """Put `token` in the node map if a venue joins it to a priced coin.

    Returns the symbol it was introduced under, or `None` if it stays out.
    Idempotent, and a token the frame already prices is left alone -- adding it
    twice would be harmless but this is called on every pair change.
    """
    token = token.lower()
    if nodes.has(token) or not bridges_for(token, nodes, venues):
        return None
    meta = token_meta(transport, token, block)
    if meta is None:
        return None
    symbol, decimals = meta
    nodes.add_token(token, symbol, decimals)
    return symbol
