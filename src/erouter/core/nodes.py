"""Node merging: tokens that are the same thing at zero cost (spec §3.1).

A wrapped native (WETH/ETH) and a linear ERC4626 vault (scrvUSD/crvUSD) are
*zero-resistance elements*.  In value coordinates a linear arc has `eps = 0` and
`G = inf` -- a short circuit -- which means the two tokens are literally the
same graph node.

Merging rather than adding an arc is not a nicety:

* it avoids a degenerate `B = 0`, `eps_f + eps_r = 0` arc pair, which would
  violate §2.6's `eps_f + eps_r > 0` guard and §12.4's `clamped => cap < inf`;
* it connects halves of the graph that are otherwise nearly disjoint -- eight
  Ethereum pools hold native ETH under the `0xEeee...` sentinel (including the
  $77M ETH/stETH pool) while 63 hold WETH.

The conversion itself is materialised only when a route is emitted, as a leg.

**Merging is allowlist-gated, never inferred.**  Linearity of `convertToAssets`
is necessary and nowhere near sufficient: of 31 linear ERC4626 tokens in the
Ethereum universe, `pufETH` reports `asset = WETH` with zero linearity error
and redeems through a *withdrawal queue*, `sUSDe` has a 7-day cooldown, and
`sfrxUSD` has `maxDeposit == 0`.  Merging any of those would declare the vault
equal to its asset at NAV and mint the market discount out of thin air.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .types import ArcKind


class ConversionKind(StrEnum):
    NATIVE_WRAP = "NATIVE_WRAP"  # 1:1, ERC20 <-> native
    ERC4626 = "ERC4626"  # shares <-> assets at the vault's rate


@dataclass(frozen=True, slots=True)
class Conversion:
    """How to get from `token` to the canonical token of its node."""

    kind: ConversionKind
    token: str  # the non-canonical side
    canonical: str  # the node's canonical token
    # Exact integer rate: canonical wei per 10**decimals of `token`.
    rate_num: int = 1
    rate_den: int = 1
    target: str = ""  # contract to call (WETH, or the vault)

    @property
    def rate(self) -> float:
        """Canonical units per unit of `token`, in human terms."""
        return self.rate_num / self.rate_den

    def to_canonical(self, amount: int) -> int:
        return amount * self.rate_num // self.rate_den

    def from_canonical(self, amount: int) -> int:
        return amount * self.rate_den // self.rate_num

    @property
    def forward_kind(self) -> ArcKind:
        """token -> canonical."""
        if self.kind is ConversionKind.NATIVE_WRAP:
            # The canonical side of a native pair is the wrapped ERC20, so
            # going from native to canonical is a wrap.
            return ArcKind.WRAP_NATIVE
        return ArcKind.ERC4626_REDEEM  # shares -> assets

    @property
    def reverse_kind(self) -> ArcKind:
        """canonical -> token."""
        if self.kind is ConversionKind.NATIVE_WRAP:
            return ArcKind.UNWRAP_NATIVE
        return ArcKind.ERC4626_DEPOSIT  # assets -> shares


@dataclass(slots=True)
class NodeMap:
    """Token address -> graph node, plus how to convert between members."""

    node_of: dict[str, int] = field(default_factory=dict)
    tokens_of: list[list[str]] = field(default_factory=list)
    canonical_of: list[str] = field(default_factory=list)
    conversion: dict[str, Conversion] = field(default_factory=dict)
    symbol_of: dict[str, str] = field(default_factory=dict)
    decimals_of: dict[str, int] = field(default_factory=dict)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------- building

    def add_token(self, address: str, symbol: str = "", decimals: int = 18) -> int:
        key = address.lower()
        if key not in self.node_of:
            node = len(self.tokens_of)
            self.node_of[key] = node
            self.tokens_of.append([key])
            self.canonical_of.append(key)
        if symbol:
            self.symbol_of.setdefault(key, symbol)
        self.decimals_of.setdefault(key, decimals)
        return self.node_of[key]

    def merge(self, conversion: Conversion) -> None:
        """Fold `conversion.token` into the node of `conversion.canonical`."""
        token = conversion.token.lower()
        canonical = conversion.canonical.lower()
        if canonical not in self.node_of:
            raise KeyError(f"canonical token {canonical} is not in the graph")
        target = self.node_of[canonical]
        if token in self.node_of and self.node_of[token] == target:
            self.conversion[token] = conversion
            return

        old = self.node_of.get(token)
        if old is not None and old != target:
            for member in self.tokens_of[old]:
                self.node_of[member] = target
                if member not in self.tokens_of[target]:
                    self.tokens_of[target].append(member)
            self.tokens_of[old] = []
        else:
            self.node_of[token] = target
            self.tokens_of[target].append(token)
        self.conversion[token] = conversion

    # -------------------------------------------------------------- lookup

    def node(self, token: str) -> int:
        return self.node_of[token.lower()]

    def has(self, token: str) -> bool:
        return token.lower() in self.node_of

    def canonical(self, token: str) -> str:
        return self.canonical_of[self.node(token)]

    def rate(self, token: str) -> float:
        """Canonical units per unit of `token` (1.0 for a canonical token)."""
        conversion = self.conversion.get(token.lower())
        return conversion.rate if conversion else 1.0

    def to_canonical_wei(self, token: str, amount: int) -> int:
        conversion = self.conversion.get(token.lower())
        return conversion.to_canonical(amount) if conversion else amount

    def from_canonical_wei(self, token: str, amount: int) -> int:
        conversion = self.conversion.get(token.lower())
        return conversion.from_canonical(amount) if conversion else amount

    def symbol(self, token: str) -> str:
        return self.symbol_of.get(token.lower(), token[:10])

    def decimals(self, token: str) -> int:
        return self.decimals_of.get(token.lower(), 18)

    def node_symbol(self, node: int) -> str:
        """A label for the merged node, e.g. `ETH/WETH`."""
        members = [t for t in self.tokens_of[node] if t in self.symbol_of]
        if not members:
            return f"node{node}"
        canonical = self.canonical_of[node]
        ordered = [canonical, *[m for m in members if m != canonical]]
        return "/".join(dict.fromkeys(self.symbol_of[m] for m in ordered if m in self.symbol_of))

    @property
    def n_nodes(self) -> int:
        return len(self.tokens_of)

    def merged_nodes(self) -> list[int]:
        return [k for k, members in enumerate(self.tokens_of) if len(members) > 1]


def rescale(a: float, B: float, rate_in: float, rate_out: float) -> tuple[float, float]:
    """Re-express an arc's derivatives in canonical units.

        a_canonical = a * R_out / R_in
        B_canonical = B * R_out / R_in^2

    `B` has units of output per input *squared*, so the input rate enters
    twice.  Getting this wrong is silent: the arc still solves, just with a
    conductance off by the price ratio.
    """
    if rate_in <= 0 or rate_out <= 0:
        raise ValueError(f"conversion rates must be positive ({rate_in}, {rate_out})")
    return a * rate_out / rate_in, B * rate_out / (rate_in * rate_in)
