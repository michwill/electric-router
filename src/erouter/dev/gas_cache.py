"""Measured execution gas, committed alongside the code that assumed it.

Gas is a property of the deployed contract and the shape of the trade, not of
the block: a pool that costs 118,778 to swap through costs about that at every
block until someone redeploys it.  So the measurements are worth keeping, and
worth committing -- a checkout should route with real gas figures without
having to execute anything first, exactly as it loads slot lists without having
to discover them.

Stored as plain JSON rather than the gzip the state cache uses.  It is three
orders of magnitude smaller, and a committed file that a human can read in a
diff is worth more than the bytes saved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from ..core.gas import GasTable
from ..core.pools import registry_key
from ..core.types import ArcKind

VERSION = 1
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "gas"


@dataclass(slots=True)
class GasCache:
    chain_id: int
    path: Path
    #: "address:kind:i>j" -> gas, so the file is legible in a diff
    legs: dict[str, int] = field(default_factory=dict)
    #: Medians a not-yet-measured leg inherits: "class:kind" and "kind".
    classes: dict[str, int] = field(default_factory=dict)
    kinds: dict[int, int] = field(default_factory=dict)
    #: address -> registry class, so the aggregates survive a reload.
    class_of: dict[str, str] = field(default_factory=dict)
    block: int = 0
    dirty: bool = False

    # ------------------------------------------------------------- load/save

    @classmethod
    def load(cls, chain_id: int, name: str, directory: Path | None = None) -> GasCache:
        path = (directory or DEFAULT_DIR) / f"{name}.json"
        cache = cls(chain_id=chain_id, path=path)
        if not path.exists():
            return cache
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != VERSION or raw.get("chain_id") != chain_id:
            return cache
        cache.legs = {k: int(v) for k, v in raw.get("legs", {}).items()}
        cache.classes = {k: int(v) for k, v in raw.get("classes", {}).items()}
        cache.kinds = {int(k): int(v) for k, v in raw.get("kinds", {}).items()}
        cache.class_of = dict(raw.get("class_of", {}))
        cache.block = int(raw.get("block", 0))
        return cache

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": VERSION,
            "chain_id": self.chain_id,
            "block": self.block,
            "legs": dict(sorted(self.legs.items())),
            "classes": dict(sorted(self.classes.items())),
            "kinds": {str(k): v for k, v in sorted(self.kinds.items())},
            "class_of": dict(sorted(self.class_of.items())),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(self.path)
        self.dirty = False

    # ------------------------------------------------------------ questions

    @staticmethod
    def key(target: str, kind: ArcKind | int, i: int, j: int) -> str:
        return f"{target.lower()}:{int(kind)}:{int(i)}>{int(j)}"

    def learn(self, measured: dict[tuple[str, int, int, int], int], *,
              block: int = 0, classes: dict[str, str] | None = None) -> int:
        """Fold in a measurement pass.  Returns how many figures changed.

        `classes` maps a pool address to its registry class, and is what lets
        an unmeasured pool inherit from pools of its own kind rather than from
        the all-pools median.  Passing it is optional; omitting it just leaves
        the class aggregates as they were.
        """
        changed = 0
        for (target, kind, i, j), gas in measured.items():
            key = self.key(target, kind, i, j)
            if self.legs.get(key) != int(gas):
                self.legs[key] = int(gas)
                changed += 1
        if classes:
            for address, name in classes.items():
                self.class_of.setdefault(address.lower(), name)
        if changed:
            self._aggregate()
            self.dirty = True
            self.block = block or self.block
        return changed

    def _aggregate(self) -> None:
        """Recompute the medians a not-yet-probed pool will be priced from.

        Median rather than max: the max exists because one *sample* of a leg
        caught a pool rebalancing inside `exchange`, which is the right figure
        for that leg and the wrong one to project onto every pool of its class.
        """
        by_kind: dict[int, list[int]] = {}
        by_class: dict[str, list[int]] = {}
        for key, gas in self.legs.items():
            address, kind, _ = key.split(":")
            by_kind.setdefault(int(kind), []).append(gas)
            name = self.class_of.get(address)
            if name:
                by_class.setdefault(f"{name}:{kind}", []).append(gas)
        self.kinds = {k: int(median(v)) for k, v in by_kind.items()}
        self.classes = {k: int(median(v)) for k, v in by_class.items()}

    def table(self, pools=None) -> GasTable:
        """The pure-core view, which is all the router itself ever sees.

        With `pools`, every pool that has no measurement of its own is given a
        per-pool default from the median of its class -- so a pool deployed
        after the last calibration is priced like its siblings instead of like
        the flat guess.  Without, the table still degrades sensibly through the
        per-kind medians.
        """
        legs: dict[tuple[str, int, int, int], int] = {}
        for key, gas in self.legs.items():
            address, kind, pair = key.split(":")
            i, j = pair.split(">")
            legs[(address, int(kind), int(i), int(j))] = int(gas)

        if pools:
            measured = {(a, k) for (a, k, _, _) in legs}
            for pool in pools:
                address = pool.address.lower()
                name = registry_key(pool.pool_type)
                for kind in (ArcKind.SWAP_STABLE, ArcKind.SWAP_CRYPTO):
                    if (address, int(kind)) in measured:
                        continue
                    got = self.classes.get(f"{name}:{int(kind)}")
                    if got:
                        legs[(address, int(kind), -1, -1)] = int(got)
        return GasTable(legs, self.kinds)

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        for key in self.legs:
            kind = ArcKind(int(key.split(":")[1]))
            by_kind[kind.name] = by_kind.get(kind.name, 0) + 1
        return {"legs": len(self.legs), "block": self.block, "by_kind": by_kind,
                "classes": len(self.classes), "kinds": dict(self.kinds)}
