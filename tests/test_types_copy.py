"""`PoolArc` copies itself through its own constructor, and copies all of it.

`copy.copy` on a slots dataclass has no `__dict__` to hand over, so it goes
through the reduce protocol and sets every field with the `setattr` builtin.
The pipeline copies the whole universe once a quote -- 820 arcs, because
`_assemble` writes `G` and `eps` back and the refit re-anchors `B`, so sharing
them would leak one quote's size into the next -- and that measured 4.00 ms
against 1.54 for the plain constructor.

The copier is generated from `fields()` for a reason worth keeping: one written
out by hand and left behind by a new field does not fail, it silently shares
that field between the copy and the original, which is the exact leak the copy
exists to prevent.  This is the test that would catch it either way.
"""

from __future__ import annotations

import copy
import dataclasses
import math

from erouter.core.types import ArcKind, FlagReason, PoolArc

#: One value per field, all different from the defaults, so a field the copier
#: forgot shows up as its default rather than passing on a coincidence.
LOADED = {
    "id": "arc-id", "pool": "0x" + "ab" * 20, "kind": ArcKind.SWAP_CRYPTO,
    "i": 3, "j": 5, "n_coins": 8, "token_in": "0x" + "11" * 20,
    "token_out": "0x" + "22" * 20, "tau": 7, "sigma": 9,
    "flag_reason": FlagReason.NONE, "convex_flag": True, "clamped": True,
    "reverse_id": "other-arc", "ladder": None, "note": "a note",
}


def loaded() -> PoolArc:
    """An arc with every field set to something distinctive."""
    values = {}
    for k, spec in enumerate(dataclasses.fields(PoolArc), start=1):
        if spec.name in LOADED:
            values[spec.name] = LOADED[spec.name]
        elif spec.type in ("int",):
            values[spec.name] = k * 3
        elif spec.type in ("float",):
            values[spec.name] = k * 1.5
        elif spec.type in ("bool",):
            values[spec.name] = True
        elif spec.type in ("str",):
            values[spec.name] = f"value-{k}"
        else:
            values[spec.name] = LOADED.get(spec.name)
    return PoolArc(**values)


def test_every_field_survives_the_copy():
    original = loaded()
    clone = copy.copy(original)
    assert clone is not original
    missed = [f.name for f in dataclasses.fields(PoolArc)
              if getattr(clone, f.name) != getattr(original, f.name)]
    assert not missed, f"__copy__ does not carry {missed}"


def test_the_copy_does_not_share_the_fields_the_pipeline_writes_back():
    # `_assemble` writes these, so a copy that aliased them would put one
    # quote's calibration into the next -- which is the whole point of copying.
    original = loaded()
    clone = copy.copy(original)
    clone.G, clone.eps, clone.B = 1.0, 2.0, 3.0
    assert (original.G, original.eps, original.B) != (1.0, 2.0, 3.0)


def test_the_copier_is_the_generated_one_and_not_the_reduce_protocol():
    # If this ever falls back, the cost returns silently: `copy.copy` still
    # gives a correct answer, four times slower.
    assert PoolArc.__copy__ is not None
    assert PoolArc.__copy__.__name__ == "__copy__"


def test_a_nan_field_copies_as_nan_rather_than_comparing_equal():
    # `eta` defaults to NaN and NaN != NaN, so the equality sweep above would
    # miss it.  Checked directly instead.
    original = PoolArc(id="x", pool="0xpool", kind=ArcKind.SWAP_STABLE, i=0, j=1,
                       n_coins=2, token_in="0xa", token_out="0xb", tau=0, sigma=1)
    assert math.isnan(original.eta)
    assert math.isnan(copy.copy(original).eta)
