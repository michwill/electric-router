"""The Rust `NodeMap` must answer what `core/nodes.py` answers.

Node merging is where a routing bug is quietest: get it wrong and the graph
still solves, it just solves a different graph -- ETH and WETH as two markets
instead of one, or a vault declared equal to its asset at a rate that is off.
Nothing raises.  So this compares the two implementations on the operations a
warm actually performs, in the order it performs them.

The interesting asymmetries are all exercised: merging a token that already
has members of its own (every member has to move), a rate that is exact in wei
and only approximate in float, and the address casing the reference is careful
about -- `node()` is called 32,000 times a route and almost always with a
lowercase address, so both sides try it as given before lowering it.
"""

from __future__ import annotations

import pytest

from erouter.core.accel import available
from erouter.core.nodes import Conversion, ConversionKind, NodeMap, rescale

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
ETH = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
CRVUSD = "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e"
SCRVUSD = "0x0655977feb2f289a4ab78af67bab0d17aab84367"
STETH = "0xae7ab96520de3a18e5e111b5eaab095312d7fe84"
WSTETH = "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0"

#: 1 scrvUSD = 1.0432... crvUSD, as the vault reported it at block 21,000,000.
VAULT_NUM = 1043251944382771456
VAULT_DEN = 10**18


def native():
    import erouter_solve

    return erouter_solve.NodeMap


def populated():
    """The same universe on both sides, built by the same calls."""
    tokens = [
        (WETH, "WETH", 18),
        (ETH, "ETH", 18),
        (CRVUSD, "crvUSD", 18),
        (SCRVUSD, "scrvUSD", 18),
        (STETH, "stETH", 18),
        (WSTETH, "wstETH", 18),
        ("0xdac17f958d2ee523a2206206994597c13d831ec7", "USDT", 6),
    ]
    reference, ported = NodeMap(), native()()
    for address, symbol, decimals in tokens:
        reference.add_token(address, symbol, decimals)
        ported.add_token(address, symbol, decimals)

    merges = [
        (ConversionKind.NATIVE_WRAP, ETH, WETH, 1, 1, WETH),
        (ConversionKind.ERC4626, SCRVUSD, CRVUSD, VAULT_NUM, VAULT_DEN, SCRVUSD),
        (ConversionKind.WSTETH, WSTETH, STETH, 1_204_183_982_113_311_744, 10**18, WSTETH),
    ]
    for kind, token, canonical, num, den, target in merges:
        reference.merge(
            Conversion(kind=kind, token=token, canonical=canonical,
                       rate_num=num, rate_den=den, target=target)
        )
        ported.merge(str(kind), token, canonical, str(num), str(den), target)
    return reference, ported


ALL_TOKENS = [WETH, ETH, CRVUSD, SCRVUSD, STETH, WSTETH,
              "0xdac17f958d2ee523a2206206994597c13d831ec7"]


def test_the_same_tokens_land_on_the_same_nodes():
    reference, ported = populated()
    assert ported.n_nodes() == reference.n_nodes
    assert ported.merged_nodes() == reference.merged_nodes()
    for token in ALL_TOKENS:
        assert ported.node(token) == reference.node(token), token
        assert ported.canonical(token) == reference.canonical(token), token
        assert ported.has(token) == reference.has(token), token
    for node in range(reference.n_nodes):
        assert ported.tokens_of(node) == reference.tokens_of[node]
        assert ported.node_symbol(node) == reference.node_symbol(node)


def test_symbols_and_decimals_agree_including_the_fallbacks():
    reference, ported = populated()
    for token in [*ALL_TOKENS, "0xnotatoken", "0xNOTATOKENEITHER"]:
        assert ported.symbol(token) == reference.symbol(token), token
        assert ported.decimals(token) == reference.decimals(token), token


def test_a_checksummed_address_resolves_the_same_way():
    """The lookups try the address as given, then lowered. Both sides."""
    reference, ported = populated()
    mixed = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    assert ported.has(mixed) == reference.has(mixed) is True
    assert ported.node(mixed) == reference.node(mixed)
    assert ported.rate(mixed) == reference.rate(mixed)


@pytest.mark.parametrize(
    "amount",
    [0, 1, 7, 10**6, 10**18, 3 * 10**18 + 1, 10**26, 2**200, 2**245],
)
def test_conversions_are_exact_to_the_wei(amount):
    reference, ported = populated()
    for token in ALL_TOKENS:
        assert int(ported.to_canonical_wei(token, str(amount))) == \
            reference.to_canonical_wei(token, amount), (token, amount)
        assert int(ported.from_canonical_wei(token, str(amount))) == \
            reference.from_canonical_wei(token, amount), (token, amount)


def test_float_rates_agree_bit_for_bit():
    reference, ported = populated()
    for token in ALL_TOKENS:
        got, want = ported.rate(token), reference.rate(token)
        assert got == want, (token, got.hex(), want.hex())


def test_conversion_kinds_agree():
    reference, ported = populated()
    for token in ALL_TOKENS:
        conversion = reference.conversion.get(token)
        if conversion is None:
            assert ported.conversion(token) is None
            assert ported.conversion_kinds(token) is None
            assert ported.is_alias(token) is False
            continue
        kind, canonical, num, den, target = ported.conversion(token)
        assert kind == str(conversion.kind)
        assert canonical == conversion.canonical
        assert (int(num), int(den)) == (conversion.rate_num, conversion.rate_den)
        assert target == conversion.target
        assert ported.conversion_kinds(token) == (
            int(conversion.forward_kind), int(conversion.reverse_kind))
        assert ported.is_alias(token) == conversion.is_alias


def test_an_alias_is_a_merge_with_nothing_to_call():
    """Gnosis EURe: two addresses over one balance."""
    v1, v2 = "0xeure000000000000000000000000000000000001", "0xeure000000000000000000000000000000000002"
    reference, ported = NodeMap(), native()()
    for address in (v1, v2):
        reference.add_token(address, "EURe", 18)
        ported.add_token(address, "EURe", 18)
    reference.merge(Conversion(kind=ConversionKind.ALIAS, token=v2, canonical=v1))
    ported.merge("ALIAS", v2, v1)

    assert ported.node(v2) == reference.node(v2) == 0
    assert ported.is_alias(v2) is reference.conversion[v2].is_alias is True
    assert ported.rate(v2) == reference.rate(v2) == 1.0
    # One symbol, printed once, not "EURe/EURe".
    assert ported.node_symbol(0) == reference.node_symbol(0) == "EURe"


def test_folding_a_populated_node_moves_every_member():
    a, b, c = "0xaa", "0xbb", "0xcc"
    reference, ported = NodeMap(), native()()
    for address in (a, b, c):
        reference.add_token(address, address.upper(), 18)
        ported.add_token(address, address.upper(), 18)
    for token, canonical in ((c, b), (b, a)):
        reference.merge(Conversion(kind=ConversionKind.ALIAS, token=token, canonical=canonical))
        ported.merge("ALIAS", token, canonical)

    for token in (a, b, c):
        assert ported.node(token) == reference.node(token)
    assert ported.tokens_of(0) == reference.tokens_of[0]
    assert ported.tokens_of(1) == reference.tokens_of[1] == []
    assert ported.merged_nodes() == reference.merged_nodes()


def test_merging_onto_an_unknown_canonical_is_refused_on_both_sides():
    reference, ported = NodeMap(), native()()
    reference.add_token(ETH, "ETH", 18)
    ported.add_token(ETH, "ETH", 18)
    with pytest.raises(KeyError, match="is not in the graph"):
        reference.merge(Conversion(kind=ConversionKind.NATIVE_WRAP, token=ETH, canonical=WETH))
    with pytest.raises(KeyError, match="is not in the graph"):
        ported.merge("NATIVE_WRAP", ETH, WETH)


@pytest.mark.parametrize(
    "a,B,rate_in,rate_out",
    [
        (1.0, 1e-6, 1.0, 1.0),
        (0.9997, 4.2e-9, 1.0432519443827714, 1.0),
        (3210.5, 1.1e-12, 1.0, 3210.5),
        (1e-8, 1e-30, 1204.18, 0.0009),
    ],
)
def test_rescale_agrees_bit_for_bit(a, B, rate_in, rate_out):
    got = native().rescale(a, B, rate_in, rate_out)
    want = rescale(a, B, rate_in, rate_out)
    assert got == want, (got, want)


def test_a_non_positive_rate_is_refused_on_both_sides():
    with pytest.raises(ValueError, match="must be positive"):
        rescale(1.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="must be positive"):
        native().rescale(1.0, 1.0, 0.0, 1.0)


def test_the_refusal_reads_the_same():
    """Including how the two spell a float, which is not the same by default."""
    try:
        rescale(1.0, 1.0, -2.5, 0.0)
    except ValueError as e:
        want = str(e)
    try:
        native().rescale(1.0, 1.0, -2.5, 0.0)
    except ValueError as e:
        got = str(e)
    assert got == want


def test_a_conversion_that_will_not_fit_refuses_rather_than_wrapping():
    """The mirror's one deliberate divergence, and it is loud.

    `to_canonical` is Python `int` on the reference side and has no ceiling;
    on the Rust side the product widens to 512 bits so every representable
    answer is exact, and a quotient past `2**256` -- which is more wei than any
    ERC20 can hold -- is refused instead of silently wrapping.
    """
    _, ported = populated()
    # Just under the ceiling: the scrvUSD rate is 1.043, so scaling it up is
    # the first product whose *quotient* will not fit either.
    huge = str(2**256 - 1)
    assert int(ported.to_canonical_wei(WETH, huge)) == 2**256 - 1  # no conversion, no product
    with pytest.raises(ValueError, match="does not fit in 256 bits"):
        ported.to_canonical_wei(SCRVUSD, huge)
    # The other direction divides, so it still answers -- exactly.
    assert int(ported.from_canonical_wei(SCRVUSD, huge)) == \
        (2**256 - 1) * VAULT_DEN // VAULT_NUM


def test_a_zero_denominator_is_caught_where_it_is_set():
    """The reference divides by it later; this refuses it now.

    `Conversion(rate_den=0)` builds fine on the Python side and raises
    `ZeroDivisionError` from the first `to_canonical` -- a long way from the
    line that set it.
    """
    _, ported = populated()
    with pytest.raises(ValueError, match="rate_den must not be zero"):
        ported.merge("ERC4626", "0xdeadbeef", WETH, "1", "0")
