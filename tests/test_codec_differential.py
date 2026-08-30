"""The Rust ABI codec must encode what `core/codec.py` encodes, byte for byte.

This is the one place in the port where "close enough" has no meaning at all:
calldata is either the bytes a contract answers to or it is not, and a selector
one bit out reaches a different function or none.

The reference is itself held to `eth_abi` by `tests/test_codec.py`, so matching
it transitively matches the real encoder.  Keccak is checked against published
vectors rather than against the reference alone, because both could be wrong
the same way -- Ethereum uses *original* Keccak padding (0x01), not SHA-3's
(0x06), and `hashlib.sha3_256` is a different hash that would produce
plausible-looking selectors no contract answers to.
"""

from __future__ import annotations

import pytest

from erouter.core import codec
from erouter.core.accel import available
from erouter.core.keccak import keccak256

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")


def native():
    import erouter_solve

    return erouter_solve


# ------------------------------------------------------------------ keccak


HASHED = [
    b"",
    b"a",
    b"abc",
    b"hello",
    b"transfer(address,uint256)",
    bytes(range(256)),
    b"a" * 135,   # one under the rate
    b"a" * 136,   # exactly the rate, where a padding bug shows
    b"a" * 137,
    b"a" * 272,
]


@pytest.mark.parametrize("data", HASHED, ids=[f"{len(d)}b" for d in HASHED])
def test_keccak_agrees(data):
    assert native().keccak256(data) == keccak256(data)


def test_keccak_is_keccak_and_not_sha3():
    """The constant that says which hash this is at a glance."""
    import hashlib

    empty = native().keccak256(b"")
    assert empty.hex() == \
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert empty != hashlib.sha3_256(b"").digest()


SIGNATURES = [
    "transfer(address,uint256)",
    "get_dy(int128,int128,uint256)",
    "get_dy(uint256,uint256,uint256)",
    "calc_withdraw_one_coin(uint256,int128)",
    "calc_token_amount(uint256[],bool)",
    "balanceOf(address)",
    "totalSupply()",
]


@pytest.mark.parametrize("signature", SIGNATURES)
def test_selectors_agree(signature):
    assert native().selector(signature) == codec.selector(signature)
    assert native().signature_types(signature) == codec.signature_types(signature)


# ------------------------------------------------------------------- types


TYPES = [
    "uint256", "uint8", "uint128", "int128", "int8", "address", "bool",
    "bytes", "string", "bytes32", "bytes4",
    "uint256[]", "uint256[3]", "address[]", "bytes[2]",
    "(uint256,bool)", "(uint256,bytes)", "((uint8,bool),address)",
    "(uint256,bool)[]", "(uint256,bool)[2]",
]


@pytest.mark.parametrize("text", TYPES)
def test_the_type_grammar_agrees(text):
    """`is_dynamic` and `head_size` decide where every offset points."""
    parsed = codec.parse_type(text)
    assert native().is_dynamic(text) == codec.is_dynamic(parsed)
    assert native().head_size(text) == codec.head_size(parsed)


@pytest.mark.parametrize("text", ["", "uint7x", "wat", "uint256[[", "bytesx"])
def test_an_unsupported_type_is_refused_on_both_sides(text):
    with pytest.raises(ValueError):
        codec.parse_type(text)
    with pytest.raises(ValueError):
        native().is_dynamic(text)


# ---------------------------------------------------------------- encoding
#
# Values cross as a small tagged form -- `("uint", "1000")`, `("int", -1)` --
# because the Rust side is statically typed and Python's are not. The tags are
# what the two agree on.

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

CASES = [
    (["uint256"], [("uint", "0")]),
    (["uint256"], [("uint", str(2**256 - 1))]),
    (["uint8"], [("uint", "255")]),
    (["int128"], [("int", -1)]),
    (["int128"], [("int", 0)]),
    (["int128"], [("int", 2**127 - 1)]),
    (["int128"], [("int", -(2**127))]),
    (["int128", "int128", "uint256"], [("int", 0), ("int", 1), ("uint", "1000")]),
    (["address"], [("address", WETH)]),
    (["address"], [("address", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")]),
    (["bool", "bool"], [("bool", True), ("bool", False)]),
    (["bytes32"], [("bytes", bytes(range(32)))]),
    (["bytes4"], [("bytes", b"\xa9\x05\x9c\xbb")]),
    (["bytes"], [("bytes", b"")]),
    (["bytes"], [("bytes", b"\x01\x02\x03")]),
    (["bytes"], [("bytes", bytes(range(32)))]),
    (["bytes"], [("bytes", bytes(range(33)))]),
    (["string"], [("string", "")]),
    (["string"], [("string", "hello")]),
    (["uint256[]"], [("array", [("uint", "7"), ("uint", "9")])]),
    (["uint256[]"], [("array", [])]),
    (["uint256[3]"], [("array", [("uint", "1"), ("uint", "2"), ("uint", "3")])]),
    (["uint256[]", "bool"], [("array", [("uint", "5")]), ("bool", True)]),
    (["(uint256,bool)"], [("tuple", [("uint", "42"), ("bool", True)])]),
    (["(uint256,bytes)"], [("tuple", [("uint", "42"), ("bytes", b"\xff")])]),
    (["bool", "bytes", "uint256"],
     [("bool", True), ("bytes", b"\x01" * 40), ("uint", "9")]),
    (["address[]"], [("array", [("address", WETH), ("address", WETH)])]),
]


def to_python(value):
    """The tagged form as `core/codec.py` wants it."""
    tag, v = value
    if tag == "uint":
        return int(v)
    if tag == "int":
        return v
    if tag in ("address", "string", "bool", "bytes"):
        return v
    if tag in ("array", "tuple"):
        return [to_python(one) for one in v]
    raise AssertionError(tag)


@pytest.mark.parametrize("types,values", CASES,
                         ids=[",".join(t) for t, _ in CASES])
def test_encoding_agrees_byte_for_byte(types, values):
    want = codec.encode(types, [to_python(v) for v in values])
    assert native().abi_encode(types, values) == want


@pytest.mark.parametrize("types,values", CASES,
                         ids=[",".join(t) for t, _ in CASES])
def test_decoding_agrees(types, values):
    data = codec.encode(types, [to_python(v) for v in values])
    want = codec.decode(types, data)
    got = native().abi_decode(types, data)
    assert [to_python(v) for v in got] == want


def test_a_whole_call_agrees():
    """Selector plus arguments -- the artefact a node is actually sent."""
    for signature, values in [
        ("get_dy(int128,int128,uint256)",
         [("int", 0), ("int", 1), ("uint", str(10**18))]),
        ("transfer(address,uint256)", [("address", WETH), ("uint", "1")]),
        ("totalSupply()", []),
        ("calc_token_amount(uint256[],bool)",
         [("array", [("uint", "1"), ("uint", "2")]), ("bool", True)]),
    ]:
        want = codec.encode_call(signature, *[to_python(v) for v in values])
        assert native().encode_call(signature, values) == want


@pytest.mark.parametrize("bad", [
    (["uint8"], [("uint", "256")]),
    (["int8"], [("int", 128)]),
    (["int8"], [("int", -129)]),
    (["bytes4"], [("bytes", b"\x01\x02")]),
    (["uint256[2]"], [("array", [("uint", "1")])]),
    (["uint256", "bool"], [("uint", "1")]),
])
def test_the_same_values_are_refused(bad):
    types, values = bad
    with pytest.raises(ValueError):
        codec.encode(types, [to_python(v) for v in values])
    with pytest.raises(ValueError):
        native().abi_encode(types, values)


def test_empty_returndata_is_refused_rather_than_read_as_zero():
    """A Curve pool that lacks a function returns empty data, not a revert."""
    with pytest.raises(ValueError):
        codec.decode_uint(b"")
    with pytest.raises(ValueError):
        native().decode_uint(b"")
    assert native().decode_uint((1234).to_bytes(32, "big")) == "1234"


def test_int256_is_refused_here_and_supported_there():
    """The port's one narrowing, stated rather than silent.

    Python integers are unbounded, so the reference carries `int256` for free.
    The port holds signed values in `i128`, and Curve's ABI uses `int128` for
    coin indices and nothing wider -- so this refuses at parse time instead of
    truncating.
    """
    assert codec.parse_type("int256") == ("int", 256)
    with pytest.raises(ValueError, match="wider than this codec carries"):
        native().is_dynamic("int256")
