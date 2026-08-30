//! keccak-256, with no dependencies.
//!
//! The mirror of `core/keccak.py`, and written for the same reason it was:
//! `eth-hash` and `sha3` crates are fine, but this crate carries no
//! dependencies it does not have to (`README.md`), and a hash is seventy lines.
//!
//! Note this is *original Keccak* padding (`0x01`), not SHA-3's (`0x06`).
//! `sha3::Sha3_256` is a different hash and cannot be substituted -- which is
//! exactly the mistake that would produce plausible-looking selectors that no
//! contract answers to.

const RATE: usize = 136; // bytes absorbed per permutation for keccak-256

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];

/// rho offsets, indexed `[x][y]`.
const ROTATIONS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

fn rotl(value: u64, shift: u32) -> u64 {
    if shift == 0 { value } else { (value << shift) | (value >> (64 - shift)) }
}

fn keccak_f(a: &mut [[u64; 5]; 5]) {
    for rc in ROUND_CONSTANTS {
        // theta
        let c: [u64; 5] =
            std::array::from_fn(|x| a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4]);
        let d: [u64; 5] =
            std::array::from_fn(|x| c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1));
        for x in 0..5 {
            for y in 0..5 {
                a[x][y] ^= d[x];
            }
        }
        // rho + pi
        let mut b = [[0u64; 5]; 5];
        for x in 0..5 {
            for y in 0..5 {
                b[y][(2 * x + 3 * y) % 5] = rotl(a[x][y], ROTATIONS[x][y]);
            }
        }
        // chi
        for x in 0..5 {
            for y in 0..5 {
                a[x][y] = b[x][y] ^ (!b[(x + 1) % 5][y] & b[(x + 2) % 5][y]);
            }
        }
        // iota
        a[0][0] ^= rc;
    }
}

/// Ethereum's keccak-256 of `data`.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut state = [[0u64; 5]; 5];
    let mut padded = data.to_vec();
    padded.push(0x01);
    while padded.len() % RATE != 0 {
        padded.push(0x00);
    }
    let last = padded.len() - 1;
    padded[last] |= 0x80;

    for block in padded.chunks(RATE) {
        for i in 0..RATE / 8 {
            let mut lane = [0u8; 8];
            lane.copy_from_slice(&block[i * 8..i * 8 + 8]);
            state[i % 5][i / 5] ^= u64::from_le_bytes(lane);
        }
        keccak_f(&mut state);
    }

    let mut out = [0u8; 32];
    for i in 0..4 {
        // 32 bytes = 4 lanes off the top of the state
        out[i * 8..i * 8 + 8].copy_from_slice(&state[i % 5][i / 5].to_le_bytes());
    }
    out
}

/// The first four bytes of the hash of a canonical signature -- a selector.
pub fn selector(signature: &str) -> [u8; 4] {
    let full = keccak256(signature.as_bytes());
    [full[0], full[1], full[2], full[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn the_empty_hash_is_the_one_ethereum_uses() {
        // The most-quoted constant in Ethereum, and the one that says at a
        // glance whether this is Keccak or SHA-3: SHA-3 of the empty string is
        // a7ff...
        assert_eq!(
            hex(&keccak256(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
    }

    #[test]
    fn known_vectors() {
        assert_eq!(
            hex(&keccak256(b"abc")),
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
        );
        assert_eq!(
            hex(&keccak256(b"hello")),
            "1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8"
        );
    }

    #[test]
    fn a_block_boundary_is_absorbed_correctly() {
        // One under the rate, exactly it, one over, and two whole blocks. The
        // padding rule is the whole difficulty, and 136 is where a rule that
        // forgets to start a fresh block goes wrong.
        for (len, want) in [
            (135usize, "34367dc248bbd832f4e3e69dfaac2f92638bd0bbd18f2912ba4ef454919cf446"),
            (136, "a6c4d403279fe3e0af03729caada8374b5ca54d8065329a3ebcaeb4b60aa386e"),
            (137, "d869f639c7046b4929fc92a4d988a8b22c55fbadb802c0c66ebcd484f1915f39"),
            (272, "cf7fcd4f705ee749930d19ca84561a9bf62516bd90a471545fa2f49fdc7e63c8"),
        ] {
            assert_eq!(hex(&keccak256(&vec![0x61u8; len])), want, "len {len}");
        }
    }

    #[test]
    fn selectors_are_the_first_four_bytes() {
        // `transfer(address,uint256)` -- the most recognisable selector there
        // is, and a wrong padding rule would not produce it.
        assert_eq!(hex(&selector("transfer(address,uint256)")), "a9059cbb");
        assert_eq!(hex(&selector("get_dy(int128,int128,uint256)")), "5e0d443f");
    }
}
