//! Minimal ABI encoder/decoder (mirror of `core/codec.py`).
//!
//! The reference is stdlib-only for the same reason this is dependency-free:
//! `eth-abi` pulls compiled dependencies that make a Pyodide build
//! impractical, and the subset a router needs is small enough to write down.
//!
//! Supported grammar -- deliberately no more than the quoter's ABI needs:
//!
//!     uint<N> int<N> address bool bytes string bytes<N>
//!     (T1,T2,...)          tuples, nested
//!     T[]  T[k]            dynamic and fixed arrays
//!
//! Selectors are derived from canonical signature strings at the call site
//! rather than loaded from ABI JSON, which is what makes the two Curve
//! dialects a one-line difference.
//!
//! **One narrowing against the reference.** Signed integers are held as
//! `i128`, so `int144` and wider are refused at parse time rather than
//! silently truncated. Curve's ABI uses `int128` for coin indices and nothing
//! wider; the reference supports `int256` because Python integers are free,
//! not because anything asks for it.

use crate::keccak::keccak256;
use ruint::aliases::U256;
use std::fmt;

pub const WORD: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecError(pub String);

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

type Result<T> = std::result::Result<T, CodecError>;

/// A parsed ABI type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Uint(u32),
    Int(u32),
    Address,
    Bool,
    Bytes,
    String,
    /// `bytes<N>`
    FixedBytes(usize),
    /// `None` length is a dynamic array.
    Array(Box<Type>, Option<usize>),
    Tuple(Vec<Type>),
}

/// A value to encode, or one decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Uint(U256),
    Int(i128),
    /// Lowercase `0x`-prefixed, as the reference decodes it.
    Address(String),
    Bool(bool),
    Bytes(Vec<u8>),
    String(String),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
}

// ---------------------------------------------------------------- types

/// Split on commas that are not nested inside brackets or parens.
fn split_top(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, ch) in text.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(text[start..i].to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = &text[start..];
    if !tail.is_empty() || !out.is_empty() {
        out.push(tail.to_string());
    }
    out.into_iter().map(|t| t.trim().to_string()).collect()
}

pub fn parse_type(text: &str) -> Result<Type> {
    let text = text.trim();
    if let Some(stripped) = text.strip_suffix(']') {
        let open_at = stripped
            .rfind('[')
            .ok_or_else(|| CodecError(format!("unsupported ABI type: {text:?}")))?;
        let inner = parse_type(&stripped[..open_at])?;
        let size = &stripped[open_at + 1..];
        let length = if size.is_empty() {
            None
        } else {
            Some(size.parse::<usize>().map_err(|_| {
                CodecError(format!("unsupported ABI type: {text:?}"))
            })?)
        };
        return Ok(Type::Array(Box::new(inner), length));
    }
    if text.starts_with('(') && text.ends_with(')') {
        let body = text[1..text.len() - 1].trim();
        if body.is_empty() {
            return Ok(Type::Tuple(Vec::new()));
        }
        let parts: Result<Vec<Type>> = split_top(body).iter().map(|t| parse_type(t)).collect();
        return Ok(Type::Tuple(parts?));
    }
    let bits = |rest: &str, whole: &str| -> Result<u32> {
        if rest.is_empty() {
            return Ok(256);
        }
        rest.parse::<u32>()
            .map_err(|_| CodecError(format!("unsupported ABI type: {whole:?}")))
    };
    Ok(match text {
        "address" => Type::Address,
        "bool" => Type::Bool,
        "bytes" => Type::Bytes,
        "string" => Type::String,
        _ if text.starts_with("uint") => Type::Uint(bits(&text[4..], text)?),
        _ if text.starts_with("int") => {
            let n = bits(&text[3..], text)?;
            if n > 128 {
                // See the module note: `i128` is the widest signed value this
                // carries, and refusing beats truncating.
                return Err(CodecError(format!(
                    "int{n} is wider than this codec carries (int128 is the \
                     widest Curve's ABI uses)"
                )));
            }
            Type::Int(n)
        }
        _ if text.starts_with("bytes") => Type::FixedBytes(
            text[5..]
                .parse::<usize>()
                .map_err(|_| CodecError(format!("unsupported ABI type: {text:?}")))?,
        ),
        _ => return Err(CodecError(format!("unsupported ABI type: {text:?}"))),
    })
}

pub fn is_dynamic(t: &Type) -> bool {
    match t {
        Type::Bytes | Type::String => true,
        Type::Array(inner, size) => size.is_none() || is_dynamic(inner),
        Type::Tuple(members) => members.iter().any(is_dynamic),
        _ => false,
    }
}

/// Bytes this type occupies in a head region (32 if it is dynamic).
pub fn head_size(t: &Type) -> usize {
    if is_dynamic(t) {
        return WORD;
    }
    match t {
        Type::Tuple(members) => members.iter().map(head_size).sum(),
        Type::Array(inner, Some(size)) => size * head_size(inner),
        _ => WORD,
    }
}

// -------------------------------------------------------------- encoding

/// Two's-complement 32-byte big-endian word.
fn word(value: U256) -> [u8; 32] {
    value.to_be_bytes()
}

fn signed_word(value: i128) -> [u8; 32] {
    let mut out = if value < 0 { [0xffu8; 32] } else { [0u8; 32] };
    out[16..].copy_from_slice(&(value as u128).to_be_bytes());
    out
}

fn address_word(text: &str) -> Result<[u8; 32]> {
    let hex = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")).unwrap_or(text);
    let value = U256::from_str_radix(hex, 16)
        .map_err(|_| CodecError(format!("not an address: {text:?}")))?;
    Ok(word(value))
}

/// Full encoding of one value (what would go in a tail region).
fn enc_value(t: &Type, v: &Value) -> Result<Vec<u8>> {
    let mismatch =
        || CodecError(format!("cannot encode {v:?} as {t:?}"));
    Ok(match (t, v) {
        (Type::Uint(bits), Value::Uint(n)) => {
            if *bits < 256 && *n >= (U256::from(1u8) << *bits as usize) {
                return Err(CodecError(format!("uint{bits} out of range: {n}")));
            }
            word(*n).to_vec()
        }
        (Type::Int(bits), Value::Int(n)) => {
            // `1i128 << 127` is `i128::MIN`, not a limit, so the widest width
            // is checked by construction rather than by arithmetic.
            if *bits < 128 {
                let limit = 1i128 << (*bits - 1);
                if *n < -limit || *n >= limit {
                    return Err(CodecError(format!("int{bits} out of range: {n}")));
                }
            }
            signed_word(*n).to_vec()
        }
        (Type::Address, Value::Address(text)) => address_word(text)?.to_vec(),
        (Type::Address, Value::Uint(n)) => word(*n).to_vec(),
        (Type::Bool, Value::Bool(b)) => {
            word(U256::from(u8::from(*b))).to_vec()
        }
        (Type::FixedBytes(size), Value::Bytes(b)) => {
            if b.len() != *size {
                return Err(CodecError(format!("bytes{size} got {} bytes", b.len())));
            }
            let mut out = b.clone();
            out.resize(WORD, 0);
            out
        }
        (Type::Bytes, Value::Bytes(b)) => dynamic_bytes(b),
        (Type::String, Value::String(s)) => dynamic_bytes(s.as_bytes()),
        (Type::Array(inner, size), Value::Array(items)) => {
            if let Some(size) = size {
                if items.len() != *size {
                    return Err(CodecError(format!(
                        "fixed array wants {size} items, got {}",
                        items.len()
                    )));
                }
            }
            let types: Vec<Type> = vec![(**inner).clone(); items.len()];
            let body = encode_tuple(&types, items)?;
            match size {
                Some(_) => body,
                None => {
                    let mut out = word(U256::from(items.len())).to_vec();
                    out.extend(body);
                    out
                }
            }
        }
        (Type::Tuple(members), Value::Tuple(items)) => encode_tuple(members, items)?,
        _ => return Err(mismatch()),
    })
}

fn dynamic_bytes(b: &[u8]) -> Vec<u8> {
    let mut out = word(U256::from(b.len())).to_vec();
    out.extend_from_slice(b);
    out.resize(out.len() + (WORD - b.len() % WORD) % WORD, 0);
    out
}

pub fn encode_tuple(types: &[Type], values: &[Value]) -> Result<Vec<u8>> {
    if types.len() != values.len() {
        return Err(CodecError(format!(
            "arity mismatch: {} types, {} values",
            types.len(),
            values.len()
        )));
    }
    let mut offset: usize = types.iter().map(head_size).sum();
    let mut heads: Vec<u8> = Vec::new();
    let mut tails: Vec<u8> = Vec::new();
    for (t, v) in types.iter().zip(values.iter()) {
        if is_dynamic(t) {
            let tail = enc_value(t, v)?;
            heads.extend_from_slice(&word(U256::from(offset)));
            offset += tail.len();
            tails.extend(tail);
        } else {
            heads.extend(enc_value(t, v)?);
        }
    }
    heads.extend(tails);
    Ok(heads)
}

pub fn encode(types: &[String], values: &[Value]) -> Result<Vec<u8>> {
    let parsed: Result<Vec<Type>> = types.iter().map(|t| parse_type(t)).collect();
    encode_tuple(&parsed?, values)
}

// -------------------------------------------------------------- decoding

fn at(data: &[u8], pos: usize) -> Result<U256> {
    if pos + WORD > data.len() {
        return Err(CodecError(format!(
            "want a word at {pos}, data is {} bytes",
            data.len()
        )));
    }
    Ok(U256::from_be_slice(&data[pos..pos + WORD]))
}

fn dec_value(t: &Type, data: &[u8], pos: usize) -> Result<Value> {
    Ok(match t {
        Type::Uint(_) => Value::Uint(at(data, pos)?),
        Type::Int(_) => {
            // Two's complement: the reference subtracts `1 << 256` when the
            // top bit is set. Reinterpreting the low 128 bits as signed does
            // the same thing for every width this codec admits, because a
            // negative value is sign-extended through them.
            let raw = at(data, pos)?;
            Value::Int((raw & U256::from(u128::MAX)).to::<u128>() as i128)
        }
        Type::Address => {
            if pos + WORD > data.len() {
                return Err(CodecError("truncated address".into()));
            }
            Value::Address(format!(
                "0x{}",
                data[pos + 12..pos + WORD]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            ))
        }
        Type::Bool => Value::Bool(!at(data, pos)?.is_zero()),
        Type::FixedBytes(size) => {
            if pos + size > data.len() {
                return Err(CodecError("truncated bytesN".into()));
            }
            Value::Bytes(data[pos..pos + size].to_vec())
        }
        Type::Bytes | Type::String => {
            let n = at(data, pos)?.to::<usize>();
            if pos + WORD + n > data.len() {
                return Err(CodecError("truncated bytes".into()));
            }
            let raw = data[pos + WORD..pos + WORD + n].to_vec();
            match t {
                Type::String => Value::String(
                    String::from_utf8(raw).map_err(|e| CodecError(e.to_string()))?,
                ),
                _ => Value::Bytes(raw),
            }
        }
        Type::Array(inner, size) => {
            let (count, base) = match size {
                Some(size) => (*size, pos),
                None => (at(data, pos)?.to::<usize>(), pos + WORD),
            };
            let types: Vec<Type> = vec![(**inner).clone(); count];
            Value::Array(decode_tuple(&types, data, base)?)
        }
        Type::Tuple(members) => Value::Tuple(decode_tuple(members, data, pos)?),
    })
}

pub fn decode_tuple(types: &[Type], data: &[u8], base: usize) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(types.len());
    let mut pos = base;
    for t in types {
        if is_dynamic(t) {
            let offset = at(data, pos)?.to::<usize>();
            out.push(dec_value(t, data, base + offset)?);
        } else {
            out.push(dec_value(t, data, pos)?);
        }
        pos += head_size(t);
    }
    Ok(out)
}

pub fn decode(types: &[String], data: &[u8]) -> Result<Vec<Value>> {
    let parsed: Result<Vec<Type>> = types.iter().map(|t| parse_type(t)).collect();
    decode_tuple(&parsed?, data, 0)
}

// ------------------------------------------------------------ signatures

/// First 4 bytes of keccak256 of a canonical signature.
pub fn selector(signature: &str) -> [u8; 4] {
    let full = keccak256(signature.as_bytes());
    [full[0], full[1], full[2], full[3]]
}

/// The argument types of a canonical signature.
pub fn signature_types(signature: &str) -> Result<Vec<String>> {
    let open = signature
        .find('(')
        .ok_or_else(|| CodecError(format!("not a signature: {signature:?}")))?;
    let close = signature
        .rfind(')')
        .ok_or_else(|| CodecError(format!("not a signature: {signature:?}")))?;
    let body = &signature[open + 1..close];
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    Ok(split_top(body))
}

pub fn encode_call(signature: &str, values: &[Value]) -> Result<Vec<u8>> {
    let mut out = selector(signature).to_vec();
    out.extend(encode(&signature_types(signature)?, values)?);
    Ok(out)
}

/// A `uint` value from its decimal spelling.
///
/// Parsing lives here rather than in each binding for the reason the pool
/// models do -- one spelling of the refusal, and no binding has to name `U256`
/// to hold a value that does not fit an `f64`.
pub fn parse_uint(text: &str) -> Result<Value> {
    Ok(Value::Uint(text.parse::<U256>().map_err(|_| {
        CodecError(format!("not a u256: {text}"))
    })?))
}

/// Decode a single uint256 return value.
///
/// Refuses empty data rather than returning 0. A Curve pool that does not
/// implement a function returns *empty* data instead of reverting, and reading
/// that as zero would silently quote every swap at nothing.
pub fn decode_uint(data: &[u8]) -> Result<U256> {
    if data.len() < WORD {
        return Err(CodecError(format!(
            "expected a 32-byte word, got {} bytes",
            data.len()
        )));
    }
    Ok(U256::from_be_slice(&data[..WORD]))
}

/// The same, spelled the way the bindings carry it.
pub fn decode_uint_str(data: &[u8]) -> Result<String> {
    Ok(decode_uint(data)?.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn a_static_call_encodes_head_only() {
        let got = encode_call(
            "get_dy(int128,int128,uint256)",
            &[Value::Int(0), Value::Int(1), Value::Uint(U256::from(1000u32))],
        )
        .unwrap();
        assert_eq!(hex(&got[..4]), "5e0d443f");
        assert_eq!(got.len(), 4 + 3 * WORD);
        assert_eq!(at(&got[4..], 2 * WORD).unwrap(), U256::from(1000u32));
    }

    #[test]
    fn a_negative_int_is_twos_complement() {
        let got = encode(&["int128".to_string()], &[Value::Int(-1)]).unwrap();
        assert_eq!(hex(&got), "f".repeat(64));
        let back = decode(&["int128".to_string()], &got).unwrap();
        assert_eq!(back, vec![Value::Int(-1)]);
    }

    #[test]
    fn a_dynamic_array_carries_its_offset_and_length() {
        let got = encode(
            &["uint256[]".to_string()],
            &[Value::Array(vec![Value::Uint(U256::from(7u8)), Value::Uint(U256::from(9u8))])],
        )
        .unwrap();
        // offset 0x20, length 2, then the items.
        assert_eq!(got.len(), 4 * WORD);
        assert_eq!(at(&got, 0).unwrap(), U256::from(32u8));
        assert_eq!(at(&got, WORD).unwrap(), U256::from(2u8));
        assert_eq!(decode(&["uint256[]".to_string()], &got).unwrap()[0],
                   Value::Array(vec![Value::Uint(U256::from(7u8)),
                                     Value::Uint(U256::from(9u8))]));
    }

    #[test]
    fn a_fixed_array_has_no_length_word() {
        let got = encode(
            &["uint256[2]".to_string()],
            &[Value::Array(vec![Value::Uint(U256::from(7u8)), Value::Uint(U256::from(9u8))])],
        )
        .unwrap();
        assert_eq!(got.len(), 2 * WORD);
        assert!(encode(
            &["uint256[3]".to_string()],
            &[Value::Array(vec![Value::Uint(U256::from(7u8))])]
        )
        .is_err());
    }

    #[test]
    fn addresses_round_trip_lowercased() {
        let weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2";
        let got = encode(&["address".to_string()], &[Value::Address(weth.into())]).unwrap();
        assert_eq!(decode(&["address".to_string()], &got).unwrap(),
                   vec![Value::Address(weth.to_string())]);
        // A checksummed address encodes the same way.
        let mixed = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
        assert_eq!(
            encode(&["address".to_string()], &[Value::Address(mixed.into())]).unwrap(),
            got
        );
    }

    #[test]
    fn bytes_and_string_pad_to_a_word() {
        let got = encode(&["bytes".to_string()], &[Value::Bytes(vec![1, 2, 3])]).unwrap();
        assert_eq!(got.len(), 3 * WORD); // offset, length, one padded word
        let got = encode(&["string".to_string()], &[Value::String("hi".into())]).unwrap();
        assert_eq!(decode(&["string".to_string()], &got).unwrap(),
                   vec![Value::String("hi".into())]);
        // Exactly one word of payload needs no padding word beyond it.
        let got = encode(&["bytes".to_string()], &[Value::Bytes(vec![0u8; 32])]).unwrap();
        assert_eq!(got.len(), 3 * WORD);
    }

    #[test]
    fn a_tuple_is_dynamic_if_any_member_is() {
        assert!(!is_dynamic(&parse_type("(uint256,bool)").unwrap()));
        assert!(is_dynamic(&parse_type("(uint256,bytes)").unwrap()));
        assert!(is_dynamic(&parse_type("uint256[]").unwrap()));
        assert!(!is_dynamic(&parse_type("uint256[2]").unwrap()));
        assert!(is_dynamic(&parse_type("bytes[2]").unwrap()));
        assert_eq!(head_size(&parse_type("(uint256,bool)").unwrap()), 64);
        assert_eq!(head_size(&parse_type("uint256[3]").unwrap()), 96);
    }

    #[test]
    fn an_out_of_range_value_is_refused() {
        assert!(encode(&["uint8".to_string()], &[Value::Uint(U256::from(256u32))]).is_err());
        assert!(encode(&["uint8".to_string()], &[Value::Uint(U256::from(255u32))]).is_ok());
        assert!(encode(&["int8".to_string()], &[Value::Int(128)]).is_err());
        assert!(encode(&["int8".to_string()], &[Value::Int(-128)]).is_ok());
    }

    #[test]
    fn an_int_wider_than_this_codec_carries_is_refused_not_truncated() {
        // The one narrowing against the reference, and it says so.
        let err = parse_type("int256").unwrap_err();
        assert!(err.0.contains("wider than this codec carries"));
        assert!(parse_type("int128").is_ok());
        assert!(parse_type("uint256").is_ok());
    }

    #[test]
    fn empty_returndata_is_refused_rather_than_read_as_zero() {
        assert!(decode_uint(&[]).is_err());
        assert!(decode_uint(&[0u8; 31]).is_err());
        assert_eq!(decode_uint(&[0u8; 32]).unwrap(), U256::ZERO);
    }

    #[test]
    fn signature_types_splits_at_the_top_level() {
        assert_eq!(signature_types("f()").unwrap(), Vec::<String>::new());
        assert_eq!(
            signature_types("f(uint256,(bool,bytes),uint8[2])").unwrap(),
            vec!["uint256", "(bool,bytes)", "uint8[2]"]
        );
    }
}
