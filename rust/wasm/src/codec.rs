//! The ABI codec, in the browser.
//!
//! The twin of `src/codec_py.rs`. Values cross as a JSON-ish tagged pair --
//! `["uint", "1000"]` -- because a browser has no tuple type a typed array can
//! carry, and a `uint256` has no JS number that holds it.

use erouter_solve::codec::{self, Type, Value};
use wasm_bindgen::prelude::*;

fn err(e: codec::CodecError) -> JsValue {
    JsError::new(&e.0).into()
}

fn tagged(value: &Value) -> js_sys::Array {
    let pair = |tag: &str, inner: JsValue| js_sys::Array::of2(&JsValue::from_str(tag), &inner);
    match value {
        Value::Uint(n) => pair("uint", JsValue::from_str(&n.to_string())),
        Value::Int(n) => pair("int", JsValue::from_str(&n.to_string())),
        Value::Address(a) => pair("address", JsValue::from_str(a)),
        Value::Bool(b) => pair("bool", JsValue::from_bool(*b)),
        Value::Bytes(b) => {
            pair("bytes", js_sys::Uint8Array::from(b.as_slice()).into())
        }
        Value::String(s) => pair("string", JsValue::from_str(s)),
        Value::Array(items) | Value::Tuple(items) => {
            let inner = js_sys::Array::new();
            for one in items {
                inner.push(&tagged(one));
            }
            let tag = if matches!(value, Value::Array(_)) { "array" } else { "tuple" };
            pair(tag, inner.into())
        }
    }
}

fn untagged(value: &JsValue) -> Result<Value, JsValue> {
    let pair = js_sys::Array::from(value);
    if pair.length() != 2 {
        return Err(JsError::new("a value is a [tag, payload] pair").into());
    }
    let tag = pair.get(0).as_string().ok_or_else(|| JsError::new("tag must be a string"))?;
    let inner = pair.get(1);
    let text = || inner.as_string().ok_or_else(|| JsError::new("expected a string"));
    Ok(match tag.as_str() {
        "uint" => codec::parse_uint(&text()?).map_err(err)?,
        "int" => {
            let raw = text()?;
            Value::Int(raw.parse::<i128>().map_err(|_| {
                JsError::new(&format!("not an i128: {raw}"))
            })?)
        }
        "address" => Value::Address(text()?),
        "bool" => Value::Bool(inner.as_bool().unwrap_or(false)),
        "bytes" => Value::Bytes(js_sys::Uint8Array::new(&inner).to_vec()),
        "string" => Value::String(text()?),
        "array" | "tuple" => {
            let items = js_sys::Array::from(&inner);
            let mut parsed = Vec::with_capacity(items.length() as usize);
            for k in 0..items.length() {
                parsed.push(untagged(&items.get(k))?);
            }
            if tag == "array" { Value::Array(parsed) } else { Value::Tuple(parsed) }
        }
        _ => return Err(JsError::new(&format!("unknown value tag: {tag}")).into()),
    })
}

fn parsed(types: &[String]) -> Result<Vec<Type>, JsValue> {
    types.iter().map(|t| codec::parse_type(t).map_err(err)).collect()
}

/// keccak-256 -- Ethereum's, with `0x01` padding, not SHA-3's.
#[wasm_bindgen]
pub fn keccak256(data: &[u8]) -> Vec<u8> {
    erouter_solve::keccak::keccak256(data).to_vec()
}

/// First four bytes of the hash of a canonical signature.
#[wasm_bindgen]
pub fn selector(signature: &str) -> Vec<u8> {
    codec::selector(signature).to_vec()
}

/// The argument types of a canonical signature.
#[wasm_bindgen(js_name = signatureTypes)]
pub fn signature_types(signature: &str) -> Result<Vec<String>, JsValue> {
    codec::signature_types(signature).map_err(err)
}

/// Whether a type's encoding carries an offset rather than its value.
#[wasm_bindgen(js_name = isDynamic)]
pub fn is_dynamic(text: &str) -> Result<bool, JsValue> {
    Ok(codec::is_dynamic(&codec::parse_type(text).map_err(err)?))
}

/// Bytes this type occupies in a head region.
#[wasm_bindgen(js_name = headSize)]
pub fn head_size(text: &str) -> Result<usize, JsValue> {
    Ok(codec::head_size(&codec::parse_type(text).map_err(err)?))
}

#[wasm_bindgen(js_name = abiEncode)]
pub fn abi_encode(types: Vec<String>, values: Vec<JsValue>) -> Result<Vec<u8>, JsValue> {
    let parsed_values: Result<Vec<Value>, JsValue> =
        values.iter().map(untagged).collect();
    codec::encode_tuple(&parsed(&types)?, &parsed_values?).map_err(err)
}

#[wasm_bindgen(js_name = abiDecode)]
pub fn abi_decode(types: Vec<String>, data: &[u8]) -> Result<Vec<JsValue>, JsValue> {
    let decoded = codec::decode_tuple(&parsed(&types)?, data, 0).map_err(err)?;
    Ok(decoded.iter().map(|v| tagged(v).into()).collect())
}

/// Selector plus arguments -- the artefact a node is actually sent.
#[wasm_bindgen(js_name = encodeCall)]
pub fn encode_call(signature: &str, values: Vec<JsValue>) -> Result<Vec<u8>, JsValue> {
    let parsed_values: Result<Vec<Value>, JsValue> =
        values.iter().map(untagged).collect();
    codec::encode_call(signature, &parsed_values?).map_err(err)
}

/// Decode a single uint256 return value, refusing empty returndata.
#[wasm_bindgen(js_name = decodeUint)]
pub fn decode_uint(data: &[u8]) -> Result<String, JsValue> {
    codec::decode_uint_str(data).map_err(err)
}
