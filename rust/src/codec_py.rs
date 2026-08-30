//! The ABI codec, across the PyO3 boundary.
//!
//! Values cross as a small tagged form -- `("uint", "1000")`, `("int", -1)` --
//! because Rust is statically typed and the reference's `Any` is not. The tag
//! is what the two sides agree on, and a `uint256` crosses as a decimal string
//! for the same reason it does everywhere else in these bindings.

use crate::codec::{self, Type, Value};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use ruint::aliases::U256;

fn err(e: codec::CodecError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.0)
}

fn tagged(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let pair = |tag: &str, inner: Py<PyAny>| -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, [tag.into_pyobject(py)?.into_any().unbind(), inner])?
            .into_any()
            .unbind())
    };
    Ok(match value {
        Value::Uint(n) => pair("uint", n.to_string().into_pyobject(py)?.into_any().unbind())?,
        Value::Int(n) => pair("int", n.into_pyobject(py)?.into_any().unbind())?,
        Value::Address(a) => pair("address", a.into_pyobject(py)?.into_any().unbind())?,
        Value::Bool(b) => {
            pair("bool", b.into_pyobject(py)?.to_owned().into_any().unbind())?
        }
        Value::Bytes(b) => pair("bytes", PyBytes::new(py, b).into_any().unbind())?,
        Value::String(s) => pair("string", s.into_pyobject(py)?.into_any().unbind())?,
        Value::Array(items) | Value::Tuple(items) => {
            let inner: PyResult<Vec<Py<PyAny>>> =
                items.iter().map(|one| tagged(py, one)).collect();
            let tag = if matches!(value, Value::Array(_)) { "array" } else { "tuple" };
            pair(tag, inner?.into_pyobject(py)?.into_any().unbind())?
        }
    })
}

fn untagged(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let (tag, inner): (String, Bound<'_, PyAny>) = value.extract()?;
    Ok(match tag.as_str() {
        "uint" => {
            let text: String = inner.extract()?;
            Value::Uint(text.parse::<U256>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!("not a u256: {text}"))
            })?)
        }
        "int" => Value::Int(inner.extract()?),
        "address" => Value::Address(inner.extract()?),
        "bool" => Value::Bool(inner.extract()?),
        "bytes" => Value::Bytes(inner.extract()?),
        "string" => Value::String(inner.extract()?),
        "array" | "tuple" => {
            let items: Vec<Bound<'_, PyAny>> = inner.extract()?;
            let parsed: PyResult<Vec<Value>> = items.iter().map(untagged).collect();
            if tag == "array" {
                Value::Array(parsed?)
            } else {
                Value::Tuple(parsed?)
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown value tag: {tag}"
            )))
        }
    })
}

fn parsed(types: &[String]) -> PyResult<Vec<Type>> {
    types.iter().map(|t| codec::parse_type(t).map_err(err)).collect()
}

/// keccak-256 -- Ethereum's, with `0x01` padding, not SHA-3's.
#[pyfunction]
pub fn keccak256(py: Python<'_>, data: &[u8]) -> Py<PyBytes> {
    PyBytes::new(py, &crate::keccak::keccak256(data)).unbind()
}

/// First four bytes of the hash of a canonical signature.
#[pyfunction]
pub fn selector(py: Python<'_>, signature: &str) -> Py<PyBytes> {
    PyBytes::new(py, &codec::selector(signature)).unbind()
}

/// The argument types of a canonical signature.
#[pyfunction]
pub fn signature_types(signature: &str) -> PyResult<Vec<String>> {
    codec::signature_types(signature).map_err(err)
}

/// Whether a type's encoding carries an offset rather than its value.
#[pyfunction]
pub fn is_dynamic(text: &str) -> PyResult<bool> {
    Ok(codec::is_dynamic(&codec::parse_type(text).map_err(err)?))
}

/// Bytes this type occupies in a head region.
#[pyfunction]
pub fn head_size(text: &str) -> PyResult<usize> {
    Ok(codec::head_size(&codec::parse_type(text).map_err(err)?))
}

#[pyfunction]
pub fn abi_encode(
    py: Python<'_>, types: Vec<String>, values: Vec<Bound<'_, PyAny>>,
) -> PyResult<Py<PyBytes>> {
    let parsed_values: PyResult<Vec<Value>> = values.iter().map(untagged).collect();
    let out = codec::encode_tuple(&parsed(&types)?, &parsed_values?).map_err(err)?;
    Ok(PyBytes::new(py, &out).unbind())
}

#[pyfunction]
pub fn abi_decode(
    py: Python<'_>, types: Vec<String>, data: &[u8],
) -> PyResult<Vec<Py<PyAny>>> {
    let decoded = codec::decode_tuple(&parsed(&types)?, data, 0).map_err(err)?;
    decoded.iter().map(|v| tagged(py, v)).collect()
}

/// Selector plus arguments -- the artefact a node is actually sent.
#[pyfunction]
pub fn encode_call(
    py: Python<'_>, signature: &str, values: Vec<Bound<'_, PyAny>>,
) -> PyResult<Py<PyBytes>> {
    let parsed_values: PyResult<Vec<Value>> = values.iter().map(untagged).collect();
    let out = codec::encode_call(signature, &parsed_values?).map_err(err)?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Decode a single uint256 return value, refusing empty returndata.
#[pyfunction]
pub fn decode_uint(data: &[u8]) -> PyResult<String> {
    Ok(codec::decode_uint(data).map_err(err)?.to_string())
}
