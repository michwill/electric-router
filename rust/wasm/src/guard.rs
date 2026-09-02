//! What the bindings refuse, and why it has to be refused here.
//!
//! The twin of the helpers at the top of `src/py.rs`, and it exists for a
//! sharper reason than symmetry: **wasm32 cannot unwind**.  On CPython a panic
//! that slips through reaches the caller as an exception and the process
//! survives; here it traps the instance and poisons the module for every later
//! call. So the browser is the target where a missing check costs the most,
//! and it is the one that had none.
//!
//! Each of these mirrors a refusal the reference already makes -- numpy on a
//! shape mismatch, `IndexError` on a node that is not a node -- rather than
//! inventing a rule the Python side does not have.

use wasm_bindgen::prelude::*;

/// Refuse two arrays the caller says are parallel and are not.
pub fn same_length(what: &str, got: usize, want: usize) -> Result<(), JsValue> {
    if got == want {
        return Ok(());
    }
    Err(JsError::new(&format!("{what} has {got} value(s), expected {want}")).into())
}

/// The largest graph these bindings will size a vector for, and the largest
/// dense system for the one place a bound squares.
///
/// The twin of the constants in `src/py.rs`, and here the reason is sharper
/// still: wasm memory is a single growable buffer, so an `n_nodes` a caller
/// invented is a `memory.grow` that fails and takes the instance with it.
pub const MAX_NODES: usize = 1 << 22;
pub const MAX_DENSE: usize = 1 << 13;

/// Refuse a node count that would size an allocation nothing can serve.
pub fn node_count(n_nodes: usize) -> Result<(), JsValue> {
    if n_nodes <= MAX_NODES {
        return Ok(());
    }
    Err(JsError::new(&format!(
        "n_nodes is {n_nodes}, past the {MAX_NODES} this will allocate for"
    ))
    .into())
}

/// Refuse an arc list whose endpoints are not nodes of the graph.
pub fn arc_nodes(tau: &[i64], sig: &[i64], n_nodes: usize) -> Result<(), JsValue> {
    same_length("sig", sig.len(), tau.len())?;
    node_count(n_nodes)?;
    let n = n_nodes as i64;
    for (side, values) in [("tau", tau), ("sig", sig)] {
        for (k, &v) in values.iter().enumerate() {
            if v < 0 || v >= n {
                return Err(JsError::new(&format!(
                    "{side}[{k}] is node {v}, outside 0..{n_nodes}"
                ))
                .into());
            }
        }
    }
    Ok(())
}

/// Refuse a node index that is not a node.
pub fn node(what: &str, at: usize, n_nodes: usize) -> Result<(), JsValue> {
    if at < n_nodes {
        return Ok(());
    }
    Err(JsError::new(&format!("{what} is node {at}, outside 0..{n_nodes}")).into())
}
