//! Calldata, across the PyO3 boundary.
//!
//! The whole call comes back as one object: `encode_route` is the last step,
//! and what a caller wants from it is the bytes plus the two numbers that say
//! what those bytes promise.

use crate::realize_py::Route;
use crate::routecall::{self, Encode, Naming, Policy, RouteCall as Inner};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use ruint::aliases::U256;

fn err(e: routecall::EncodingError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.0)
}

fn u256(text: &str) -> PyResult<U256> {
    text.parse()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("not a u256: {text}")))
}

/// A ready-to-send call, and what it is and is not protecting.
#[pyclass]
pub struct RouteCall {
    inner: Inner,
}

#[pymethods]
impl RouteCall {
    /// Encode `route` for `ElectricRouter.execute`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (route, *, receiver, set_approvals=true, min_out="0",
                        amount_in=None, quoted_out=None, volatile=None,
                        naming="needed", allow_unbounded=false,
                        fee_share=routecall::FEE_SHARE,
                        floor_bp=routecall::FLOOR_BP,
                        volatile_floor_bp=routecall::VOLATILE_FLOOR_BP,
                        slippage_bp=None))]
    fn encode_route(
        route: PyRef<'_, Route>, receiver: &str, set_approvals: bool, min_out: &str,
        amount_in: Option<&str>, quoted_out: Option<&str>, volatile: Option<Vec<String>>,
        naming: &str, allow_unbounded: bool, fee_share: f64, floor_bp: f64,
        volatile_floor_bp: f64, slippage_bp: Option<f64>,
    ) -> PyResult<RouteCall> {
        let naming = Naming::parse(naming).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "naming must be one of needed/none/all, got {naming:?}"
            ))
        })?;
        let loose = volatile.unwrap_or_default();
        let opts = Encode {
            receiver,
            set_approvals,
            min_out: u256(min_out)?,
            amount_in: amount_in.map(u256).transpose()?,
            quoted_out: quoted_out.map(u256).transpose()?,
            naming,
            allow_unbounded,
            policy: Policy {
                volatile: &loose,
                fee_share,
                floor_bp,
                volatile_floor_bp,
                slippage_bp,
            },
        };
        Ok(RouteCall { inner: routecall::encode_route(&route.inner, &opts).map_err(err)? })
    }

    /// The shortest entry point that still expresses this call.
    #[pyo3(signature = (sender=""))]
    fn calldata(&self, py: Python<'_>, sender: &str) -> PyResult<Py<PyBytes>> {
        let out = self.inner.calldata(sender).map_err(err)?;
        Ok(PyBytes::new(py, &out).unbind())
    }

    #[getter]
    fn amount_in(&self) -> String {
        self.inner.amount_in.to_string()
    }

    #[getter]
    fn pools(&self) -> Vec<String> {
        self.inner.pools.clone()
    }

    #[getter]
    fn params(&self) -> Vec<String> {
        self.inner.params.iter().map(|p| p.to_string()).collect()
    }

    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    #[getter]
    fn receiver(&self) -> String {
        self.inner.receiver.clone()
    }

    #[getter]
    fn min_out(&self) -> String {
        self.inner.min_out.to_string()
    }

    #[getter]
    fn token_in(&self) -> String {
        self.inner.token_in.clone()
    }

    #[getter]
    fn token_out(&self) -> String {
        self.inner.token_out.clone()
    }

    #[getter]
    fn guaranteed_out(&self) -> String {
        self.inner.guaranteed_out.to_string()
    }

    #[getter]
    fn quoted_out(&self) -> String {
        self.inner.quoted_out.to_string()
    }

    #[getter]
    fn unbounded(&self) -> Vec<usize> {
        self.inner.unbounded.clone()
    }

    #[getter]
    fn tolerance_bp(&self) -> f64 {
        self.inner.tolerance_bp()
    }

    /// `(pool, kind, i, j, n, frac, min_rate, in_ref, out_ref)` per leg.
    #[allow(clippy::type_complexity)]
    fn steps(&self) -> PyResult<Vec<(String, u8, i32, i32, i32, String, String, usize, usize)>> {
        Ok(self
            .inner
            .steps()
            .map_err(err)?
            .iter()
            .map(|s| {
                (s.pool.clone(), s.kind.code(), s.i, s.j, s.n,
                 s.frac.to_string(), s.min_rate.to_string(), s.in_ref, s.out_ref)
            })
            .collect())
    }
}

/// Each leg's share of the balance standing at its source when it runs.
#[pyfunction]
pub fn fractions(route: PyRef<'_, Route>) -> PyResult<Vec<String>> {
    Ok(routecall::fractions(&route.inner)
        .map_err(err)?
        .iter()
        .map(|v| v.to_string())
        .collect())
}

/// `(min_rate per leg, indices the bound does not really cover)`.
#[pyfunction]
#[pyo3(signature = (route, *, volatile=None, fee_share=routecall::FEE_SHARE,
                    floor_bp=routecall::FLOOR_BP,
                    volatile_floor_bp=routecall::VOLATILE_FLOOR_BP,
                    slippage_bp=None))]
pub fn min_rates(
    route: PyRef<'_, Route>, volatile: Option<Vec<String>>, fee_share: f64,
    floor_bp: f64, volatile_floor_bp: f64, slippage_bp: Option<f64>,
) -> PyResult<(Vec<String>, Vec<usize>)> {
    let loose = volatile.unwrap_or_default();
    let policy = Policy {
        volatile: &loose, fee_share, floor_bp, volatile_floor_bp, slippage_bp,
    };
    let (rates, unbounded) = routecall::min_rates(&route.inner, &policy).map_err(err)?;
    Ok((rates.iter().map(|v| v.to_string()).collect(), unbounded))
}

/// How far below its quote the automatic rule lets each leg land.
#[pyfunction]
#[pyo3(signature = (route, *, volatile=None, fee_share=routecall::FEE_SHARE,
                    floor_bp=routecall::FLOOR_BP,
                    volatile_floor_bp=routecall::VOLATILE_FLOOR_BP))]
pub fn tolerances(
    route: PyRef<'_, Route>, volatile: Option<Vec<String>>, fee_share: f64,
    floor_bp: f64, volatile_floor_bp: f64,
) -> Vec<f64> {
    let loose = volatile.unwrap_or_default();
    routecall::tolerances(&route.inner, &loose, fee_share, floor_bp, volatile_floor_bp)
}

/// The room each leg needs for what moves under it, whatever it charges.
#[pyfunction]
#[pyo3(signature = (route, *, volatile=None, floor_bp=routecall::FLOOR_BP,
                    volatile_floor_bp=routecall::VOLATILE_FLOOR_BP))]
pub fn movement_floors(
    route: PyRef<'_, Route>, volatile: Option<Vec<String>>, floor_bp: f64,
    volatile_floor_bp: f64,
) -> Vec<f64> {
    let loose = volatile.unwrap_or_default();
    routecall::movement_floors(&route.inner, &loose, floor_bp, volatile_floor_bp)
}

/// `(what the bounds promise, the minimum output each leg enforces)`.
#[pyfunction]
pub fn walk_bounds(
    route: PyRef<'_, Route>, fracs: Vec<String>, rates: Vec<String>,
) -> PyResult<(String, Vec<String>)> {
    let fracs: PyResult<Vec<U256>> = fracs.iter().map(|v| u256(v)).collect();
    let rates: PyResult<Vec<U256>> = rates.iter().map(|v| u256(v)).collect();
    let (promised, floors) = routecall::walk_bounds(&route.inner, &fracs?, &rates?);
    Ok((promised.to_string(), floors.iter().map(|v| v.to_string()).collect()))
}

/// One packed leg, as a word.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pack_step(
    kind: u8, i: i32, j: i32, n: i32, frac: &str, min_rate: &str, in_ref: usize,
    out_ref: usize,
) -> PyResult<String> {
    let kind = crate::types::ArcKind::from_code(kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("no such kind: {kind}"))
    })?;
    let step = routecall::Step {
        pool: String::new(), kind, i, j, n,
        frac: u256(frac)?, min_rate: u256(min_rate)?, in_ref, out_ref,
    };
    Ok(step.pack().map_err(err)?.to_string())
}

/// The inverse, refusing a word with reserved bits set.
#[pyfunction]
pub fn unpack_step(word: &str) -> PyResult<(u8, i32, i32, i32, String, String, usize, usize)> {
    let step = routecall::unpack(u256(word)?, "").map_err(err)?;
    Ok((step.kind.code(), step.i, step.j, step.n, step.frac.to_string(),
        step.min_rate.to_string(), step.in_ref, step.out_ref))
}
