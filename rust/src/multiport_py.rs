//! Multi-port elements, across the PyO3 boundary.
//!
//! The shape rules are here; the arithmetic is on `Pools`, because that is
//! where the models live. An element is admissible or it is not, and the only
//! way to hold one is to have constructed it -- so this class *is* the check.

use crate::multiport::{self, MultiPort, MultiPortError, BPS, LP};
use crate::types::ArcKind;
use pyo3::prelude::*;

fn err(e: MultiPortError) -> PyErr {
    // The reference raises `MultiPortError`, which is a plain `Exception`.
    // `ValueError` is the nearest built-in and is what every other refusal in
    // these bindings uses.
    pyo3::exceptions::PyValueError::new_err(e.0)
}

fn kind_of(code: u8) -> PyResult<ArcKind> {
    ArcKind::from_code(code)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no such kind: {code}")))
}

/// `inputs -> outputs` through one pool, priced on advancing state.
#[pyclass]
pub struct Element {
    pub(crate) inner: MultiPort,
}

#[pymethods]
impl Element {
    /// Ports cross as `(coin, bps)` pairs, `LP` for the LP token.
    #[new]
    fn new(
        pool: &str,
        n_coins: i32,
        inputs: Vec<(i32, i64)>,
        outputs: Vec<(i32, i64)>,
    ) -> PyResult<Self> {
        let inner = MultiPort::new(pool, n_coins, ports(&inputs), ports(&outputs)).map_err(err)?;
        Ok(Element { inner })
    }

    /// The element `(kind, i, j)` triples on one pool form, or why they do not.
    #[staticmethod]
    fn from_triples(
        pool: &str,
        n_coins: i32,
        kinds: Vec<u8>,
        i: Vec<i32>,
        j: Vec<i32>,
    ) -> PyResult<Element> {
        if kinds.len() != i.len() || kinds.len() != j.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "kinds, i and j must be the same length",
            ));
        }
        let mut triples = Vec::with_capacity(kinds.len());
        for k in 0..kinds.len() {
            triples.push((kind_of(kinds[k])?, i[k], j[k]));
        }
        let inner = multiport::element_from(pool, n_coins, &triples).map_err(err)?;
        Ok(Element { inner })
    }

    /// `(input port, output port)` for one leg, `LP` for the LP token.
    #[staticmethod]
    fn ports_of(kind: u8, i: i32, j: i32) -> PyResult<(i32, i32)> {
        multiport::ports_of(kind_of(kind)?, i, j).map_err(err)
    }

    #[getter]
    fn pool(&self) -> String {
        self.inner.pool.clone()
    }

    #[getter]
    fn n_coins(&self) -> i32 {
        self.inner.n_coins
    }

    #[getter]
    fn inputs(&self) -> Vec<(i32, i64)> {
        flat(&self.inner.inputs)
    }

    #[getter]
    fn outputs(&self) -> Vec<(i32, i64)> {
        flat(&self.inner.outputs)
    }

    #[getter]
    fn ports(&self) -> usize {
        self.inner.ports()
    }

    /// A port on the LP token rather than on one of the pool's coins.
    #[staticmethod]
    fn lp() -> i32 {
        LP
    }

    #[staticmethod]
    fn bps() -> i64 {
        BPS
    }
}

fn ports(flat: &[(i32, i64)]) -> Vec<multiport::Port> {
    flat.iter().map(|&(coin, bps)| multiport::Port::new(coin, bps)).collect()
}

fn flat(ports: &[multiport::Port]) -> Vec<(i32, i64)> {
    ports.iter().map(|p| (p.coin, p.bps)).collect()
}
