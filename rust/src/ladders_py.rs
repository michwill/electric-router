//! The resident ladders, across the PyO3 boundary.
//!
//! Three crossings a refine, where the reference made none and did the work
//! instead: ask what is missing, hand back what the chain said, take the fits.
//! The ladders themselves never cross, which is the whole point -- see
//! `ladders.rs`.

use crate::ladders::{Ladders as Inner, Meta, Plan};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

type Fit = (f64, f64, f64, bool, bool, &'static str, f64, f64, f64);

#[pyclass]
pub struct Ladders {
    inner: Inner,
}

#[pymethods]
impl Ladders {
    #[new]
    fn new() -> Self {
        Self { inner: Inner::new() }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Register one arc's coarse ladder. Returns its slot.
    fn add(&mut self, decimals_in: u32, decimals_out: u32, reserve_in: u128,
           deltas: Vec<u128>, quotes: Vec<u128>, attempted: u32) -> PyResult<usize> {
        if deltas.len() != quotes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "deltas and quotes must be the same length"));
        }
        Ok(self.inner.add(Meta { decimals_in, decimals_out, reserve_in },
                          deltas, quotes, attempted))
    }

    /// A quote's own copy of the warm ladders.
    fn fork(&self) -> Ladders {
        Ladders { inner: self.inner.fork() }
    }

    /// What is still missing, as `(slots, deltas)` -- one entry per probe.
    ///
    /// `want` is flat and `spans` bounds each slot's sizes within it, so a
    /// ragged input crosses as two arrays rather than a list of lists.
    fn plan_sized<'py>(&self, py: Python<'py>, slots: Vec<u32>, want: Vec<u128>,
                       spans: Vec<u32>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        if spans.len() != slots.len() + 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "spans must bound every slot: len(slots) + 1"));
        }
        let plan = self.inner.plan_sized(&slots, &want, &spans);
        Ok((PyList::new(py, plan.slot)?, PyList::new(py, plan.delta)?))
    }

    /// Fold what the chain answered into the ladders.
    ///
    /// `status` is 0 for a value and otherwise an index into `names`, so a
    /// refusal keeps the name the quoter gave it without a string per probe.
    fn absorb(&mut self, slots: Vec<u32>, deltas: Vec<u128>, values: Vec<u128>,
              status: Vec<u8>, names: Vec<String>) -> PyResult<()> {
        let plan = Plan { slot: slots, delta: deltas };
        self.inner.absorb(&plan, &values, &status, &names)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Fit the named ladders, in the caller's order.
    fn recalibrate<'py>(&self, py: Python<'py>, slots: Vec<u32>, drift_tol: f64)
        -> PyResult<Bound<'py, PyList>> {
        let out: Vec<Option<Fit>> = self.inner.recalibrate(&slots, drift_tol)
            .into_iter()
            .map(|f| f.map(|f| (f.a, f.b, f.cap, f.clamped, f.convex_flag,
                                f.flag.as_str(), f.drift, f.eta, f.calib_delta)))
            .collect();
        PyList::new(py, out)
    }

    /// One ladder's measurements, for a caller that reports rather than fits.
    fn points<'py>(&self, py: Python<'py>, slot: usize)
        -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let deltas = self.inner.deltas_of(slot).unwrap_or(&[]);
        let quotes = self.inner.quotes_of(slot).unwrap_or(&[]);
        Ok((PyList::new(py, deltas)?, PyList::new(py, quotes)?))
    }

    fn attempted(&self, slot: usize) -> u32 {
        self.inner.attempted_of(slot)
    }

    fn failures<'py>(&self, py: Python<'py>, slot: usize) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        if let Some(got) = self.inner.failures_of(slot) {
            for (name, count) in got {
                out.set_item(name, count)?;
            }
        }
        Ok(out)
    }
}
