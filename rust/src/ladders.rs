//! The refine stage's ladders, resident for the whole stage.
//!
//! Not a faster `plan_sized`. The point is that the ladders stop moving: they
//! are handed over once, and `plan_sized`, `collect`, `merge` and the fit then
//! read them in place. `Ladder.as_float` does not get quicker, it stops
//! existing -- and so does the second conversion behind it, because
//! `calibrate_many` was rebuilding every list as `float` after `as_float` had
//! already divided it.
//!
//! What stays on the Python side is identity: the pool address, the arc kind,
//! the coin indices. Those are what a `Probe` is addressed by and they are not
//! arithmetic. What lives here is the numbers -- deltas, quotes, and the
//! decimals needed to read them as human units.
//!
//! A quote forks the warm ladders rather than rebuilding them, which is the
//! same thing `pipeline` was doing with `copy.copy` per arc and a fresh dict
//! for the failures.

use crate::calibrate::{calibrate, Calibration};
use std::collections::HashMap;

/// What a ladder needs to know about its arc to read its own numbers.
#[derive(Clone)]
pub struct Meta {
    pub decimals_in: u32,
    pub decimals_out: u32,
    pub reserve_in: u128,
}

/// One batch of probes to ask for: which ladder each belongs to, and at what
/// size. Flat rather than ragged because the caller sends it as one list.
#[derive(Default)]
pub struct Plan {
    pub slot: Vec<u32>,
    pub delta: Vec<u128>,
}

impl Plan {
    pub fn len(&self) -> usize {
        self.slot.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slot.is_empty()
    }
}

#[derive(Clone, Default)]
pub struct Ladders {
    meta: Vec<Meta>,
    deltas: Vec<Vec<u128>>,
    quotes: Vec<Vec<u128>>,
    attempted: Vec<u32>,
    /// Failure counts by status name, for the report rather than the fit.
    failures: Vec<HashMap<String, u32>>,
    /// Which ladders gained a point since the last fit, so `recalibrate` can
    /// re-fit those and leave the rest alone.
    dirty: Vec<bool>,
}

impl Ladders {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// Register one arc's coarse ladder. Returns its slot.
    pub fn add(&mut self, meta: Meta, deltas: Vec<u128>, quotes: Vec<u128>,
               attempted: u32) -> usize {
        self.meta.push(meta);
        self.deltas.push(deltas);
        self.quotes.push(quotes);
        self.attempted.push(attempted);
        self.failures.push(HashMap::new());
        self.dirty.push(false);
        self.meta.len() - 1
    }

    /// A quote's own copy of the warm ladders.
    ///
    /// Every quote refines from the same coarse start, so the warm set is
    /// never mutated -- which is what `pipeline` was expressing with a
    /// `copy.copy` per arc and a fresh `failures` dict.
    pub fn fork(&self) -> Self {
        let mut out = self.clone();
        out.dirty.iter_mut().for_each(|d| *d = false);
        out
    }

    pub fn deltas_of(&self, slot: usize) -> Option<&[u128]> {
        self.deltas.get(slot).map(|v| v.as_slice())
    }

    pub fn quotes_of(&self, slot: usize) -> Option<&[u128]> {
        self.quotes.get(slot).map(|v| v.as_slice())
    }

    pub fn attempted_of(&self, slot: usize) -> u32 {
        self.attempted.get(slot).copied().unwrap_or(0)
    }

    pub fn failures_of(&self, slot: usize) -> Option<&HashMap<String, u32>> {
        self.failures.get(slot)
    }

    /// Probe explicit per-arc sizes, skipping anything already measured.
    ///
    /// `want` is ragged: `spans[k]` bounds the sizes asked for `slots[k]`. The
    /// floor is one millionth of a unit of the input token, which is what
    /// stops a size rounding to a probe of zero.
    pub fn plan_sized(&self, slots: &[u32], want: &[u128], spans: &[u32]) -> Plan {
        let mut plan = Plan::default();
        for (k, slot) in slots.iter().enumerate() {
            let s = *slot as usize;
            let Some(have) = self.deltas.get(s) else { continue };
            let (lo, hi) = (spans[k] as usize, spans[k + 1] as usize);
            let meta = &self.meta[s];
            let floor = 10u128.pow(meta.decimals_in.saturating_sub(6)).max(1);

            // Sorted and de-duplicated, then filtered against what the ladder
            // already holds -- the same order the reference produces, because
            // `plan_sized` feeds `collect` and the two are zipped.
            let mut sized: Vec<u128> = want[lo..hi].iter().map(|d| (*d).max(floor)).collect();
            sized.sort_unstable();
            sized.dedup();
            for delta in sized {
                if !have.contains(&delta) {
                    plan.slot.push(*slot);
                    plan.delta.push(delta);
                }
            }
        }
        plan
    }

    /// `collect` and `merge` in one pass, because separating them only exists
    /// to name the two halves: a probe that answered joins its ladder, and one
    /// that did not is counted rather than recorded as a zero.
    ///
    /// **A failure is never a quote of zero.** `a = 0` would read as a valid
    /// answer and NaN the fit that follows.
    pub fn absorb(&mut self, plan: &Plan, values: &[u128], status: &[u8],
                  names: &[String]) -> Result<(), String> {
        if plan.len() != values.len() || plan.len() != status.len() {
            return Err("plan, values and status must be the same length".into());
        }
        // Gathered per slot first so the merge sorts once rather than per
        // probe, which is what makes this one pass instead of `len(plan)`.
        let mut fresh: HashMap<u32, Vec<(u128, u128)>> = HashMap::new();
        for k in 0..plan.len() {
            let slot = plan.slot[k];
            let s = slot as usize;
            if s >= self.meta.len() {
                continue;
            }
            self.attempted[s] += 1;
            let ok = status[k] == 0;
            if !ok || values[k] == 0 {
                let name = if !ok {
                    names.get(status[k] as usize - 1).cloned()
                        .unwrap_or_else(|| "FAILED".to_string())
                } else {
                    "ZERO".to_string()
                };
                *self.failures[s].entry(name).or_insert(0) += 1;
                continue;
            }
            fresh.entry(slot).or_default().push((plan.delta[k], values[k]));
        }

        for (slot, mut points) in fresh {
            let s = slot as usize;
            // The reference merges through a dict, so a repeated delta takes
            // the newer quote; then the pairs are sorted by size.
            let mut merged: HashMap<u128, u128> =
                self.deltas[s].iter().copied().zip(self.quotes[s].iter().copied()).collect();
            points.sort_unstable();
            for (d, q) in points {
                merged.insert(d, q);
            }
            let mut pairs: Vec<(u128, u128)> = merged.into_iter().collect();
            pairs.sort_unstable();
            self.deltas[s] = pairs.iter().map(|(d, _)| *d).collect();
            self.quotes[s] = pairs.iter().map(|(_, q)| *q).collect();
            self.dirty[s] = true;
        }
        Ok(())
    }

    /// Fit every ladder with enough points, reading the numbers in place.
    ///
    /// `slots` is the caller's order -- the arcs it will write the answers
    /// back onto -- and the answers come back in it, `None` where the ladder
    /// was too short or the fit refused.
    ///
    /// The scaling to human units happens here, per point, rather than through
    /// a list built for the purpose. That list is the whole reason this exists.
    pub fn recalibrate(&self, slots: &[u32], drift_tol: f64)
        -> Vec<Option<Calibration>> {
        slots.iter().map(|slot| {
            let s = *slot as usize;
            if s >= self.meta.len() || self.deltas[s].len() < 3 {
                return None;
            }
            let meta = &self.meta[s];
            let scale_in = 10f64.powi(meta.decimals_in as i32);
            let scale_out = 10f64.powi(meta.decimals_out as i32);
            let deltas: Vec<f64> =
                self.deltas[s].iter().map(|d| *d as f64 / scale_in).collect();
            let quotes: Vec<f64> =
                self.quotes[s].iter().map(|q| *q as f64 / scale_out).collect();
            let quantum = 10f64.powi(-(meta.decimals_out as i32));
            calibrate(&deltas, &quotes, None, false, drift_tol, None, None, quantum).ok()
        }).collect()
    }
}
