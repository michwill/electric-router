//! Sparse LDL^T for a symmetric positive definite matrix whose pattern is fixed.
//!
//! The circuit's Hessian is a weighted Laplacian of the token graph: its
//! pattern is the graph's node adjacency and does not change between Newton
//! iterations, only its values do. So, as SPICE does, the ordering and the
//! elimination tree are worked out once (`Symbolic`) and each iteration only
//! refactors the numbers (`Symbolic::factor`). A handful of hub tokens touch
//! almost everything; a minimum-degree ordering eliminates the leaves first
//! and leaves the hubs as a small dense core, where a dense factorisation of
//! the whole 300-node matrix would redo ten million flops every iteration.
//!
//! Up-looking LDL^T after T. Davis, "Algorithm 849: A concise sparse Cholesky
//! factorization package" (2005): etree and column counts in the symbolic
//! pass, one row of L per step in the numeric one.

const NONE: usize = usize::MAX;

/// The ordering, the elimination tree and where each entry lives.
#[derive(Debug, Clone)]
pub struct Symbolic {
    pub n: usize,
    /// `perm[k]` is the original index eliminated k-th; `pinv` its inverse.
    perm: Vec<usize>,
    pinv: Vec<usize>,
    /// Upper triangle of the permuted matrix, compressed by column (rows <= col).
    ap: Vec<usize>,
    ai: Vec<usize>,
    parent: Vec<usize>,
    lp: Vec<usize>,
}

/// A numeric factorisation against a `Symbolic`.
#[derive(Debug, Clone)]
pub struct Factor {
    li: Vec<usize>,
    lx: Vec<f64>,
    d: Vec<f64>,
}

impl Symbolic {
    /// From the off-diagonal edges `(i, j)` of an `n x n` pattern (either
    /// order, duplicates allowed); the diagonal is always present.
    pub fn new(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(i, j) in edges {
            if i != j && i < n && j < n {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
        for row in adj.iter_mut() {
            row.sort_unstable();
            row.dedup();
        }
        let perm = minimum_degree(&adj);
        let mut pinv = vec![0usize; n];
        for (k, &v) in perm.iter().enumerate() {
            pinv[v] = k;
        }
        // Upper triangle of P A P^T, by column.
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (v, row) in adj.iter().enumerate() {
            for &w in row {
                let (a, b) = (pinv[v], pinv[w]);
                if a < b {
                    cols[b].push(a);
                }
            }
        }
        let mut ap = vec![0usize; n + 1];
        let mut ai = Vec::new();
        for (k, col) in cols.iter_mut().enumerate() {
            col.sort_unstable();
            col.dedup();
            ai.extend_from_slice(col);
            ai.push(k); // the diagonal, last in its column
            ap[k + 1] = ai.len();
        }
        // Elimination tree and column counts of L.
        let mut parent = vec![NONE; n];
        let mut flag = vec![NONE; n];
        let mut lnz = vec![0usize; n];
        for k in 0..n {
            flag[k] = k;
            for p in ap[k]..ap[k + 1] {
                let mut i = ai[p];
                if i < k {
                    while flag[i] != k {
                        if parent[i] == NONE {
                            parent[i] = k;
                        }
                        lnz[i] += 1;
                        flag[i] = k;
                        i = parent[i];
                    }
                }
            }
        }
        let mut lp = vec![0usize; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + lnz[k];
        }
        Symbolic { n, perm, pinv, ap, ai, parent, lp }
    }

    /// Where the entry `(i, j)` of the original matrix lives in the value
    /// array `factor` takes, or `None` if it is not in the pattern.
    pub fn slot(&self, i: usize, j: usize) -> Option<usize> {
        let (a, b) = (self.pinv[i], self.pinv[j]);
        let (row, col) = if a <= b { (a, b) } else { (b, a) };
        let span = &self.ai[self.ap[col]..self.ap[col + 1]];
        span.binary_search(&row).ok().map(|off| self.ap[col] + off)
    }

    /// How many values `factor` takes.
    pub fn nnz(&self) -> usize {
        self.ai.len()
    }

    /// Factor the matrix whose upper-triangle values are `ax` (indexed by
    /// `slot`). `None` if a pivot is not positive.
    pub fn factor(&self, ax: &[f64]) -> Option<Factor> {
        let n = self.n;
        let mut li = vec![0usize; self.lp[n]];
        let mut lx = vec![0.0f64; self.lp[n]];
        let mut d = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];
        let mut pattern = vec![0usize; n];
        let mut flag = vec![NONE; n];
        let mut lnz = vec![0usize; n];
        for k in 0..n {
            y[k] = 0.0;
            let mut top = n;
            flag[k] = k;
            for p in self.ap[k]..self.ap[k + 1] {
                let mut i = self.ai[p];
                y[i] += ax[p];
                let mut len = 0;
                while flag[i] != k {
                    pattern[len] = i;
                    len += 1;
                    flag[i] = k;
                    i = self.parent[i];
                }
                while len > 0 {
                    len -= 1;
                    top -= 1;
                    pattern[top] = pattern[len];
                }
            }
            d[k] = y[k];
            y[k] = 0.0;
            for &i in &pattern[top..n] {
                let yi = y[i];
                y[i] = 0.0;
                let p2 = self.lp[i] + lnz[i];
                for p in self.lp[i]..p2 {
                    y[li[p]] -= lx[p] * yi;
                }
                let lki = yi / d[i];
                d[k] -= lki * yi;
                li[p2] = k;
                lx[p2] = lki;
                lnz[i] += 1;
            }
            if !(d[k] > 0.0) {
                return None;
            }
        }
        Some(Factor { li, lx, d })
    }

    /// Solve `A x = b` in place, `b` in the original ordering.
    pub fn solve(&self, f: &Factor, b: &mut [f64]) {
        let n = self.n;
        let mut x: Vec<f64> = (0..n).map(|k| b[self.perm[k]]).collect();
        for j in 0..n {
            for p in self.lp[j]..self.lp[j + 1] {
                x[f.li[p]] -= f.lx[p] * x[j];
            }
        }
        for j in 0..n {
            x[j] /= f.d[j];
        }
        for j in (0..n).rev() {
            for p in self.lp[j]..self.lp[j + 1] {
                x[j] -= f.lx[p] * x[f.li[p]];
            }
        }
        for k in 0..n {
            b[self.perm[k]] = x[k];
        }
    }
}

/// Greedy minimum degree on the explicit elimination graph: eliminate the node
/// with the fewest neighbours, join its neighbours into a clique, repeat.
/// Ties go to the lower index, so the ordering is deterministic.
fn minimum_degree(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut graph: Vec<std::collections::BTreeSet<usize>> =
        adj.iter().map(|row| row.iter().copied().collect()).collect();
    let mut gone = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let v = (0..n)
            .filter(|&v| !gone[v])
            .min_by_key(|&v| (graph[v].len(), v))
            .expect("a node is left");
        let nbrs: Vec<usize> = graph[v].iter().copied().collect();
        for &a in &nbrs {
            graph[a].remove(&v);
            for &b in &nbrs {
                if a != b {
                    graph[a].insert(b);
                }
            }
        }
        graph[v].clear();
        gone[v] = true;
        order.push(v);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut m: Vec<Vec<f64>> = a.to_vec();
        let mut x = b.to_vec();
        for c in 0..n {
            for r in c + 1..n {
                let f = m[r][c] / m[c][c];
                for k in c..n {
                    m[r][k] -= f * m[c][k];
                }
                x[r] -= f * x[c];
            }
        }
        for c in (0..n).rev() {
            for k in c + 1..n {
                x[c] -= m[c][k] * x[k];
            }
            x[c] /= m[c][c];
        }
        x
    }

    #[test]
    fn a_star_with_a_chain_solves_like_the_dense_matrix() {
        // A hub (0) touching everything, plus a chain 1-2-3-4: the shape of a
        // token graph with WETH at the middle.
        let n = 6;
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4)];
        let sym = Symbolic::new(n, &edges);
        let mut a = vec![vec![0.0; n]; n];
        let mut w = 1.0;
        for &(i, j) in &edges {
            w += 0.37;
            a[i][i] += w;
            a[j][j] += w;
            a[i][j] -= w;
            a[j][i] -= w;
        }
        for (k, row) in a.iter_mut().enumerate() {
            row[k] += 0.1 * (k as f64 + 1.0); // grounded: positive definite
        }
        let mut ax = vec![0.0; sym.nnz()];
        for i in 0..n {
            for j in i..n {
                if a[i][j] != 0.0 {
                    ax[sym.slot(i, j).expect("in the pattern")] += a[i][j];
                }
            }
        }
        let f = sym.factor(&ax).expect("positive definite");
        let b: Vec<f64> = (0..n).map(|k| (k as f64) - 2.5).collect();
        let mut x = b.clone();
        sym.solve(&f, &mut x);
        let want = dense_solve(&a, &b);
        for (got, want) in x.iter().zip(&want) {
            assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{got} vs {want}");
        }
    }

    #[test]
    fn a_star_factors_without_fill() {
        // Eliminating the hub first would fill the whole matrix; minimum
        // degree takes the leaves first, and L keeps only the three edges.
        let sym = Symbolic::new(4, &[(0, 1), (0, 2), (0, 3)]);
        assert_eq!(sym.lp[4], 3);
    }
}
