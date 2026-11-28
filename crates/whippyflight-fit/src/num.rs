//! Numerics the fit shares: order statistics, least squares, rounding.

use nalgebra::{DMatrix, DVector};

/// The `q`th percentile (0–100) of `x`, linearly interpolated between order
/// statistics (numpy's default); NaN for no data.
pub fn percentile(x: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = x.to_vec();
    percentile_in_place(&mut v, q)
}

/// [`percentile`], reordering `v`.
pub fn percentile_in_place(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let frac = pos - lo as f64;
    let (_, &mut a, rest) = v.select_nth_unstable_by(lo, f64::total_cmp);
    let b = if hi == lo { a } else { rest.iter().copied().fold(f64::INFINITY, f64::min) };
    a + (b - a) * frac
}

pub fn median(x: &[f64]) -> f64 {
    percentile(x, 50.0)
}

pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Population standard deviation.
pub fn std(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Round half to even, as Python's `round`.
pub fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// `x` rounded to `digits` decimals, for reports.
pub fn round_to(x: f64, digits: i32) -> f64 {
    let s = 10f64.powi(digits);
    (x * s).round_ties_even() / s
}

/// Pearson correlation.
pub fn corrcoef(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    sab / (saa * sbb).sqrt()
}

/// Least squares `X β ≈ y`: (β, its covariance σ̂²(XᵀX)⁻¹). Columns are
/// scaled to unit norm first, so regressors of very different magnitudes
/// (rates against squared motor speeds) don't cost precision.
pub fn ols_cov(x: &DMatrix<f64>, y: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let k = x.ncols();
    let scale: Vec<f64> = (0..k).map(|j| x.column(j).norm()).map(|s| if s > 0.0 { s } else { 1.0 }).collect();
    let mut xs = x.clone();
    for (j, s) in scale.iter().enumerate() {
        xs.column_mut(j).scale_mut(1.0 / s);
    }
    let xtx = xs.tr_mul(&xs);
    let xty = xs.tr_mul(y);
    let svd = xtx.clone().svd(true, true);
    let inv = svd.pseudo_inverse(1e-14).unwrap_or_else(|_| DMatrix::zeros(k, k));
    let beta_s = &inv * xty;
    let resid = y - &xs * &beta_s;
    let dof = (y.len() as f64 - k as f64).max(1.0);
    let s2 = resid.dot(&resid) / dof;
    let mut beta = beta_s;
    let mut cov = inv * s2;
    for i in 0..k {
        beta[i] /= scale[i];
        for j in 0..k {
            cov[(i, j)] /= scale[i] * scale[j];
        }
    }
    (beta, cov)
}

/// Least squares: (β, standard errors, R²).
pub fn ols(x: &DMatrix<f64>, y: &DVector<f64>) -> (DVector<f64>, DVector<f64>, f64) {
    let (beta, cov) = ols_cov(x, y);
    let se = DVector::from_iterator(beta.len(), (0..beta.len()).map(|i| cov[(i, i)].sqrt()));
    let r = r2(x, y, &beta);
    (beta, se, r)
}

/// Just β.
pub fn lstsq(x: &DMatrix<f64>, y: &DVector<f64>) -> DVector<f64> {
    ols_cov(x, y).0
}

/// 1 − SSE/SST of `β` on the rows.
pub fn r2(x: &DMatrix<f64>, y: &DVector<f64>, beta: &DVector<f64>) -> f64 {
    let (sse, sst) = sse_sst(x, y, beta);
    1.0 - sse / sst
}

/// (Σ residual², Σ (y − ȳ)²) of `β` on the rows.
pub fn sse_sst(x: &DMatrix<f64>, y: &DVector<f64>, beta: &DVector<f64>) -> (f64, f64) {
    if y.is_empty() {
        return (0.0, 0.0);
    }
    let resid = y - x * beta;
    let m = y.mean();
    (resid.dot(&resid), y.iter().map(|v| (v - m) * (v - m)).sum())
}

/// The rows of `x` (and `y`) where `keep`.
pub fn select_rows(x: &DMatrix<f64>, y: &DVector<f64>, keep: &[bool]) -> (DMatrix<f64>, DVector<f64>) {
    let idx: Vec<usize> = keep.iter().enumerate().filter_map(|(i, &k)| k.then_some(i)).collect();
    (x.select_rows(&idx), y.select_rows(&idx))
}

/// Rows stacked.
pub fn vstack(parts: &[DMatrix<f64>]) -> DMatrix<f64> {
    let cols = parts.first().map_or(0, |p| p.ncols());
    let rows = parts.iter().map(|p| p.nrows()).sum();
    let mut out = DMatrix::zeros(rows, cols);
    let mut r = 0;
    for p in parts {
        out.rows_mut(r, p.nrows()).copy_from(p);
        r += p.nrows();
    }
    out
}

pub fn concat(parts: &[DVector<f64>]) -> DVector<f64> {
    DVector::from_iterator(parts.iter().map(|p| p.len()).sum(), parts.iter().flat_map(|p| p.iter().copied()))
}

/// A one-pole low-pass `y[i] = k·x[i] + (1 − k)·y[i−1]` started at rest on
/// `x[0]` (scipy's `lfilter([k], [1, k − 1], x, zi=[(1 − k)·x[0]])`).
pub fn one_pole(x: &[f64], k: f64) -> Vec<f64> {
    let mut y = Vec::with_capacity(x.len());
    let mut prev = x.first().copied().unwrap_or(0.0);
    for &v in x {
        prev = k * v + (1.0 - k) * prev;
        y.push(prev);
    }
    y
}

/// Brent's root of `f` in [a, b], where `f(a)` and `f(b)` differ in sign.
pub fn brent(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    const XTOL: f64 = 2e-12;
    const RTOL: f64 = 4.0 * f64::EPSILON;
    let (mut fa, mut fb) = (f(a), f(b));
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    let (mut c, mut fc) = (a, fa);
    let (mut d, mut e) = (b - a, b - a);
    for _ in 0..200 {
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * RTOL * b.abs() + 0.5 * XTOL;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return b;
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                let qa = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * m * qa * (qa - r) - (b - a) * (r - 1.0));
                q = (qa - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_interpolate_like_numpy() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(percentile(&x, 50.0), 3.0);
        assert!((percentile(&x, 99.5) - 4.98).abs() < 1e-12);
        assert_eq!(percentile(&x, 0.0), 1.0);
        assert_eq!(percentile(&x, 100.0), 5.0);
        assert!((percentile(&x, 30.0) - 1.4).abs() < 1e-12);
    }

    #[test]
    fn rounds_half_to_even() {
        assert_eq!(round_even(2.5), 2.0);
        assert_eq!(round_even(3.5), 4.0);
        assert_eq!(round_even(-0.5), 0.0);
    }

    #[test]
    fn least_squares_recovers_a_badly_scaled_fit() {
        let n = 200;
        let x = DMatrix::from_fn(n, 2, |i, j| if j == 0 { (i as f64 * 0.37).sin() } else { 1e5 * (i as f64 * 0.11).cos() });
        let y = DVector::from_fn(n, |i, _| 2.0 * x[(i, 0)] - 3e-5 * x[(i, 1)]);
        let (beta, se, r) = ols(&x, &y);
        assert!((beta[0] - 2.0).abs() < 1e-9 && (beta[1] + 3e-5).abs() < 1e-14, "{beta}");
        assert!(se.iter().all(|s| *s < 1e-6));
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn brent_finds_a_root() {
        let r = brent(|x| x * x - 2.0, 0.0, 2.0);
        assert!((r - 2f64.sqrt()).abs() < 1e-11);
    }

    #[test]
    fn one_pole_starts_at_rest() {
        assert_eq!(one_pole(&[2.0, 2.0, 2.0], 0.3), vec![2.0, 2.0, 2.0]);
        let y = one_pole(&[0.0, 1.0], 0.5);
        assert_eq!(y, vec![0.0, 0.5]);
    }
}
