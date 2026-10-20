//! EKF Measurement Update — Comparator Path
//!
//! Treatment boundary: This file contains conventional Extended Kalman Filter
//! mathematics. No Kernel V8 operators are applied here. This code serves as
//! the comparator against which the ABR relational path is measured.
//!
//! ─── EKF Measurement Update Equations ────────────────────────────────────────
//!
//!   Innovation:     y = z − H·x
//!   Innovation cov: S = H·P·Hᵀ + R
//!   Kalman gain:    K = P·Hᵀ·S⁻¹
//!   State update:   x̂ = x + K·y
//!   Cov update:     P̂ = (I − K·H)·P
//!
//! ─── Complexity Analysis ──────────────────────────────────────────────────────
//!
//!   Let n = state dimension, m = measurement dimension.
//!
//!   H·P (m×n · n×n):          O(m·n²)  multiplications
//!   (H·P)·Hᵀ (m×n · n×m):    O(m²·n)  multiplications  → produces m×m matrix S
//!   S⁻¹ (m×m inversion):      O(m³)    — cubic in measurement dimension
//!   P·Hᵀ (n×n · n×m):         O(n²·m)  multiplications
//!   K = P·Hᵀ · S⁻¹ (n×m·m×m):O(n·m²)  multiplications  → produces n×m gain K
//!   K·y (n×m · m):             O(n·m)
//!   K·H (n×m · m×n):           O(n·m·n) = O(n²·m)
//!   (I−KH)·P (n×n · n×n):     O(n³)    multiplications
//!
//!   Dominant cost: O(n³) for covariance update + O(m³) for innovation inversion.
//!   Verifier finding V-02: inversion is cubic in m (measurement dim), not n.
//!   Both dimensions are recorded in EkfResult for transparency.

/// Dimensions used in this comparison.
/// Chosen to represent a realistic AR visual-inertial landmark update.
pub const STATE_DIM: usize = 6;        // [x, y, z, roll, pitch, yaw]
pub const MEASUREMENT_DIM: usize = 2;  // [pixel_u, pixel_v]

// ── Matrix type ────────────────────────────────────────────────────────────────

/// Row-major dense matrix of f64.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data:     Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n { m.set(i, i, 1.0); }
        m
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Matrix multiply: (self: r×k) · (other: k×c) → r×c
    /// Returns (result, multiply_ops) where multiply_ops counts scalar multiplications.
    pub fn mul(&self, other: &Matrix) -> (Matrix, usize) {
        assert_eq!(self.cols, other.rows, "dimension mismatch in multiply");
        let mut result = Matrix::zeros(self.rows, other.cols);
        let mut ops = 0usize;
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                    ops += 1;
                }
                result.set(i, j, sum);
            }
        }
        (result, ops)
    }

    /// Matrix add: element-wise, same dimensions.
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] += other.data[i];
        }
        result
    }

    /// Matrix subtract: element-wise, same dimensions.
    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] -= other.data[i];
        }
        result
    }

    /// Transpose.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Invert a square matrix using Gaussian elimination with partial pivoting.
    /// Returns (inverse, ops) where ops counts row operations (each is O(n) multiplications).
    /// Complexity: O(n³) in the dimension of this matrix.
    pub fn invert(&self) -> (Matrix, usize) {
        assert_eq!(self.rows, self.cols, "invert requires square matrix");
        let n = self.rows;
        let mut a = self.clone();
        let mut inv = Matrix::identity(n);
        let mut ops = 0usize;

        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = a.get(col, col).abs();
            for row in (col + 1)..n {
                let v = a.get(row, col).abs();
                if v > max_val { max_val = v; max_row = row; }
            }
            // Swap rows in a and inv
            for c in 0..n {
                let tmp_a = a.get(col, c);
                a.set(col, c, a.get(max_row, c));
                a.set(max_row, c, tmp_a);
                let tmp_i = inv.get(col, c);
                inv.set(col, c, inv.get(max_row, c));
                inv.set(max_row, c, tmp_i);
            }
            // Scale pivot row
            let pivot = a.get(col, col);
            for c in 0..n {
                a.set(col, c, a.get(col, c) / pivot);
                inv.set(col, c, inv.get(col, c) / pivot);
                ops += 1;
            }
            // Eliminate column
            for row in 0..n {
                if row == col { continue; }
                let factor = a.get(row, col);
                for c in 0..n {
                    let new_a = a.get(row, c) - factor * a.get(col, c);
                    a.set(row, c, new_a);
                    let new_i = inv.get(row, c) - factor * inv.get(col, c);
                    inv.set(row, c, new_i);
                    ops += 1;
                }
            }
        }
        (inv, ops)
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f64) -> Matrix {
        let mut result = self.clone();
        for v in result.data.iter_mut() { *v *= s; }
        result
    }

    /// Return the data slice (for display / verification).
    pub fn data(&self) -> &[f64] { &self.data }

    /// Pretty-print for a vector (single column or single row).
    pub fn as_vec_string(&self) -> String {
        if self.cols == 1 {
            let vals: Vec<String> = (0..self.rows)
                .map(|i| format!("{:+.6}", self.get(i, 0)))
                .collect();
            format!("[{}]", vals.join(", "))
        } else if self.rows == 1 {
            let vals: Vec<String> = (0..self.cols)
                .map(|j| format!("{:+.6}", self.get(0, j)))
                .collect();
            format!("[{}]", vals.join(", "))
        } else {
            format!("({}×{} matrix)", self.rows, self.cols)
        }
    }
}

// ── EKF inputs ────────────────────────────────────────────────────────────────

/// Inputs to a single EKF measurement update.
pub struct EkfInputs {
    /// Prior state estimate (n × 1).
    pub x: Matrix,
    /// Prior covariance (n × n).
    pub p: Matrix,
    /// Measurement Jacobian (m × n).
    pub h: Matrix,
    /// Measurement noise covariance (m × m).
    pub r_noise: Matrix,
    /// Measurement vector (m × 1).
    pub z: Matrix,
}

impl EkfInputs {
    /// Construct a representative AR landmark measurement update.
    /// Values are chosen to be numerically realistic for a visual-inertial system.
    pub fn ar_example() -> Self {
        let n = STATE_DIM;
        let m = MEASUREMENT_DIM;

        // Prior state: camera at small offset from origin with slight roll
        let mut x = Matrix::zeros(n, 1);
        x.set(0, 0,  0.05);  // x = 5 cm
        x.set(1, 0, -0.02);  // y = -2 cm
        x.set(2, 0,  1.50);  // z = 1.5 m depth
        x.set(3, 0,  0.01);  // roll  = ~0.57°
        x.set(4, 0, -0.01);  // pitch = ~-0.57°
        x.set(5, 0,  0.00);  // yaw   = 0

        // Prior covariance: moderate uncertainty
        let mut p = Matrix::identity(n);
        for i in 0..n { p.set(i, i, if i < 3 { 0.01 } else { 0.001 }); }

        // Measurement Jacobian: projects 3-D position into pixel (u, v)
        // Simplified: u ≈ f*X/Z, v ≈ f*Y/Z  (focal length f = 47 px/degree × π/180 ≈ 822 px)
        let f = 822.0_f64;
        let z_depth = x.get(2, 0);
        let mut h = Matrix::zeros(m, n);
        h.set(0, 0, f / z_depth);   // ∂u/∂x
        h.set(0, 2, -f * x.get(0, 0) / (z_depth * z_depth)); // ∂u/∂z
        h.set(1, 1, f / z_depth);   // ∂v/∂y
        h.set(1, 2, -f * x.get(1, 0) / (z_depth * z_depth)); // ∂v/∂z

        // Measurement noise: 2 pixel std dev → 4 px² variance on diagonal
        let mut r_noise = Matrix::zeros(m, m);
        r_noise.set(0, 0, 4.0);
        r_noise.set(1, 1, 4.0);

        // Measurement: detected feature at (u=28.4, v=-10.9) pixels
        let mut z = Matrix::zeros(m, 1);
        z.set(0, 0, 28.4);
        z.set(1, 0, -10.9);

        EkfInputs { x, p, h, r_noise, z }
    }

    /// ar_example with a custom measurement vector [z_u, z_v] in pixels.
    /// All other inputs (prior state, covariance, H, R) are identical to ar_example.
    /// Used by the multi-step simulation to vary z_observed across frames.
    pub fn ar_example_with_z(z_u: f64, z_v: f64) -> Self {
        let mut base = Self::ar_example();
        base.z.set(0, 0, z_u);
        base.z.set(1, 0, z_v);
        base
    }
}

// ── EKF result ────────────────────────────────────────────────────────────────

/// All outputs from the EKF measurement update step, with operation accounting.
pub struct EkfResult {
    // Outputs
    pub x_posterior:       Matrix,   // updated state (n × 1)
    pub p_posterior:       Matrix,   // updated covariance (n × n)
    pub innovation:        Matrix,   // y = z − H·x  (m × 1)
    pub kalman_gain:       Matrix,   // K  (n × m)

    /// State correction projected back to pixel (observable) space: H · K · y  (m × 1).
    ///
    /// This is the EKF's implied pixel-space correction — what the state update
    /// means in the units the measurement was taken in.
    /// Enables direct comparison with ABR's display_correction [u, v] in pixels.
    /// Not counted in total_ops (reporting field only).
    pub pixel_correction:  Matrix,   // H · Δx̂ = H · K · y  (m × 1)

    // Dimensions (recorded per Verifier finding V-02)
    pub state_dim:         usize,
    pub measurement_dim:   usize,

    // Operation counts (scalar multiplications)
    pub ops_hp:            usize,    // H·P
    pub ops_hpht:          usize,    // (H·P)·Hᵀ
    pub ops_inversion:     usize,    // S⁻¹  — cubic in m
    pub ops_pht:           usize,    // P·Hᵀ
    pub ops_gain:          usize,    // K = P·Hᵀ · S⁻¹
    pub ops_state_update:  usize,    // K·y
    pub ops_kh:            usize,    // K·H
    pub ops_cov_update:    usize,    // (I − K·H)·P
    pub total_ops:         usize,
}

// ── EKF measurement update ────────────────────────────────────────────────────

/// Execute a single EKF measurement update step, counting all scalar operations.
///
/// NOTE on operation counting scope:
///   Counted:   scalar multiplications in matrix-multiply and inversion routines.
///   Excluded:  scalar additions within mul/add/sub, loop indexing, pivot comparisons,
///              and the innovation computation (H·x, z−H·x).
///   Rationale: multiplications dominate at large n,m; additions excluded from total_ops.
///   This exclusion is documented and not a silent omission.
pub fn ekf_measurement_update(inputs: &EkfInputs) -> EkfResult {
    let n = inputs.x.rows;
    let m = inputs.z.rows;

    let h  = &inputs.h;
    let p  = &inputs.p;
    let r  = &inputs.r_noise;
    let x  = &inputs.x;
    let z  = &inputs.z;
    let ht = h.transpose();

    // Innovation: y = z − H·x
    let (hx, _)       = h.mul(x);
    let innovation    = z.sub(&hx);

    // Innovation covariance: S = H·P·Hᵀ + R
    let (hp, ops_hp)    = h.mul(p);
    let (hpht, ops_hpht)= hp.mul(&ht);
    let s               = hpht.add(r);

    // Kalman gain: K = P·Hᵀ · S⁻¹
    let (s_inv, ops_inv) = s.invert();
    let (pht, ops_pht)   = p.mul(&ht);
    let (k, ops_gain)    = pht.mul(&s_inv);

    // State update: x̂ = x + K·y
    let (ky, ops_state)  = k.mul(&innovation);
    let x_posterior      = x.add(&ky);

    // Covariance update: P̂ = (I − K·H)·P
    let i_n              = Matrix::identity(n);
    let (kh, ops_kh)     = k.mul(h);
    let i_minus_kh       = i_n.sub(&kh);
    let (p_posterior, ops_cov) = i_minus_kh.mul(p);

    // Pixel-space correction: H · K · y = H · Δx̂
    // ky already holds K·y (the state delta); project through H.
    // Not counted in total_ops — reporting field only.
    let (pixel_correction, _) = h.mul(&ky);

    let total_ops = ops_hp + ops_hpht + ops_inv + ops_pht + ops_gain
                  + ops_state + ops_kh + ops_cov;

    EkfResult {
        x_posterior,
        p_posterior,
        innovation,
        kalman_gain: k,
        pixel_correction,
        state_dim:         n,
        measurement_dim:   m,
        ops_hp,
        ops_hpht,
        ops_inversion:     ops_inv,
        ops_pht,
        ops_gain,
        ops_state_update:  ops_state,
        ops_kh,
        ops_cov_update:    ops_cov,
        total_ops,
    }
}

// ── EKF tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── [MATH] Matrix unit tests ───────────────────────────────────────────────

    /// [MATH] 2×2 matrix multiply: reference values computed independently.
    /// A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    /// A·B = [[1·5+2·7, 1·6+2·8],[3·5+4·7, 3·6+4·8]] = [[19,22],[43,50]]
    #[test]
    fn math_matrix_multiply_2x2_reference() {
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 1.0); a.set(0, 1, 2.0);
        a.set(1, 0, 3.0); a.set(1, 1, 4.0);

        let mut b = Matrix::zeros(2, 2);
        b.set(0, 0, 5.0); b.set(0, 1, 6.0);
        b.set(1, 0, 7.0); b.set(1, 1, 8.0);

        let (c, ops) = a.mul(&b);
        assert!(approx_eq(c.get(0, 0), 19.0, 1e-12), "c[0,0] = {}, expected 19", c.get(0,0));
        assert!(approx_eq(c.get(0, 1), 22.0, 1e-12), "c[0,1] = {}, expected 22", c.get(0,1));
        assert!(approx_eq(c.get(1, 0), 43.0, 1e-12), "c[1,0] = {}, expected 43", c.get(1,0));
        assert!(approx_eq(c.get(1, 1), 50.0, 1e-12), "c[1,1] = {}, expected 50", c.get(1,1));
        // 2×2·2×2 = 2·2·2 = 8 multiplications
        assert_eq!(ops, 8, "2×2·2×2 must count 8 scalar multiplications");
    }

    /// [MATH] Identity matrix multiply: A·I = A.
    #[test]
    fn math_matrix_multiply_by_identity() {
        let mut a = Matrix::zeros(3, 3);
        a.set(0, 0, 2.0); a.set(0, 1, -1.0); a.set(0, 2, 3.0);
        a.set(1, 0, 0.0); a.set(1, 1,  4.0); a.set(1, 2, -2.0);
        a.set(2, 0, 1.0); a.set(2, 1,  0.0); a.set(2, 2,  5.0);
        let identity = Matrix::identity(3);
        let (result, _) = a.mul(&identity);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(result.get(i, j), a.get(i, j), 1e-12),
                    "A·I[{i},{j}] = {}, expected {}", result.get(i,j), a.get(i,j)
                );
            }
        }
    }

    /// [MATH] 2×2 matrix inversion: reference values computed independently.
    /// A = [[2,1],[1,3]], det = 6−1 = 5
    /// A⁻¹ = (1/5)·[[3,−1],[−1,2]] = [[0.6,−0.2],[−0.2,0.4]]
    #[test]
    fn math_matrix_invert_2x2_reference() {
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 2.0); a.set(0, 1, 1.0);
        a.set(1, 0, 1.0); a.set(1, 1, 3.0);

        let (inv, _) = a.invert();
        assert!(approx_eq(inv.get(0, 0),  0.6, 1e-10), "inv[0,0] = {}, expected 0.6",  inv.get(0,0));
        assert!(approx_eq(inv.get(0, 1), -0.2, 1e-10), "inv[0,1] = {}, expected -0.2", inv.get(0,1));
        assert!(approx_eq(inv.get(1, 0), -0.2, 1e-10), "inv[1,0] = {}, expected -0.2", inv.get(1,0));
        assert!(approx_eq(inv.get(1, 1),  0.4, 1e-10), "inv[1,1] = {}, expected 0.4",  inv.get(1,1));
    }

    /// [MATH] A·A⁻¹ = I (inversion round-trip).
    #[test]
    fn math_matrix_invert_round_trip() {
        let mut a = Matrix::zeros(3, 3);
        a.set(0, 0, 4.0); a.set(0, 1, 7.0); a.set(0, 2, 2.0);
        a.set(1, 0, 3.0); a.set(1, 1, 6.0); a.set(1, 2, 1.0);
        a.set(2, 0, 2.0); a.set(2, 1, 5.0); a.set(2, 2, 3.0);

        let (inv, _) = a.invert();
        let (product, _) = a.mul(&inv);
        let identity = Matrix::identity(3);

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(product.get(i, j), identity.get(i, j), 1e-8),
                    "A·A⁻¹[{i},{j}] = {:.10}, expected {:.1} (round-trip failed)",
                    product.get(i, j), identity.get(i, j)
                );
            }
        }
    }

    /// [MATH] Matrix transpose: (Aᵀ)ᵀ = A.
    #[test]
    fn math_matrix_transpose_double() {
        let mut a = Matrix::zeros(2, 3);
        a.set(0, 0, 1.0); a.set(0, 1, 2.0); a.set(0, 2, 3.0);
        a.set(1, 0, 4.0); a.set(1, 1, 5.0); a.set(1, 2, 6.0);
        let att = a.transpose().transpose();
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq(att.get(i, j), a.get(i, j), 1e-12),
                    "(Aᵀ)ᵀ[{i},{j}] must equal A[{i},{j}]");
            }
        }
    }

    // ── [MATH] EKF analytical result ──────────────────────────────────────────

    /// [MATH] Innovation y = z − H·x is analytically exact for the ar_example.
    ///
    /// H·x for ar_example: by construction of H, H[0,0]·x[0] + H[0,2]·x[2] ≈ 0
    /// because H[0,0] = f/z and H[0,2] = −f·x₀/z², so their sum = f/z·x₀ − f·x₀/z = 0.
    /// Therefore y = z − 0 = z = [28.4, −10.9].
    #[test]
    fn math_ekf_innovation_analytical() {
        let inputs = EkfInputs::ar_example();
        let result = ekf_measurement_update(&inputs);
        assert!(
            approx_eq(result.innovation.get(0, 0), 28.4, 1e-6),
            "innovation[0] = {}, expected 28.4 (H·x = 0 by construction)",
            result.innovation.get(0, 0)
        );
        assert!(
            approx_eq(result.innovation.get(1, 0), -10.9, 1e-6),
            "innovation[1] = {}, expected -10.9 (H·x = 0 by construction)",
            result.innovation.get(1, 0)
        );
    }

    // ── [IMPL] EKF regression ─────────────────────────────────────────────────

    /// [IMPL] EKF posterior state regression: must match first-run reference to 1e-4.
    /// Reference: run 2026-10-19, inputs ar_example().
    #[test]
    fn impl_ekf_regression_posterior_state() {
        let inputs = EkfInputs::ar_example();
        let result = ekf_measurement_update(&inputs);
        // Reference values from first conformant run
        let ref_x0 =  0.101690_f64;
        let ref_x1 = -0.039838_f64;
        let ref_x2 =  1.498013_f64;
        assert!(
            approx_eq(result.x_posterior.get(0, 0), ref_x0, 1e-4),
            "x_posterior[0] = {:.6}, reference {:.6}", result.x_posterior.get(0,0), ref_x0
        );
        assert!(
            approx_eq(result.x_posterior.get(1, 0), ref_x1, 1e-4),
            "x_posterior[1] = {:.6}, reference {:.6}", result.x_posterior.get(1,0), ref_x1
        );
        assert!(
            approx_eq(result.x_posterior.get(2, 0), ref_x2, 1e-4),
            "x_posterior[2] = {:.6}, reference {:.6}", result.x_posterior.get(2,0), ref_x2
        );
    }

    // ── [IMPL] Operation accounting ───────────────────────────────────────────

    /// [IMPL] EKF total_ops must equal sum of all component op counts.
    #[test]
    fn impl_ekf_ops_total_is_sum_of_components() {
        let inputs = EkfInputs::ar_example();
        let r = ekf_measurement_update(&inputs);
        let sum = r.ops_hp + r.ops_hpht + r.ops_inversion + r.ops_pht
                + r.ops_gain + r.ops_state_update + r.ops_kh + r.ops_cov_update;
        assert_eq!(
            r.total_ops, sum,
            "total_ops ({}) must equal sum of components ({})", r.total_ops, sum
        );
    }

    /// [IMPL] Inversion ops must be ≤ m³ (measurement dim cubed).
    ///
    /// Verifier finding V-02: inversion is cubic in measurement dimension, not state dimension.
    /// Gaussian elimination on m×m matrix: at most m³ row operations.
    #[test]
    fn impl_ekf_inversion_ops_bounded_by_m_cubed() {
        let inputs = EkfInputs::ar_example();
        let r = ekf_measurement_update(&inputs);
        let m = r.measurement_dim;
        let bound = m * m * m;
        assert!(
            r.ops_inversion <= bound,
            "[V-02] inversion ops {} must be ≤ m³ = {} (m={})", r.ops_inversion, bound, m
        );
    }

    /// [IMPL] Covariance update ops must be ≤ n³ (state dim cubed).
    #[test]
    fn impl_ekf_cov_update_ops_bounded_by_n_cubed() {
        let inputs = EkfInputs::ar_example();
        let r = ekf_measurement_update(&inputs);
        let n = r.state_dim;
        let bound = n * n * n;
        assert!(
            r.ops_cov_update <= bound,
            "cov_update ops {} must be ≤ n³ = {} (n={})", r.ops_cov_update, bound, n
        );
    }

    /// [IMPL] State and measurement dimensions recorded correctly.
    #[test]
    fn impl_ekf_dimensions_recorded() {
        let inputs = EkfInputs::ar_example();
        let r = ekf_measurement_update(&inputs);
        assert_eq!(r.state_dim, STATE_DIM,
            "state_dim must equal STATE_DIM constant");
        assert_eq!(r.measurement_dim, MEASUREMENT_DIM,
            "measurement_dim must equal MEASUREMENT_DIM constant");
    }

    /// [IMPL] Total EKF ops must equal known reference count (500).
    /// Regression guard: if the implementation changes, this catches silent op shifts.
    #[test]
    fn impl_ekf_ops_total_regression() {
        let inputs = EkfInputs::ar_example();
        let r = ekf_measurement_update(&inputs);
        assert_eq!(
            r.total_ops, 500,
            "EKF total ops = {}, regression reference = 500", r.total_ops
        );
    }
}
