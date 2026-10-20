//! ABR Measurement Update — Relational Treatment Path
//!
//! Treatment boundary: This file applies Kernel V8 operators exclusively to the
//! declared topology from declaration.rs. No statistical estimation or matrix
//! inversion is used. Operators are applied in declared sequence: A → B → R.
//!
//! ─── Observational Input (V0.3) ───────────────────────────────────────────────
//!
//!   AbrInputs now carries the same raw observation as the EKF comparator:
//!     z_observed  = detected pixel position [u, v] from the HoloLens 2 camera
//!     z_predicted = predicted pixel position [u, v] from world model (H·x equivalent)
//!
//!   These are placed as locus values on the declared topology before operator
//!   application. The A operator then computes the residual directly:
//!     A(e₀) = z_observed − z_predicted   (camera_feature − landmark)
//!
//!   This is the same residual the EKF computes as y = z − H·x, but here it
//!   emerges from the declared relational structure, not from a matrix product.
//!
//! ─── Kernel V8 Operators (§11) ────────────────────────────────────────────────
//!
//!   A(x)[e] = x[s] − x[t]
//!     Adjacency operator: directed difference between source and target locus values.
//!     On 2D pixel vectors: [u_src − u_tgt, v_src − v_tgt]
//!
//!   B(g)[e] = g[e] + Σ_{f ∈ succ(e)} g[f]
//!     Forward accumulation: each edge accumulates the values of its successors.
//!
//!   R(g)[e] = g[e] + ρ[src(e)] · (Σ_{succ(e)} g − Σ_{pred(e)} g)
//!     Relational weighting: ρ[i] = ρ_base · χ[i] / (χ₀ + χ[i])
//!
//! ─── Complexity ───────────────────────────────────────────────────────────────
//!
//!   O(|E|) = O(n·k) per operator pass. No inversion. No backpropagation.
//!   For this topology: |E| = 2 → 7 total relational ops.
//!   The 2D pixel vector at each edge is a constant-factor detail (dim=2 fixed);
//!   complexity class is unchanged. Ops counted per edge, not per scalar component.

use crate::declaration::{MDeclaration, RelationalEdge};

// ── ABR inputs ─────────────────────────────────────────────────────────────────

/// Observable measurement data from the reference device (HoloLens 2).
///
/// V0.6: three locus values are carried separately to preserve the continuous
/// process semantics declared by Origin. camera_feature is the sensor refresh;
/// landmark and display_update are owned by the operators and must not be
/// externally reset between frames.
///
///   camera_feature (0) = z_observed    — refreshed from sensor each frame
///   landmark       (1) = z_predicted   — persistent locus; operators accumulate
///   display_update (2) = display_locus — hologram position (application convention)
pub struct AbrInputs {
    /// Detected pixel position of the tracked feature [u, v].
    /// Observed directly from the camera frame — not predicted.
    /// Corresponds to EKF's measurement vector z.
    /// Refreshed from sensor every frame; this is the only locus the caller owns.
    pub z_observed:    [f64; 2],

    /// Persistent landmark locus estimate [u, v] in pixel space.
    /// Maintained by the operators across frames — NOT the hologram position.
    /// Initialized once; updated each frame by: landmark += ρ[landmark] × A(e₀).
    /// Corresponds to H·x in EKF notation. Zero at initialization (H·x = 0).
    pub z_predicted:   [f64; 2],

    /// Current hologram position [u, v] in pixel space.
    /// Reflects where the display is currently showing the hologram.
    /// Updated each frame by the application convention:
    ///   hologram(t+1) = hologram(t) − display_correction(t)
    /// Initialized to [0, 0] (display aligned with world origin at start).
    pub display_locus: [f64; 2],
}

impl AbrInputs {
    /// Canonical AR example matching the EKF comparator exactly.
    ///
    ///   z_observed    = [28.4, −10.9] px — detected feature position (= EKF's z)
    ///   z_predicted   = [0.0,   0.0]  px — H·x = 0 by construction (same as EKF)
    ///   display_locus = [0.0,   0.0]  px — hologram at origin (= z_predicted at init)
    ///
    /// A(e₀) = z_observed − z_predicted = [28.4, −10.9] (= EKF innovation y).
    /// A(e₁) = z_predicted − display_locus = [0, 0] (landmark = hologram at init).
    /// CORR-01 is thereby resolved: both paths act on the same observation.
    pub fn ar_example() -> Self {
        AbrInputs {
            z_observed:    [28.4, -10.9],
            z_predicted:   [0.0,   0.0 ],
            display_locus: [0.0,   0.0 ],
        }
    }
}

// ── Edge value map ─────────────────────────────────────────────────────────────

/// Mutable 2D pixel-vector edge values during operator application.
/// Each edge carries a [u, v] pair in pixel space.
#[derive(Debug, Clone)]
pub struct EdgeValues {
    pub values: Vec<[f64; 2]>,
}

impl EdgeValues {
    pub fn new(n_edges: usize) -> Self {
        EdgeValues { values: vec![[0.0, 0.0]; n_edges] }
    }

    pub fn get(&self, edge_id: usize) -> [f64; 2] {
        self.values[edge_id]
    }

    pub fn set(&mut self, edge_id: usize, v: [f64; 2]) {
        self.values[edge_id] = v;
    }

    /// Combine two edge value maps by element-wise addition.
    /// Used to merge a fresh A snapshot with carried relational state from the previous frame.
    pub fn add(&self, other: &EdgeValues) -> EdgeValues {
        let values = self.values.iter().zip(other.values.iter())
            .map(|(a, b)| [a[0] + b[0], a[1] + b[1]])
            .collect();
        EdgeValues { values }
    }

    /// Scale all edge values by a scalar factor.
    pub fn scale(&self, factor: f64) -> EdgeValues {
        let values = self.values.iter()
            .map(|v| [v[0] * factor, v[1] * factor])
            .collect();
        EdgeValues { values }
    }
}

// ── Operator A ─────────────────────────────────────────────────────────────────

/// A(x)[e] = x[s] − x[t]  (applied component-wise to 2D pixel vectors)
///
/// On the declared topology:
///   e₀ (camera_feature→landmark): A[e₀] = z_observed − z_predicted = innovation [u,v]
///   e₁ (landmark→display_update): A[e₁] = z_predicted − display_prior = [0,0]
///     (display_prior = z_predicted before correction → no prior display error)
///
/// Returns (edge_values_after_A, ops_A) where ops_A counts per-edge operations.
pub fn apply_a(
    decl:         &MDeclaration,
    locus_values: &[[f64; 2]],   // one 2D pixel value per locus
) -> (EdgeValues, usize) {
    let mut g   = EdgeValues::new(decl.edges.len());
    let mut ops = 0usize;

    for edge in &decl.edges {
        let s = locus_values[edge.source];
        let t = locus_values[edge.target];
        g.set(edge.id, [s[0] - t[0], s[1] - t[1]]);
        ops += 1;   // one 2D directed-difference op per edge
    }

    (g, ops)
}

// ── Operator B ─────────────────────────────────────────────────────────────────

/// B(g)[e] = g[e] + Σ_{f ∈ succ(e)} g[f]
///
/// For e₀ (camera_feature→landmark):
///   succ(e₀) = {e₁} → B[e₀] = [28.4,−10.9] + [0,0] = [28.4,−10.9]
/// For e₁ (landmark→display_update):
///   succ(e₁) = {} → B[e₁] = [0,0]  (terminal, unchanged)
///
/// Returns (edge_values_after_B, ops_B).
pub fn apply_b(
    decl:  &MDeclaration,
    g_in:  &EdgeValues,
) -> (EdgeValues, usize) {
    let mut g_out = g_in.clone();
    let mut ops   = 0usize;

    for edge in &decl.edges {
        let succs   = successors_of_edge(decl, edge);
        let n_succ  = succs.len();
        let succ_sum = succs.iter().fold([0.0f64, 0.0f64], |acc, f| {
            let v = g_in.get(f.id);
            [acc[0] + v[0], acc[1] + v[1]]
        });
        let g_e = g_in.get(edge.id);
        g_out.set(edge.id, [g_e[0] + succ_sum[0], g_e[1] + succ_sum[1]]);
        ops += n_succ;  // one 2D accumulation per successor
    }

    (g_out, ops)
}

// ── Operator R ─────────────────────────────────────────────────────────────────

/// R(g)[e] = g[e] + ρ[src(e)] · (Σ_{succ(e)} g − Σ_{pred(e)} g)
///
/// ρ[i] = ρ_base · χ[i] / (χ₀ + χ[i])   (from M declaration)
///
/// For e₀ (camera_feature→landmark), src=camera_feature (ρ=0.020):
///   succ = {e₁} → B[e₁] = [0,0]; pred = {} → 0
///   R[e₀] = [28.4,−10.9] + 0.020·([0,0]−0) = [28.4,−10.9]
///
/// For e₁ (landmark→display_update), src=landmark (ρ=1/60):
///   succ = {}; pred = {e₀} → B[e₀] = [28.4,−10.9]
///   R[e₁] = [0,0] + (1/60)·(0 − [28.4,−10.9])
///          = [−28.4/60, 10.9/60] = [−0.4733, +0.1817] px
///   Negative u-component: corrective direction (opposing upstream residual).
///
/// Returns (edge_values_after_R, ops_R, rho_values).
pub fn apply_r(
    decl:  &MDeclaration,
    g_in:  &EdgeValues,
) -> (EdgeValues, usize, Vec<f64>) {
    let mut g_out     = g_in.clone();
    let mut ops       = 0usize;
    let mut rho_values = vec![0.0; decl.loci.len()];

    for locus in &decl.loci {
        rho_values[locus.id] = decl.rho(locus.id);
    }

    for edge in &decl.edges {
        let rho_src   = rho_values[edge.source];
        let succ_edges = successors_of_edge(decl, edge);
        let pred_edges = predecessors_of_edge(decl, edge);

        let succ_sum = succ_edges.iter().fold([0.0f64, 0.0f64], |acc, f| {
            let v = g_in.get(f.id);
            [acc[0] + v[0], acc[1] + v[1]]
        });
        let pred_sum = pred_edges.iter().fold([0.0f64, 0.0f64], |acc, f| {
            let v = g_in.get(f.id);
            [acc[0] + v[0], acc[1] + v[1]]
        });

        let g_e  = g_in.get(edge.id);
        let diff = [succ_sum[0] - pred_sum[0], succ_sum[1] - pred_sum[1]];
        g_out.set(edge.id, [g_e[0] + rho_src * diff[0],
                             g_e[1] + rho_src * diff[1]]);

        ops += succ_edges.len() + pred_edges.len() + 1;  // sums + one ρ-multiply
    }

    (g_out, ops, rho_values)
}

// ── Helper functions ───────────────────────────────────────────────────────────

fn successors_of_edge<'a>(decl: &'a MDeclaration, edge: &RelationalEdge)
    -> Vec<&'a RelationalEdge>
{
    decl.edges.iter().filter(|f| f.source == edge.target).collect()
}

fn predecessors_of_edge<'a>(decl: &'a MDeclaration, edge: &RelationalEdge)
    -> Vec<&'a RelationalEdge>
{
    decl.edges.iter().filter(|f| f.target == edge.source).collect()
}

// ── ABR result ─────────────────────────────────────────────────────────────────

/// Complete result of one ABR measurement update step.
pub struct AbrResult {
    /// Edge values after A operator (residuals in pixel space).
    pub g_after_a:  EdgeValues,
    /// Edge values after B operator (accumulated).
    pub g_after_b:  EdgeValues,
    /// Edge values after R operator (final relational state).
    pub g_after_r:  EdgeValues,

    /// Saturating weights ρ per locus (for reporting and verification).
    pub rho_values: Vec<f64>,

    /// Correction to apply to the display [u, v] in pixels.
    /// Value of edge e₁ (landmark → display_update) after R.
    /// Negative components are in the corrective direction (opposing upstream residual).
    pub display_correction: [f64; 2],

    // Operation counts (per-edge, dimension-independent)
    pub ops_a:     usize,
    pub ops_b:     usize,
    pub ops_r:     usize,
    pub total_ops: usize,
}

// ── ABR measurement update ─────────────────────────────────────────────────────

/// Execute one ABR measurement update: A → B → R on the declared topology.
///
/// Locus values are set from inputs before operator application:
///   camera_feature (0) = z_observed    — raw detected pixel [u,v] (sensor refresh)
///   landmark       (1) = z_predicted   — persistent locus estimate [u,v]
///   display_update (2) = display_locus — hologram position [u,v]
///
/// A operator then computes:
///   g[e₀] = z_observed − z_predicted   (= EKF innovation y = z − H·x)
///   g[e₁] = z_predicted − display_locus (= landmark − hologram; zero at init)
///
/// V0.6: display_locus is now separate from z_predicted. Both loci are maintained
/// by the caller across frames; only camera_feature is refreshed from the sensor.
/// This resolves the continuous-process violation in V0.5 (landmark was being reset
/// to hologram each frame, discarding accumulated locus knowledge).
///
/// This resolves CORR-01: both paths now act on the same observation.
pub fn abr_measurement_update(
    decl:   &MDeclaration,
    inputs: &AbrInputs,
) -> AbrResult {
    let locus_values: [[f64; 2]; 3] = [
        inputs.z_observed,    // camera_feature: observed pixel position (sensor)
        inputs.z_predicted,   // landmark:       persistent locus estimate
        inputs.display_locus, // display_update: hologram position
    ];

    let (g_a, ops_a) = apply_a(decl, &locus_values);
    let (g_b, ops_b) = apply_b(decl, &g_a);
    let (g_r, ops_r, rho_values) = apply_r(decl, &g_b);

    let display_correction = g_r.get(1);  // edge e₁ (landmark → display_update)
    let total_ops = ops_a + ops_b + ops_r;

    AbrResult {
        g_after_a: g_a,
        g_after_b: g_b,
        g_after_r: g_r,
        rho_values,
        display_correction,
        ops_a,
        ops_b,
        ops_r,
        total_ops,
    }
}

// ── ABR tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration::MDeclaration;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    // ── [IMPL] A operator ──────────────────────────────────────────────────────

    /// [IMPL] A: directed difference — source minus target, component-wise.
    /// e₀ (camera_feature→landmark):    A[e₀] = [3.0,0.0] − [1.0,0.0] = [2.0,0.0]
    /// e₁ (landmark→display_update):    A[e₁] = [1.0,0.0] − [0.0,0.0] = [1.0,0.0]
    #[test]
    fn impl_a_operator_directed_difference() {
        let decl = MDeclaration::ar_measurement_update();
        let locus_values = [[3.0, 0.0], [1.0, 0.0], [0.0, 0.0]];
        let (g, _) = apply_a(&decl, &locus_values);
        assert!(approx_eq(g.get(0)[0], 2.0, 1e-12),
            "A[e₀][u] = {}, expected 2.0", g.get(0)[0]);
        assert!(approx_eq(g.get(1)[0], 1.0, 1e-12),
            "A[e₁][u] = {}, expected 1.0", g.get(1)[0]);
    }

    /// [IMPL] A: direction is not symmetric (source−target ≠ target−source).
    #[test]
    fn impl_a_operator_direction_is_not_symmetric() {
        let decl = MDeclaration::ar_measurement_update();
        let locus_values = [[5.0, 0.0], [2.0, 0.0], [0.0, 0.0]];
        let (g, _) = apply_a(&decl, &locus_values);
        assert!(approx_eq(g.get(0)[0], 3.0, 1e-12),
            "A[e₀][u] must be source−target = 3.0, not −3.0; got {}", g.get(0)[0]);
    }

    /// [IMPL] A: equal source and target → zero vector.
    #[test]
    fn impl_a_operator_equal_values_zero() {
        let decl = MDeclaration::ar_measurement_update();
        let locus_values = [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]];
        let (g, _) = apply_a(&decl, &locus_values);
        assert!(approx_eq(g.get(0)[0], 0.0, 1e-12), "A[e₀][u] must be 0.0");
        assert!(approx_eq(g.get(0)[1], 0.0, 1e-12), "A[e₀][v] must be 0.0");
    }

    /// [IMPL] A: negative residual is valid.
    #[test]
    fn impl_a_operator_negative_residual() {
        let decl = MDeclaration::ar_measurement_update();
        let locus_values = [[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]];
        let (g, _) = apply_a(&decl, &locus_values);
        assert!(approx_eq(g.get(0)[0], -3.0, 1e-12),
            "A[e₀][u] = {}, expected −3.0 (negative residual valid)", g.get(0)[0]);
    }

    /// [IMPL] A: v-component (vertical) processed independently of u.
    #[test]
    fn impl_a_operator_v_component_independent() {
        let decl = MDeclaration::ar_measurement_update();
        let locus_values = [[28.4, -10.9], [0.0, 0.0], [0.0, 0.0]];
        let (g, _) = apply_a(&decl, &locus_values);
        assert!(approx_eq(g.get(0)[0],  28.4, 1e-12), "A[e₀][u] = {}", g.get(0)[0]);
        assert!(approx_eq(g.get(0)[1], -10.9, 1e-12), "A[e₀][v] = {}", g.get(0)[1]);
    }

    /// [IMPL] A: op count = one per edge (dimension-independent).
    #[test]
    fn impl_a_operator_op_count() {
        let decl = MDeclaration::ar_measurement_update();
        let locus_values = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let (_, ops) = apply_a(&decl, &locus_values);
        assert_eq!(ops, decl.edges.len(),
            "A ops must equal |E| = {}; got {}", decl.edges.len(), ops);
    }

    // ── [IMPL] B operator ──────────────────────────────────────────────────────

    /// [IMPL] B: terminal edge (e₁, no successors) unchanged.
    #[test]
    fn impl_b_terminal_edge_unchanged() {
        let decl = MDeclaration::ar_measurement_update();
        let mut g_in = EdgeValues::new(2);
        g_in.set(0, [1.0, 0.5]);
        g_in.set(1, [0.75, 0.25]);
        let (g_b, _) = apply_b(&decl, &g_in);
        assert!(approx_eq(g_b.get(1)[0], 0.75, 1e-12),
            "B[e₁][u] = {}, expected 0.75 (terminal)", g_b.get(1)[0]);
        assert!(approx_eq(g_b.get(1)[1], 0.25, 1e-12),
            "B[e₁][v] = {}, expected 0.25 (terminal)", g_b.get(1)[1]);
    }

    /// [IMPL] B: non-terminal edge (e₀, succ={e₁}) accumulates e₁'s value.
    #[test]
    fn impl_b_nonterminal_accumulates_successor() {
        let decl = MDeclaration::ar_measurement_update();
        let mut g_in = EdgeValues::new(2);
        g_in.set(0, [1.0, 0.0]);
        g_in.set(1, [0.5, 0.0]);
        let (g_b, _) = apply_b(&decl, &g_in);
        assert!(approx_eq(g_b.get(0)[0], 1.5, 1e-12),
            "B[e₀][u] = {}, expected 1.5 (1.0 + 0.5)", g_b.get(0)[0]);
    }

    /// [IMPL] B: zero input → zero output.
    #[test]
    fn impl_b_zero_input_zero_output() {
        let decl = MDeclaration::ar_measurement_update();
        let g_in = EdgeValues::new(2);
        let (g_b, _) = apply_b(&decl, &g_in);
        assert!(approx_eq(g_b.get(0)[0], 0.0, 1e-12), "B[e₀][u] must be 0");
        assert!(approx_eq(g_b.get(1)[0], 0.0, 1e-12), "B[e₁][u] must be 0");
    }

    // ── [IMPL] R operator ──────────────────────────────────────────────────────

    /// [IMPL] R: ρ values match declared formula ρ = ρ_base·χ/(χ₀+χ).
    #[test]
    fn impl_r_rho_values_match_declaration() {
        let decl = MDeclaration::ar_measurement_update();
        let g_in = EdgeValues::new(2);
        let (_, _, rho) = apply_r(&decl, &g_in);
        assert!((rho[0] - 0.025 * 8.0 / (2.0 + 8.0)).abs() < 1e-12,
            "ρ[camera_feature] = {:.9}", rho[0]);
        assert!((rho[1] - 0.025 * 4.0 / (2.0 + 4.0)).abs() < 1e-12,
            "ρ[landmark] = {:.9}", rho[1]);
        assert!((rho[2] - 0.025 * 2.0 / (2.0 + 2.0)).abs() < 1e-12,
            "ρ[display_update] = {:.9}", rho[2]);
    }

    /// [IMPL] R: zero input → zero output (zero is a fixed point).
    #[test]
    fn impl_r_zero_input_zero_output() {
        let decl = MDeclaration::ar_measurement_update();
        let g_in = EdgeValues::new(2);
        let (g_r, _, _) = apply_r(&decl, &g_in);
        assert!(approx_eq(g_r.get(0)[0], 0.0, 1e-12), "R[e₀][u] must be 0 for zero input");
        assert!(approx_eq(g_r.get(1)[0], 0.0, 1e-12), "R[e₁][u] must be 0 for zero input");
    }

    /// [IMPL] R: e₀ formula — R[e₀] = g[e₀] + ρ[0]·(g[e₁] − 0)
    ///   src(e₀) = camera_feature (id=0), succ = {e₁}, pred = {}
    #[test]
    fn impl_r_e0_formula_with_successor() {
        let decl = MDeclaration::ar_measurement_update();
        let mut g_in = EdgeValues::new(2);
        g_in.set(0, [2.0, 0.0]);
        g_in.set(1, [1.0, 0.0]);
        let (g_r, _, rho) = apply_r(&decl, &g_in);
        let expected = 2.0 + rho[0] * (1.0 - 0.0);
        assert!(approx_eq(g_r.get(0)[0], expected, 1e-10),
            "R[e₀][u] = {:.9}, expected {:.9}", g_r.get(0)[0], expected);
    }

    /// [IMPL] R: e₁ formula — R[e₁] = g[e₁] + ρ[1]·(0 − g[e₀])
    ///   src(e₁) = landmark (id=1), succ = {}, pred = {e₀}
    #[test]
    fn impl_r_e1_formula_with_predecessor() {
        let decl = MDeclaration::ar_measurement_update();
        let mut g_in = EdgeValues::new(2);
        g_in.set(0, [1.0, 0.0]);
        g_in.set(1, [0.0, 0.0]);
        let (g_r, _, rho) = apply_r(&decl, &g_in);
        let expected = 0.0 + rho[1] * (0.0 - 1.0);
        assert!(approx_eq(g_r.get(1)[0], expected, 1e-10),
            "R[e₁][u] = {:.9}, expected {:.9}", g_r.get(1)[0], expected);
    }

    /// [IMPL] R: nonzero input preserved — R does not zero nonzero edges.
    #[test]
    fn impl_r_nonzero_input_preserved() {
        let decl = MDeclaration::ar_measurement_update();
        let mut g_in = EdgeValues::new(2);
        g_in.set(0, [5.0, 0.0]);
        g_in.set(1, [0.0, 0.0]);
        let (g_r, _, _) = apply_r(&decl, &g_in);
        assert!(g_r.get(0)[0].abs() > 0.0,
            "R[e₀][u] must be nonzero for nonzero input; got {}", g_r.get(0)[0]);
    }

    // ── [IMPL] ABR regression (V0.3) ──────────────────────────────────────────

    /// [IMPL] ABR regression: ar_example() with unified z=[28.4,−10.9] inputs.
    ///
    /// Reference computed 2026-10-19 from:
    ///   z_observed = [28.4, −10.9], z_predicted = [0.0, 0.0]
    ///
    /// A outputs (exact: integer inputs):
    ///   g_after_a[e₀] = [28.4, −10.9]   (same as EKF innovation)
    ///   g_after_a[e₁] = [0.0,   0.0]
    ///
    /// B outputs:
    ///   g_after_b[e₀] = [28.4, −10.9]   (succ={e₁} with value [0,0])
    ///   g_after_b[e₁] = [0.0,   0.0]    (terminal)
    ///
    /// R outputs:
    ///   g_after_r[e₀] = [28.4, −10.9]   (succ=[0,0], pred=none)
    ///   g_after_r[e₁] = −ρ[1]·[28.4,−10.9] = [−28.4/60, 10.9/60]
    ///                  ≈ [−0.473333, +0.181667] px
    #[test]
    fn impl_abr_regression_ar_example() {
        let decl   = MDeclaration::ar_measurement_update();
        let inputs = AbrInputs::ar_example();
        let result = abr_measurement_update(&decl, &inputs);

        // A outputs: directed differences from locus values
        assert!(approx_eq(result.g_after_a.get(0)[0],  28.4,  1e-12),
            "g_after_a[e₀][u] = {}", result.g_after_a.get(0)[0]);
        assert!(approx_eq(result.g_after_a.get(0)[1], -10.9,  1e-12),
            "g_after_a[e₀][v] = {}", result.g_after_a.get(0)[1]);
        assert!(approx_eq(result.g_after_a.get(1)[0],   0.0,  1e-12),
            "g_after_a[e₁][u] = {}", result.g_after_a.get(1)[0]);
        assert!(approx_eq(result.g_after_a.get(1)[1],   0.0,  1e-12),
            "g_after_a[e₁][v] = {}", result.g_after_a.get(1)[1]);

        // B outputs
        assert!(approx_eq(result.g_after_b.get(0)[0],  28.4,  1e-12),
            "g_after_b[e₀][u] = {}", result.g_after_b.get(0)[0]);
        assert!(approx_eq(result.g_after_b.get(0)[1], -10.9,  1e-12),
            "g_after_b[e₀][v] = {}", result.g_after_b.get(0)[1]);

        // R outputs: e₁ = −ρ[landmark]·[28.4,−10.9]
        let rho1    = 0.025_f64 * 4.0 / (2.0 + 4.0);  // 1/60
        let ref_u_r = -rho1 * 28.4;
        let ref_v_r =  rho1 * 10.9;
        assert!(approx_eq(result.g_after_r.get(0)[0],  28.4,   1e-12),
            "g_after_r[e₀][u] = {}", result.g_after_r.get(0)[0]);
        assert!(approx_eq(result.g_after_r.get(1)[0], ref_u_r, 1e-10),
            "g_after_r[e₁][u] = {:.9}, expected {:.9}",
            result.g_after_r.get(1)[0], ref_u_r);
        assert!(approx_eq(result.g_after_r.get(1)[1], ref_v_r, 1e-10),
            "g_after_r[e₁][v] = {:.9}, expected {:.9}",
            result.g_after_r.get(1)[1], ref_v_r);

        // Display correction = g_after_r[e₁]
        assert!(approx_eq(result.display_correction[0], ref_u_r, 1e-10),
            "display_correction[u] = {:.9}", result.display_correction[0]);
        assert!(approx_eq(result.display_correction[1], ref_v_r, 1e-10),
            "display_correction[v] = {:.9}", result.display_correction[1]);
    }

    // ── [IMPL] ABR operation accounting ───────────────────────────────────────

    /// [IMPL] total_ops = ops_a + ops_b + ops_r.
    #[test]
    fn impl_abr_ops_total_is_sum_of_components() {
        let decl = MDeclaration::ar_measurement_update();
        let r = abr_measurement_update(&decl, &AbrInputs::ar_example());
        assert_eq!(r.total_ops, r.ops_a + r.ops_b + r.ops_r,
            "total_ops ({}) != component sum ({})", r.total_ops, r.ops_a + r.ops_b + r.ops_r);
    }

    /// [IMPL] Total ops regression: 7 for ar_example() topology.
    /// Unchanged from V0.2 — op counting is per-edge, not per scalar component.
    #[test]
    fn impl_abr_ops_total_regression() {
        let decl = MDeclaration::ar_measurement_update();
        let r = abr_measurement_update(&decl, &AbrInputs::ar_example());
        assert_eq!(r.total_ops, 7,
            "ABR total ops = {}, regression reference = 7", r.total_ops);
    }

    /// [IMPL] ABR ops bounded by O(|E|): each per-operator bound holds.
    #[test]
    fn impl_abr_ops_bounded_by_edge_count() {
        let decl = MDeclaration::ar_measurement_update();
        let r = abr_measurement_update(&decl, &AbrInputs::ar_example());
        let e = decl.edges.len();
        assert!(r.ops_a <= e,     "A ops {} ≤ |E| = {}", r.ops_a, e);
        assert!(r.ops_b <= e,     "B ops {} ≤ |E| = {}", r.ops_b, e);
        assert!(r.ops_r <= 3 * e, "R ops {} ≤ 3·|E| = {}", r.ops_r, 3 * e);
    }
}
