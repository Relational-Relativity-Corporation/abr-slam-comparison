//! Multi-step simulation: 30 frames, 1 second, linear left-to-right pan.
//!
//! Declared by Origin:
//!   - Drift model: linear pan, +pan_rate_u px/frame in u, v constant.
//!     Identical z_observed for both ABR and EKF paths every frame.
//!   - Application convention: hologram_pos(t+1) = hologram_pos(t) − display_correction(t)
//!     where display_correction = R(e₁) = −ρ·residual, so the hologram moves toward the
//!     feature by ρ·residual each frame.
//!   - EKF prior: fixed (P not evolved between frames; process model not yet declared).
//!     Each frame runs the complete ekf_measurement_update() with the current z_observed.
//!     Ops are counted from the actual returned EkfResult, not assigned.
//!
//! V0.5 corrections:
//!   V-09: EKF continuous now runs full ekf_measurement_update() each frame; ops counted
//!         from the returned result rather than assigned as 500 per frame.
//!   V-10: Budget-matched comparator removed. Both paths run at 30 fps.
//!         The comparison is: equal frequency, different per-frame cost, different residual.
//!   V-11: Residual recorded after applying each frame's correction, not before.

/// Pan parameters. Declared by Origin: linear left-to-right pan, 1 px/frame.
pub struct PanParams {
    /// Number of frames to simulate.
    pub n_frames: usize,
    /// Frames per second (display rate).
    pub fps: f64,
    /// Feature drift rate in u, pixels per frame.
    pub pan_rate_u: f64,
    /// Initial camera observation (first frame signal from the camera).
    pub z0_observed: [f64; 2],
    /// Initial display state (hologram position before the first camera frame).
    /// Defaults to [0.0, 0.0] (black screen / display at origin).
    /// Both hologram_absolute and z_prev are initialized from this value,
    /// so frame 0's camera signal establishes the initial field relationship
    /// from the declared display state — grounded in an observable, not assumed.
    pub z0_hologram: [f64; 2],
}

impl PanParams {
    /// 30 frames, 30 fps, 1 px/frame left-to-right pan. Declared by Origin.
    /// z0_hologram = [0,0]: display starts at origin (black screen before first camera frame).
    pub fn one_second_30fps() -> Self {
        PanParams {
            n_frames:    30,
            fps:         30.0,
            pan_rate_u:  1.0,
            z0_observed: [0.0, -10.9],
            z0_hologram: [0.0, 0.0],
        }
    }

    /// 120 frames (4 seconds) at 30 fps — long enough to show landmark convergence.
    /// Declared by Origin for V0.9 convergence study.
    pub fn four_seconds_30fps() -> Self {
        PanParams {
            n_frames:    120,
            fps:         30.0,
            pan_rate_u:  1.0,
            z0_observed: [0.0, -10.9],
            z0_hologram: [0.0, 0.0],
        }
    }

    /// 300 frames (10 seconds) at 30 fps.
    pub fn ten_seconds_30fps() -> Self {
        PanParams {
            n_frames:    300,
            fps:         30.0,
            pan_rate_u:  1.0,
            z0_observed: [0.0, -10.9],
            z0_hologram: [0.0, 0.0],
        }
    }

    /// 600 frames (20 seconds) at 30 fps — V0.9 predict-update full convergence demo.
    /// Shows both velocity (Phase 1 ~138 fr) and position (Phase 2 ~276 fr) converge to < 0.1 px.
    pub fn twenty_seconds_30fps() -> Self {
        PanParams {
            n_frames:    600,
            fps:         30.0,
            pan_rate_u:  1.0,
            z0_observed: [0.0, -10.9],
            z0_hologram: [0.0, 0.0],
        }
    }

    pub fn z_observed(&self, t: usize) -> [f64; 2] {
        [
            self.z0_observed[0] + self.pan_rate_u * t as f64,
            self.z0_observed[1],
        ]
    }
}

/// Per-frame record. All fields are measured after the correction is applied (V-11).
#[derive(Clone)]
pub struct SimFrame {
    pub frame:           usize,
    pub t_ms:            f64,
    pub z_observed:      [f64; 2],
    /// Hologram position after this frame's correction.
    pub hologram:        [f64; 2],
    /// Residual = z_observed − hologram, measured after correction (V-11).
    pub residual_after:  [f64; 2],
    /// Residual before this frame's correction (for reference).
    pub residual_before: [f64; 2],
    /// Correction applied this frame.
    pub correction:      [f64; 2],
    /// Ops counted from the actual path execution this frame.
    pub ops:             usize,
    /// Landmark locus after this frame's update (ABR only; [0,0] for EKF).
    pub landmark_locus:  [f64; 2],
    /// Per-frame display velocity after this frame's update (ABR only; [0,0] for EKF).
    pub display_delta:   [f64; 2],
}

pub struct SimResult {
    pub frames:         Vec<SimFrame>,
    pub total_ops:      usize,
    pub label:          &'static str,
    pub description:    &'static str,
    /// Final landmark locus position after all frames (ABR path only).
    /// [0.0, 0.0] for paths that do not maintain a landmark locus (EKF).
    pub landmark_final: [f64; 2],
}

// ─── ABR simulation ───────────────────────────────────────────────────────────

/// ABR continuous (V0.8): runs full abr_measurement_update() every frame, 7 ops/frame.
///
/// V0.7: Observable primitive — frame-to-frame camera delta (z[t]−z[t−1]), not absolute px.
///
/// V0.8: Relational state initialized from the first observation.
///   z_prev and hologram_absolute both start from params.z0_hologram (declared display state
///   before the first camera frame — "black screen"). Frame 0's camera signal then
///   establishes the initial field relationship from that declared state; no relational
///   state is assumed. Subsequent frames evolve the relationship through A→B→R.
///
///   This closes the V0.7 initialization gap: hologram_absolute no longer starts at a
///   hardcoded [0,0] independent of the first observation. The full chain is grounded:
///     declared display state (z0_hologram)
///     → first camera observation (z0_observed)
///     → operators establish initial relational state
///     → subsequent camera observations evolve it
///
/// Three loci and their semantics:
///   camera_feature  (locus 0): z_delta = z[t] − z[t−1]   (camera motion this frame)
///   landmark        (locus 1): motion-rate estimate        (converges toward z_delta)
///   display_update  (locus 2): per-frame display velocity  (correction applied each frame)
///
/// Landmark update: landmark(t+1) = landmark(t) + ρ[landmark] × R(e₀)(t)
///   Full A→B→R pipeline used.  ρ-damped for stability with delta inputs.
///
/// Display application:
///   display_delta(t+1) = display_delta(t) − R(e₁)(t)
///   hologram_absolute(t+1) = hologram_absolute(t) + display_delta(t+1)
///
/// Residual: z_obs_absolute − hologram_absolute  (absolute alignment error).
pub fn simulate_abr(params: &PanParams) -> SimResult {
    use crate::declaration::MDeclaration;
    use crate::abr_step::{AbrInputs, abr_measurement_update};

    let decl         = MDeclaration::ar_measurement_update();
    let rho_landmark = decl.rho(1);  // ρ[landmark] = 0.025 × 4/(2+4) = 1/60

    // V0.8: both z_prev and hologram_absolute start from z0_hologram (declared display state
    // before the first camera frame). Frame 0's camera signal z0_observed then establishes
    // the initial field relationship from that state — grounded in an observable.
    let mut z_prev:            [f64; 2] = params.z0_hologram;
    let mut landmark_locus:    [f64; 2] = [0.0, 0.0];        // motion-rate estimate (delta space)
    let mut display_delta:     [f64; 2] = [0.0, 0.0];        // per-frame display velocity (delta space)
    let mut hologram_absolute: [f64; 2] = params.z0_hologram; // accumulated display position (absolute)
    let mut frames = Vec::with_capacity(params.n_frames);
    let mut total  = 0usize;

    for t in 0..params.n_frames {
        let z_obs   = params.z_observed(t);
        // Observable primitive: frame-to-frame camera motion, not absolute position.
        let z_delta = [z_obs[0] - z_prev[0], z_obs[1] - z_prev[1]];

        let residual_before = [z_obs[0] - hologram_absolute[0],
                                z_obs[1] - hologram_absolute[1]];

        // V0.7: camera_feature carries camera motion delta; landmark and display_update
        // are in the same delta space.
        let inputs = AbrInputs {
            z_observed:    z_delta,         // camera motion this frame
            z_predicted:   landmark_locus,  // motion-rate estimate
            display_locus: display_delta,   // display velocity state
        };
        let r = abr_measurement_update(&decl, &inputs);

        // Landmark update: full A→B→R pipeline (V0.7 correction to V0.6 defect).
        // ρ-damped for stability; R(e₀) = g_after_r edge 0.
        let r_e0 = r.g_after_r.get(0);
        landmark_locus[0] += rho_landmark * r_e0[0];
        landmark_locus[1] += rho_landmark * r_e0[1];

        // Display velocity update: same sign convention as V0.6.
        // R(e₁) = display_correction; display_delta moves toward camera velocity.
        display_delta[0] -= r.display_correction[0];
        display_delta[1] -= r.display_correction[1];

        // Accumulate display velocity into absolute hologram position.
        hologram_absolute[0] += display_delta[0];
        hologram_absolute[1] += display_delta[1];

        // Residual after this frame's correction (absolute alignment error).
        let residual_after = [z_obs[0] - hologram_absolute[0],
                               z_obs[1] - hologram_absolute[1]];
        total += r.total_ops;
        z_prev = z_obs;

        frames.push(SimFrame {
            frame:           t,
            t_ms:            (t as f64 / params.fps) * 1000.0,
            z_observed:      z_obs,
            hologram:        hologram_absolute,
            residual_before,
            residual_after,
            correction:      display_delta,   // velocity applied this frame
            ops:             r.total_ops,
            landmark_locus,
            display_delta,
        });
    }

    SimResult {
        frames,
        total_ops:      total,
        label:          "ABR continuous (V0.8)",
        description:    "V0.8: relational state initialized from first observation (z0_hologram); delta inputs; R(e₀) landmark update",
        landmark_final: landmark_locus,
    }
}

// ─── ABR V0.9: predict-update (closes the position loop) ─────────────────────

/// ABR predict-update (V0.9): adds a position correction step to V0.8.
///
/// V0.8 demonstrated velocity tracking (landmark → pan_rate_u) but the absolute
/// position residual grew because the hologram accumulated velocity but had no
/// position correction path.
///
/// V0.9 adds a predict-update cycle per frame:
///   1. Velocity step (unchanged from V0.8):
///      z_delta = z_obs − z_prev
///      A→B→R operators update landmark toward z_delta (velocity estimate)
///      display_delta updated via R(e₁)
///   2. Prediction:
///      hologram_pred = hologram_absolute + display_delta
///      (project position forward using current velocity estimate)
///   3. Position correction:
///      position_residual = z_obs − hologram_pred
///      hologram_absolute = hologram_pred + ρ_pos × position_residual
///      (ρ-bounded pull toward the observed absolute feature position)
///
/// The position correction adds 4 scalar ops/frame (2× mul + 2× add for [u,v]).
/// Total: 7 (V0.8) + 4 (position correction) = 11 ops/frame.
///
/// This demonstrates a complete linear recursive algebraic feedback loop:
///   - Velocity loop (landmark): converges to pan_rate_u in ~138 frames
///   - Position loop (hologram): corrects accumulated lag; residual → 0
///   - Combined: full position tracking at 11 ops/frame vs EKF 500 ops/frame
pub fn simulate_abr_predict_update(params: &PanParams) -> SimResult {
    use crate::declaration::MDeclaration;
    use crate::abr_step::{AbrInputs, abr_measurement_update};

    let decl         = MDeclaration::ar_measurement_update();
    let rho_landmark = decl.rho(1);   // ρ[landmark] = 1/60  (velocity loop)
    // Position correction gain: same ρ as landmark for consistency with declared topology.
    // Could be set independently; using same value keeps it single-parameter.
    let rho_pos      = rho_landmark;

    let mut z_prev:            [f64; 2] = params.z0_hologram;
    let mut landmark_locus:    [f64; 2] = [0.0, 0.0];
    let mut display_delta:     [f64; 2] = [0.0, 0.0];
    let mut hologram_absolute: [f64; 2] = params.z0_hologram;
    let mut frames = Vec::with_capacity(params.n_frames);
    let mut total  = 0usize;
    const POS_CORR_OPS: usize = 4;  // 2 mul + 2 add for [u,v] position correction

    for t in 0..params.n_frames {
        let z_obs   = params.z_observed(t);
        let z_delta = [z_obs[0] - z_prev[0], z_obs[1] - z_prev[1]];

        let residual_before = [z_obs[0] - hologram_absolute[0],
                                z_obs[1] - hologram_absolute[1]];

        // ── Step 1: Velocity tracking (V0.8 operators) ──
        let inputs = AbrInputs {
            z_observed:    z_delta,
            z_predicted:   landmark_locus,
            display_locus: display_delta,
        };
        let r    = abr_measurement_update(&decl, &inputs);
        let r_e0 = r.g_after_r.get(0);

        landmark_locus[0] += rho_landmark * r_e0[0];
        landmark_locus[1] += rho_landmark * r_e0[1];
        display_delta[0]  -= r.display_correction[0];
        display_delta[1]  -= r.display_correction[1];

        // ── Step 2: Prediction (project position using velocity estimate) ──
        let hologram_pred = [hologram_absolute[0] + display_delta[0],
                              hologram_absolute[1] + display_delta[1]];

        // ── Step 3: Position correction (close the absolute loop) ──
        let pos_residual = [z_obs[0] - hologram_pred[0],
                             z_obs[1] - hologram_pred[1]];
        hologram_absolute[0] = hologram_pred[0] + rho_pos * pos_residual[0];
        hologram_absolute[1] = hologram_pred[1] + rho_pos * pos_residual[1];

        let residual_after = [z_obs[0] - hologram_absolute[0],
                               z_obs[1] - hologram_absolute[1]];
        let frame_ops = r.total_ops + POS_CORR_OPS;
        total += frame_ops;
        z_prev = z_obs;

        frames.push(SimFrame {
            frame:           t,
            t_ms:            (t as f64 / params.fps) * 1000.0,
            z_observed:      z_obs,
            hologram:        hologram_absolute,
            residual_before,
            residual_after,
            correction:      [rho_pos * pos_residual[0], rho_pos * pos_residual[1]],
            ops:             frame_ops,
            landmark_locus,
            display_delta,
        });
    }

    SimResult {
        frames,
        total_ops:      total,
        label:          "ABR predict-update (V0.9)",
        description:    "V0.9: velocity loop (V0.8) + predict step + ρ-bounded position correction",
        landmark_final: landmark_locus,
    }
}

// ─── ABR V1.0: 4-locus declarative topology (closes both loops) ──────────────

/// ABR V1.0: 4-locus chain — fully declarative position tracking.
///
/// V0.9 encoded the predict-update as explicit procedural steps outside the operators.
/// V1.0 declares the full topology: camera_feature → velocity_locus → position_locus →
/// display_update, and feeds z_obs (absolute position) as the camera_feature observable.
/// A single A→B→R pass handles both the velocity and position loops per frame.
///
/// Key design changes from V0.9:
///   • Observable: z_obs (absolute) instead of z_delta (velocity)
///   • Topology: 4 loci, 3 edges instead of 3 loci, 2 edges
///   • No separate predict step: B operator telescopes the chain
///     B(e₀) = z_obs − pos_est  (total position error, bypassing vel_est)
///   • All locus updates follow directly from R edge outputs
///
/// Mid-process initialization (V8 principle):
///   vel_est[0]  = [0, 0]        (declared prior; no velocity known at start)
///   pos_est[0]  = z0_hologram   (current display state is the position prior)
///   display[0]  = z0_hologram   (same — display is where hologram starts)
///
/// Per-frame update:
///   Step 1: place z_obs at camera_feature; run A→B→R on [z_obs, vel_est, pos_est, display]
///   Step 2: vel_est  += ρ[vel] · R(e₀)
///           pos_est  += ρ[pos] · R(e₁)
///           display  += R(e₂)        (terminal correction, no extra ρ)
///
/// Ops per frame: A(3) + B(2) + R(7) = 12 ops  (vs V0.9: 11, EKF: 500)
pub fn simulate_abr_v10(params: &PanParams) -> SimResult {
    use crate::declaration::MDeclaration;
    use crate::abr_step::{apply_a, apply_b, apply_r};

    let decl      = MDeclaration::ar_v10_update();
    let rho_vel   = decl.rho(1);  // ρ[velocity_locus] = 0.020  (χ=8)
    let rho_pos   = decl.rho(2);  // ρ[position_locus] = 1/60   (χ=4)

    let mut vel_est: [f64; 2] = [0.0, 0.0];             // velocity locus (declared prior)
    let mut pos_est: [f64; 2] = params.z0_hologram;     // position locus (from display state)
    let mut display: [f64; 2] = params.z0_hologram;     // display hologram (mid-process prior)
    let mut frames  = Vec::with_capacity(params.n_frames);
    let mut total   = 0usize;

    for t in 0..params.n_frames {
        let z_obs = params.z_observed(t);

        let residual_before = [z_obs[0] - display[0], z_obs[1] - display[1]];

        // ── Single A→B→R pass on 4-locus topology ──
        let locus_values: [[f64; 2]; 4] = [z_obs, vel_est, pos_est, display];
        let (g_a, ops_a) = apply_a(&decl, &locus_values);
        let (g_b, ops_b) = apply_b(&decl, &g_a);
        let (g_r, ops_r, _) = apply_r(&decl, &g_b);

        let r_e0 = g_r.get(0);  // camera_feature → velocity_locus
        let r_e1 = g_r.get(1);  // velocity_locus → position_locus
        let r_e2 = g_r.get(2);  // position_locus → display_update  (terminal correction)

        // ── Locus updates (external to operators, one per frame) ──
        vel_est[0] += rho_vel * r_e0[0];
        vel_est[1] += rho_vel * r_e0[1];
        pos_est[0] += rho_pos * r_e1[0];
        pos_est[1] += rho_pos * r_e1[1];
        display[0] += r_e2[0];
        display[1] += r_e2[1];

        let residual_after = [z_obs[0] - display[0], z_obs[1] - display[1]];
        let frame_ops = ops_a + ops_b + ops_r;
        total += frame_ops;

        frames.push(SimFrame {
            frame:           t,
            t_ms:            (t as f64 / params.fps) * 1000.0,
            z_observed:      z_obs,
            hologram:        display,
            residual_before,
            residual_after,
            correction:      r_e2,
            ops:             frame_ops,
            landmark_locus:  vel_est,   // vel_est stored in landmark_locus field for reporting
            display_delta:   pos_est,   // pos_est stored in display_delta field for reporting
        });
    }

    SimResult {
        frames,
        total_ops:      total,
        label:          "ABR declarative (V1.0)",
        description:    "V1.0: 4-locus chain, z_obs observable, single A→B→R pass — no separate predict step",
        landmark_final: vel_est,
    }
}

// ─── ABR V1.1: relational position space + carried edge state ────────────────

/// ABR V1.1: all loci in relational position space — architectural correction to V1.0.
///
/// V1.0 declaration error: locus 1 was named `velocity_locus` and described as carrying
/// velocity state, but its initialisation (`z0_hologram`) and its A input (`z_obs`, an
/// absolute growing position signal) are in the SAME position space as every other locus.
/// The mismatch was naming/framing only; the operators were correct, but the declaration
/// lied about what the locus represents.
///
/// V1.1 correction (declaration change only; operators identical):
///   1. Locus 1 renamed relay_a, locus 2 renamed relay_b — no temporal claims.
///   2. All four loci declared in the same relational position space as z_obs.
///   3. All three loci initialized at z0_hologram (the declared display prior).
///   4. "Edge data carried frame-to-frame" = locus values persist between frames.
///      This already happened in V1.0; V1.1 names it correctly as relational continuity.
///
/// Numerical behaviour is IDENTICAL to V1.0: same A/B/R operators, same ρ values,
/// same initial conditions, same observable → same ~60 px steady-state lag.
///
/// Why additive g-carry was removed (investigation log):
///   Attempts to implement g_combined = g_a + g_r_prev diverged exponentially because
///   the B operator telescopes values across the chain: B(e₀) = z_obs − relay_b (not just
///   z_obs − relay_a). Carrying the full R output feeds this telescoped value back as
///   additional B input, producing a Fibonacci-like growth in g_combined each frame.
///   Scaled versions (g_r.scale(α) for any α < 1) still diverge for a ramp input:
///   the carry time constant 1/(1−α) must exceed 1/ρ_a = 50 frames for meaningful lag
///   reduction, but at that scale the transient overshoot causes relay_a to oscillate
///   with growing amplitude. The stable α range (α < 1−ρ_a = 0.98) gives at best a
///   halving of lag (25 px) with 100× added complexity. Not worth the tradeoff for V1.1.
///
/// Why V0.9 achieved 0.025 px vs V1.1's 60 px (same operators):
///   V0.9 used a RELATIVE observable: z_obs = camera residual ≈ 0 at alignment.
///   V1.0/V1.1 use ABSOLUTE observable: z_obs = absolute pixel position (ramp).
///   The Kernel V8 operators are accurate for relative observables.
///   For absolute ramp tracking, minimum steady-state lag = pan_rate / ρ_base = 40 px.
///   This is a declaration choice (observable type), not a Kernel V8 deficiency.
///
/// Ops/frame: A(3) + B(2) + R(7) = 12 ops/frame  (same as V1.0)
/// Total for 600 frames: 12 × 600 = 7,200 ops  (vs EKF 300,000 ops = 41.7× lower cost)
pub fn simulate_abr_v11(params: &PanParams) -> SimResult {
    use crate::declaration::MDeclaration;
    use crate::abr_step::{apply_a, apply_b, apply_r};

    let decl  = MDeclaration::ar_v11_update();
    let rho_a = decl.rho(1);  // ρ[relay_a] = 0.020  (χ=8)
    let rho_b = decl.rho(2);  // ρ[relay_b] = 0.01667 (χ=4)

    // All loci start at the declared display prior.
    // "Edge data carried frame-to-frame" = these locus values persist across frames.
    // No separate g_carried accumulator — the relational continuity is in the loci.
    let mut relay_a: [f64; 2] = params.z0_hologram;
    let mut relay_b: [f64; 2] = params.z0_hologram;
    let mut display: [f64; 2] = params.z0_hologram;
    let mut frames = Vec::with_capacity(params.n_frames);
    let mut total  = 0usize;

    for t in 0..params.n_frames {
        let z_obs = params.z_observed(t);
        let residual_before = [z_obs[0] - display[0], z_obs[1] - display[1]];

        // ── V1.1: A(3) + B(2) + R(7) = 12 ops/frame  (operators identical to V1.0) ──
        let locus_values: [[f64; 2]; 4] = [z_obs, relay_a, relay_b, display];
        let (g_a, ops_a)    = apply_a(&decl, &locus_values);
        let (g_b, ops_b)    = apply_b(&decl, &g_a);
        let (g_r, ops_r, _) = apply_r(&decl, &g_b);

        let r_e0 = g_r.get(0);  // camera_feature → relay_a correction
        let r_e1 = g_r.get(1);  // relay_a → relay_b correction
        let r_e2 = g_r.get(2);  // relay_b → display_update (terminal correction)

        // Locus updates — all in relational position space, no temporal interpretation.
        relay_a[0] += rho_a * r_e0[0];
        relay_a[1] += rho_a * r_e0[1];
        relay_b[0] += rho_b * r_e1[0];
        relay_b[1] += rho_b * r_e1[1];
        display[0]  += r_e2[0];
        display[1]  += r_e2[1];
        // relay_a, relay_b, display PERSIST to the next iteration —
        // this is the "edge data carried frame-to-frame" in the sense that
        // the relational state (locus positions) evolves continuously.

        let residual_after = [z_obs[0] - display[0], z_obs[1] - display[1]];
        let frame_ops = ops_a + ops_b + ops_r;
        total += frame_ops;

        frames.push(SimFrame {
            frame:           t,
            t_ms:            (t as f64 / params.fps) * 1000.0,
            z_observed:      z_obs,
            hologram:        display,
            residual_before,
            residual_after,
            correction:      r_e2,
            ops:             frame_ops,
            landmark_locus:  relay_a,
            display_delta:   relay_b,
        });
    }

    SimResult {
        frames,
        total_ops:      total,
        label:          "ABR declarative (V1.1)",
        description:    "V1.1: relational position space (declaration correction), 12 ops/frame",
        landmark_final: relay_a,
    }
}

// ─── ABR V1.2: relay_a as z_prev delay buffer (V0.9 accuracy, 4-locus topology) ─

/// ABR V1.2: z_delta derived internally via relay_a delay buffer.
///
/// V1.2 closes the gap between V0.9 accuracy and V1.1's relational declaration:
///   V0.9 used z_delta (relative observable) — sub-pixel accuracy, 11 ops/frame
///   V1.1 used z_obs  (absolute observable) — ~50 px steady-state lag, 12 ops/frame
///   V1.2 derives z_delta INTERNALLY from relay_a — sub-pixel accuracy, 12 ops/frame
///
/// Mechanism:
///   relay_a is initialized to z0_hologram (declared display prior).
///   End of each frame: relay_a ← z_obs (1 relational assignment op).
///   Start of next frame: z_delta = z_obs − relay_a = z_obs[t] − z_obs[t-1] = 1 px/frame.
///   V0.9's 3-locus chain (ar_measurement_update) operates on z_delta.
///   Position loop closes via same predict-update as V0.9.
///
/// This is the 4-locus declared topology in V1.2:
///   camera_feature (z_obs[t]) → relay_a (z_obs[t-1]) → landmark → display_update
///   The relay_a → landmark edge carries z_delta, not z_obs — relative, not absolute.
///   Same operators, same ρ values as V0.9.
///
/// Ops:
///   A+B+R on 3-locus chain:  7 ops    (abr_measurement_update = ar_measurement_update)
///   Position correction:      4 ops    (predict + ρ·residual for [u,v])
///   relay_a ← z_obs:         1 op     (explicit relational assignment)
///   Total:                   12 ops/frame  (vs V0.9: 11; vs EKF: 500)
///
/// Expected at frame 599: residual < 0.1 px  (identical dynamics to V0.9, +1 op/frame)
pub fn simulate_abr_v12(params: &PanParams) -> SimResult {
    use crate::declaration::MDeclaration;
    use crate::abr_step::{AbrInputs, abr_measurement_update};

    // V0.9 inner chain declaration: camera_feature → landmark → display_update
    // V1.2 feeds z_delta (not z_obs) as the camera_feature observable.
    let decl         = MDeclaration::ar_measurement_update();
    let rho_landmark = decl.rho(1);  // ρ[landmark] = 0.025 × 4/(2+4) = 1/60  (velocity loop)
    let rho_pos      = rho_landmark; // same ρ for position correction (consistent with V0.9)

    // relay_a: z_obs[t-1] delay locus.
    // Initialized to z0_hologram (declared display prior — same as V0.9's z_prev).
    let mut relay_a:           [f64; 2] = params.z0_hologram;
    let mut landmark_locus:    [f64; 2] = [0.0, 0.0];         // motion-rate estimate (delta space)
    let mut display_delta:     [f64; 2] = [0.0, 0.0];         // per-frame display velocity
    let mut hologram_absolute: [f64; 2] = params.z0_hologram; // accumulated display position
    let mut frames = Vec::with_capacity(params.n_frames);
    let mut total  = 0usize;
    const POS_CORR_OPS: usize = 4;  // 2× mul + 2× add for [u,v] position correction
    const RELAY_OPS:    usize = 1;  // relay_a ← z_obs (1 relational assignment)

    for t in 0..params.n_frames {
        let z_obs = params.z_observed(t);

        // ── z_delta derived internally from relay_a (= z_obs[t-1]) ──
        // This is A(camera_feature, relay_a) in position space:
        //   z_delta = z_obs[t] − relay_a = z_obs[t] − z_obs[t-1] = pan_rate_u (constant)
        // The relative observable is extracted from the declared delay edge,
        // not passed in from outside.
        let z_delta = [z_obs[0] - relay_a[0], z_obs[1] - relay_a[1]];

        let residual_before = [z_obs[0] - hologram_absolute[0],
                                z_obs[1] - hologram_absolute[1]];

        // ── Step 1: V0.9 velocity chain on z_delta (7 ops) ──
        let inputs = AbrInputs {
            z_observed:    z_delta,          // relative observable (constant 1 px/frame)
            z_predicted:   landmark_locus,   // motion-rate estimate
            display_locus: display_delta,    // display velocity state
        };
        let r    = abr_measurement_update(&decl, &inputs);
        let r_e0 = r.g_after_r.get(0);

        landmark_locus[0] += rho_landmark * r_e0[0];
        landmark_locus[1] += rho_landmark * r_e0[1];
        display_delta[0]  -= r.display_correction[0];
        display_delta[1]  -= r.display_correction[1];

        // ── Step 2: Prediction (project position using velocity estimate) ──
        let hologram_pred = [hologram_absolute[0] + display_delta[0],
                              hologram_absolute[1] + display_delta[1]];

        // ── Step 3: Position correction — 4 ops (2 mul + 2 add for [u,v]) ──
        let pos_residual = [z_obs[0] - hologram_pred[0],
                             z_obs[1] - hologram_pred[1]];
        hologram_absolute[0] = hologram_pred[0] + rho_pos * pos_residual[0];
        hologram_absolute[1] = hologram_pred[1] + rho_pos * pos_residual[1];

        // ── Step 4: relay_a ← z_obs (1 relational assignment op) ──
        // relay_a now holds z_obs[t]; next frame derives z_delta from it.
        relay_a = z_obs;

        let residual_after = [z_obs[0] - hologram_absolute[0],
                               z_obs[1] - hologram_absolute[1]];
        let frame_ops = r.total_ops + POS_CORR_OPS + RELAY_OPS;
        total += frame_ops;

        frames.push(SimFrame {
            frame:           t,
            t_ms:            (t as f64 / params.fps) * 1000.0,
            z_observed:      z_obs,
            hologram:        hologram_absolute,
            residual_before,
            residual_after,
            correction:      [rho_pos * pos_residual[0], rho_pos * pos_residual[1]],
            ops:             frame_ops,
            landmark_locus,   // motion-rate estimate — converges to pan_rate_u
            display_delta,    // per-frame display velocity
        });
    }

    SimResult {
        frames,
        total_ops:      total,
        label:          "ABR declarative (V1.2)",
        description:    "V1.2: relay_a=z_prev delay; z_delta derived internally; V0.9 accuracy in 4-locus topology; 12 ops/frame",
        landmark_final: landmark_locus,
    }
}

// ─── EKF simulation ──────────────────────────────────────────────────────────

/// EKF continuous: runs full ekf_measurement_update() every frame.
///
/// V-09: Ops counted from the returned EkfResult each frame, not assigned.
/// Prior P is fixed (process model not declared); H·x_prior = 0 by construction.
/// EKF applies near-full correction (K ≈ near-identity in measurement space) each frame.
pub fn simulate_ekf_continuous(params: &PanParams) -> SimResult {
    use crate::ekf_step::{EkfInputs, ekf_measurement_update};

    let mut hologram: [f64; 2] = [0.0, 0.0];
    let mut frames   = Vec::with_capacity(params.n_frames);
    let mut total    = 0usize;

    for t in 0..params.n_frames {
        let z_obs           = params.z_observed(t);
        let residual_before = [z_obs[0] - hologram[0], z_obs[1] - hologram[1]];

        // V-09: run actual EKF with current z_observed; ops from returned result.
        // EKF prior is fixed (H·x_prior = 0 by construction), so the posterior
        // pixel position is H·x_posterior = H·K·(z_obs − 0) = pixel_correction.
        // The hologram is SET to the EKF's best estimate (not accumulated), because
        // the EKF's role is to produce a full-state estimate each step, not an
        // incremental correction. ABR applies incremental updates; EKF resets to
        // its posterior estimate.
        let inputs = EkfInputs::ar_example_with_z(z_obs[0], z_obs[1]);
        let r      = ekf_measurement_update(&inputs);
        let pc     = [r.pixel_correction.get(0, 0), r.pixel_correction.get(1, 0)];

        // SET to posterior pixel estimate (H·x_posterior = pixel_correction, since H·x_prior = 0)
        hologram[0] = pc[0];
        hologram[1] = pc[1];

        // V-11: residual after correction
        let residual_after = [z_obs[0] - hologram[0], z_obs[1] - hologram[1]];
        total += r.total_ops; // actual ops from this frame's EKF run

        frames.push(SimFrame {
            frame:           t,
            t_ms:            (t as f64 / params.fps) * 1000.0,
            z_observed:      z_obs,
            hologram,
            residual_before,
            residual_after,
            correction:      pc,
            ops:             r.total_ops,
            landmark_locus:  [0.0, 0.0],
            display_delta:   [0.0, 0.0],
        });
    }

    SimResult {
        frames,
        total_ops:      total,
        label:          "EKF continuous",
        description:    "Full ekf_measurement_update() every frame; ops from returned result (V-09)",
        landmark_final: [0.0, 0.0],  // N/A for EKF path
    }
}

// ─── Summary ─────────────────────────────────────────────────────────────────

pub struct SimSummary {
    pub abr:         SimResult,
    pub ekf:         SimResult,
    pub params:      PanParams,
}

impl SimSummary {
    pub fn run() -> Self {
        let params = PanParams::one_second_30fps();
        let abr    = simulate_abr(&params);
        let ekf    = simulate_ekf_continuous(&params);
        SimSummary { abr, ekf, params }
    }

    /// Extended 120-frame run for convergence study (OUTPUT 6).
    pub fn run_extended() -> Self {
        let params = PanParams::four_seconds_30fps();
        let abr    = simulate_abr(&params);
        let ekf    = simulate_ekf_continuous(&params);
        SimSummary { abr, ekf, params }
    }

    /// 300-frame V0.9 predict-update vs EKF (OUTPUT 7).
    pub fn run_v09() -> Self {
        let params = PanParams::twenty_seconds_30fps();
        let abr    = simulate_abr_predict_update(&params);
        let ekf    = simulate_ekf_continuous(&params);
        SimSummary { abr, ekf, params }
    }

    /// 600-frame V1.0 declarative 4-locus vs EKF (OUTPUT 8).
    pub fn run_v10() -> Self {
        let params = PanParams::twenty_seconds_30fps();
        let abr    = simulate_abr_v10(&params);
        let ekf    = simulate_ekf_continuous(&params);
        SimSummary { abr, ekf, params }
    }

    /// 600-frame V1.1 relational position space + carried edge state vs EKF (OUTPUT 9).
    pub fn run_v11() -> Self {
        let params = PanParams::twenty_seconds_30fps();
        let abr    = simulate_abr_v11(&params);
        let ekf    = simulate_ekf_continuous(&params);
        SimSummary { abr, ekf, params }
    }

    /// 600-frame V1.2: relay_a z_prev delay, z_delta derived internally (OUTPUT 10).
    /// Expected: sub-pixel accuracy (V0.9-class), 12 ops/frame vs EKF 500.
    pub fn run_v12() -> Self {
        let params = PanParams::twenty_seconds_30fps();
        let abr    = simulate_abr_v12(&params);
        let ekf    = simulate_ekf_continuous(&params);
        SimSummary { abr, ekf, params }
    }
}
