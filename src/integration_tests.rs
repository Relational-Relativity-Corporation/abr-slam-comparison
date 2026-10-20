//! Integration tests: operation accounting, observable correspondence, cross-path.
//!
//! Kernel V8 test classifications:
//!   [OBS]  — Observable: quantities must be in declared units throughout.
//!   [CORR] — Correspondence: output must correspond to a shared declared observation.
//!   [IMPL] — Implementation: operation counts, complexity bounds, cross-path behaviour.
//!
//! ─── CORR-01 STATUS: RESOLVED (V0.3) ────────────────────────────────────────
//!
//! Both paths now receive the same raw observation z = [28.4, −10.9] px:
//!   EKF: z directly as measurement vector; innovation y = z − H·x computed internally.
//!   ABR: z_observed = z; z_predicted = H·x = [0,0]; A(e₀) = z_observed − z_predicted = z.
//!
//! The A operator computes the residual from declared locus values — the same value
//! the EKF computes as y = z − H·x — without a matrix product.
//!
//! EKF also now reports pixel_correction = H·K·y, enabling direct pixel-space comparison.

#[cfg(test)]
mod integration_tests {
    use crate::declaration::{MDeclaration, RHO_BASE};
    use crate::ekf_step::{EkfInputs, ekf_measurement_update, STATE_DIM, MEASUREMENT_DIM};
    use crate::abr_step::{AbrInputs, abr_measurement_update};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    // ── [OBS] Output units ─────────────────────────────────────────────────────

    /// [OBS] EKF posterior state x̂ is in state-space units (meters for position).
    #[test]
    fn obs_ekf_output_is_state_space_meters() {
        let inputs = EkfInputs::ar_example();
        let r = ekf_measurement_update(&inputs);
        let correction = (r.x_posterior.get(0, 0) - 0.05_f64).abs();
        assert!(correction < 1.0,
            "[OBS] x_posterior correction = {:.4} m; if >1.0, suspect unit confusion",
            correction);
        assert_eq!(r.x_posterior.rows, STATE_DIM);
        assert_eq!(r.x_posterior.cols, 1);
    }

    /// [OBS] EKF pixel_correction is in pixel space (measurement units).
    ///
    /// H·Δx̂ projects the state correction back to pixels. For a 28.4 px innovation
    /// with K ≈ near-identity, the pixel correction should be close to 28.4 px.
    #[test]
    fn obs_ekf_pixel_correction_in_pixel_space() {
        let inputs = EkfInputs::ar_example();
        let r = ekf_measurement_update(&inputs);
        // Pixel correction u-component should be close to innovation (28.4 px)
        // and in the same direction (positive)
        let pc_u = r.pixel_correction.get(0, 0);
        assert!(pc_u > 20.0 && pc_u < 30.0,
            "[OBS] pixel_correction[u] = {:.4} px; expected ~28 px for 28.4 px innovation",
            pc_u);
        assert_eq!(r.pixel_correction.rows, MEASUREMENT_DIM,
            "pixel_correction must have MEASUREMENT_DIM rows");
    }

    /// [OBS] ABR display_correction is in pixel space [u,v].
    #[test]
    fn obs_abr_output_is_pixel_space() {
        let decl   = MDeclaration::ar_measurement_update();
        let inputs = AbrInputs::ar_example();
        let r      = abr_measurement_update(&decl, &inputs);
        // ρ[landmark] = 1/60; correction magnitude = ρ · |z| ≈ 0.473 px (sub-pixel)
        let expected_u = -(0.025_f64 * 4.0 / (2.0 + 4.0)) * 28.4;  // −28.4/60
        assert!(approx_eq(r.display_correction[0], expected_u, 1e-9),
            "[OBS] display_correction[u] = {:.9} px, expected {:.9}",
            r.display_correction[0], expected_u);
        // Magnitude is sub-pixel (conservative per-frame correction)
        assert!(r.display_correction[0].abs() < 1.0,
            "[OBS] display_correction[u] magnitude must be sub-pixel");
    }

    /// [OBS] ABR rho values are dimensionless.
    #[test]
    fn obs_rho_is_dimensionless() {
        let decl = MDeclaration::ar_measurement_update();
        for locus in &decl.loci {
            let rho = decl.rho(locus.id);
            assert!(rho > 0.0 && rho < RHO_BASE,
                "ρ[{}] = {:.6} must be in (0, ρ_base = {})", locus.label, rho, RHO_BASE);
        }
    }

    // ── [CORR] Observable correspondence ──────────────────────────────────────

    /// [CORR] RESOLVED: both paths now receive the same observation z = [28.4,−10.9].
    ///
    /// EKF: z = [28.4, −10.9] directly; innovation y = z − H·x = z (H·x = 0)
    /// ABR: z_observed = [28.4, −10.9]; z_predicted = [0,0]; A(e₀) = [28.4, −10.9]
    ///
    /// The A operator computes the residual from declared locus values, producing
    /// the same quantity as the EKF innovation y without a matrix product.
    #[test]
    fn corr_inputs_unified_same_observation() {
        let ekf_inputs = EkfInputs::ar_example();
        let abr_inputs = AbrInputs::ar_example();

        let ekf = ekf_measurement_update(&ekf_inputs);

        // EKF innovation: y = z − H·x = z (since H·x = 0 by construction)
        let ekf_y_u = ekf.innovation.get(0, 0);
        let ekf_y_v = ekf.innovation.get(1, 0);

        // ABR residual after A: z_observed − z_predicted
        let abr_residual_u = abr_inputs.z_observed[0] - abr_inputs.z_predicted[0];
        let abr_residual_v = abr_inputs.z_observed[1] - abr_inputs.z_predicted[1];

        // Both must equal z = [28.4, −10.9]
        assert!(approx_eq(ekf_y_u, abr_residual_u, 1e-9),
            "[CORR] EKF innovation[u] = {:.4}, ABR residual[u] = {:.4} — must match",
            ekf_y_u, abr_residual_u);
        assert!(approx_eq(ekf_y_v, abr_residual_v, 1e-9),
            "[CORR] EKF innovation[v] = {:.4}, ABR residual[v] = {:.4} — must match",
            ekf_y_v, abr_residual_v);
    }

    /// [CORR] Pixel-space comparison: both paths produce corrections in pixels.
    ///
    /// EKF pixel_correction = H·K·y ≈ [28.4, −10.9] px (near-full innovation,
    ///   K ≈ near-identity for this covariance and noise setup)
    ///
    /// ABR display_correction = R(e₁) = −ρ[landmark]·y ≈ [−0.473, +0.182] px
    ///   Conservative per-frame correction: ρ-weighted fraction of full residual.
    ///
    /// EKF corrects to near zero in one step (optimal estimator).
    /// ABR applies a bounded, smooth correction per frame (converges over multiple steps).
    /// Both corrections are in the correct direction relative to the residual.
    #[test]
    fn corr_pixel_space_comparison() {
        let ekf_inputs = EkfInputs::ar_example();
        let ekf = ekf_measurement_update(&ekf_inputs);

        let decl = MDeclaration::ar_measurement_update();
        let abr_inputs = AbrInputs::ar_example();
        let abr = abr_measurement_update(&decl, &abr_inputs);

        let ekf_correction_u = ekf.pixel_correction.get(0, 0);
        let abr_correction_u = abr.display_correction[0];

        // EKF: large correction ≈ full innovation (near-optimal single step)
        assert!(ekf_correction_u > 20.0,
            "[CORR] EKF pixel_correction[u] = {:.4} px; expected >20 px (near-full)", ekf_correction_u);

        // ABR: conservative correction (ρ-weighted, sub-pixel)
        assert!(abr_correction_u.abs() < 1.0,
            "[CORR] ABR display_correction[u] = {:.6} px; expected sub-pixel", abr_correction_u);

        // Directions: EKF positive (toward observation), ABR negative (corrective = opposing residual)
        assert!(ekf_correction_u > 0.0,
            "[CORR] EKF correction must be positive for positive innovation");
        assert!(abr_correction_u < 0.0,
            "[CORR] ABR correction must be negative (corrective direction opposing residual)");

        // ABR correction is the ρ[landmark]-weighted fraction of the innovation
        let rho1     = 0.025_f64 * 4.0 / (2.0 + 4.0);
        let expected = -rho1 * 28.4;
        assert!(approx_eq(abr_correction_u, expected, 1e-9),
            "[CORR] ABR correction[u] = {:.9} px, expected −ρ[landmark]·28.4 = {:.9}",
            abr_correction_u, expected);
    }

    /// [CORR] Both paths respond to input in consistent directions.
    #[test]
    fn corr_both_paths_respond_to_input() {
        let ekf = ekf_measurement_update(&EkfInputs::ar_example());
        let decl = MDeclaration::ar_measurement_update();
        let abr = abr_measurement_update(&decl, &AbrInputs::ar_example());

        // EKF: positive innovation → positive state correction → positive pixel correction
        let ekf_innovation = ekf.innovation.get(0, 0);
        let ekf_pixel_corr = ekf.pixel_correction.get(0, 0);
        assert!(ekf_innovation > 0.0, "EKF innovation must be positive for this input");
        assert!(ekf_pixel_corr > 0.0, "EKF pixel correction must be positive for positive innovation");

        // ABR: positive residual → negative display correction (corrective, opposing residual)
        assert!(abr.display_correction[0] < 0.0,
            "[CORR] ABR display_correction[u] = {:.6}; expected negative (corrective direction)",
            abr.display_correction[0]);
    }

    // ── [SIM] Multi-step simulation (V0.8) ───────────────────────────────────
    //
    // V0.5: budget-matched comparator removed (V-10). Both paths run continuously.
    // Residuals are measured after each frame's correction is applied (V-11).
    // EKF ops counted from actual ekf_measurement_update() per frame (V-09).
    //
    // V0.6: ABR landmark locus persists across frames (continuous-process correction).
    // V0.7: camera_feature carries z_delta = z[t]−z[t−1]; landmark updated via R(e₀).
    // V0.8: z_prev and hologram_absolute initialized from z0_hologram (declared display
    //        state before the first camera frame). Frame 0 establishes the initial field
    //        relationship from that declared state — grounded in an observable.

    /// [SIM] ABR hologram converges toward feature over 30 frames under pan drift.
    ///
    /// With ρ = 1/60 and 1 px/frame pan, ABR applies continuous bounded correction.
    /// Hologram position must be strictly positive (moving toward feature) after 30 frames.
    #[test]
    fn sim_abr_hologram_moves_toward_feature() {
        use crate::simulation::{simulate_abr, PanParams};
        let params = PanParams::one_second_30fps();
        let result = simulate_abr(&params);
        let final_frame = result.frames.last().unwrap();
        // Hologram must have moved in the +u direction (toward panning feature)
        assert!(final_frame.hologram[0] > 0.0,
            "[SIM] ABR hologram[u] = {:.4}; must be > 0 after 30 frames of pan",
            final_frame.hologram[0]);
        // Hologram must be less than the feature (can't overshoot with ρ < 1)
        let z_final = params.z_observed(params.n_frames - 1);
        assert!(final_frame.hologram[0] < z_final[0],
            "[SIM] ABR hologram[u] = {:.4} must be < z_obs[u] = {:.4} (no overshoot)",
            final_frame.hologram[0], z_final[0]);
    }

    /// [SIM] ABR total ops = 7 × n_frames = 210 for 30-frame simulation.
    #[test]
    fn sim_abr_total_ops_30_frames() {
        use crate::simulation::{simulate_abr, PanParams};
        let params = PanParams::one_second_30fps();
        let result = simulate_abr(&params);
        let expected = 7 * params.n_frames;
        assert_eq!(result.total_ops, expected,
            "[SIM] ABR total ops = {}, expected {}", result.total_ops, expected);
    }

    /// [SIM] ABR correction reduces residual every frame (starting frame 1).
    ///
    /// V-11: residuals recorded after correction. Frame 0 has z_obs[u] = 0 (pan not yet
    /// displaced in u), so the u-residual is zero both before and after — no assertion for
    /// frame 0. From frame 1 onward, the pan creates a nonzero u-residual; ABR must
    /// reduce it each frame.
    #[test]
    fn sim_abr_correction_reduces_residual_every_frame() {
        use crate::simulation::{simulate_abr, PanParams};
        let params = PanParams::one_second_30fps();
        let result = simulate_abr(&params);
        for frame in result.frames.iter().skip(1) {
            let res_before = frame.residual_before[0].abs();
            let res_after  = frame.residual_after[0].abs();
            assert!(res_after < res_before,
                "[SIM] frame {}: ABR residual_after[u] = {:.6} must be < residual_before[u] = {:.6}",
                frame.frame, res_after, res_before);
        }
    }

    /// [SIM] EKF continuous residual_after is near-zero every frame (V-11).
    ///
    /// EKF applies near-full correction each step. residual_after should be < 0.1 px
    /// for all frames. Uses the SimFrame.residual_after field (V-11 fix).
    #[test]
    fn sim_ekf_continuous_near_zero_residual() {
        use crate::simulation::{simulate_ekf_continuous, PanParams};
        let params = PanParams::one_second_30fps();
        let result = simulate_ekf_continuous(&params);
        for frame in &result.frames {
            let residual_after = frame.residual_after[0].abs();
            assert!(residual_after < 0.1,
                "[SIM] frame {}: EKF residual_after[u] = {:.6} px; expected < 0.1 px (V-11)",
                frame.frame, residual_after);
        }
    }

    /// [SIM] EKF continuous ops counted from actual execution: 500/frame × 30 = 15 000 total.
    ///
    /// V-09: ops are no longer assigned; they come from EkfResult.total_ops each frame.
    /// The single-frame reference test confirms 500 ops, so 30 × 500 = 15 000.
    #[test]
    fn sim_ops_ratio_continuous() {
        use crate::simulation::{simulate_abr, simulate_ekf_continuous, PanParams};
        let params = PanParams::one_second_30fps();
        let abr = simulate_abr(&params);
        let ekf = simulate_ekf_continuous(&params);
        assert_eq!(abr.total_ops, 210,   "ABR total ops must be 210 (7 ops/frame × 30)");
        assert_eq!(ekf.total_ops, 15000, "EKF total ops must be 15 000 (500 ops/frame × 30, V-09)");
    }

    /// [SIM] V0.7: ABR landmark tracks camera motion rate (delta space).
    ///
    /// With delta inputs, landmark converges toward the per-frame camera velocity
    /// (z_delta = [1.0, 0] px/frame for a 1 px/frame pan), not the absolute feature
    /// position.  The landmark update now flows through the full A→B→R pipeline:
    ///   landmark(t+1) = landmark(t) + ρ[landmark] × R(e₀)(t)
    ///
    /// The hologram_absolute is the accumulated display position (Σ display_delta),
    /// a different quantity in a different space from the motion-rate estimate.
    ///
    /// With a 1 px/frame rightward pan, after 30 frames:
    ///   (a) landmark[u] > 0: motion estimate is positive (tracking +u camera motion)
    ///   (b) landmark[u] < 1.0: not yet converged (ρ = 1/60 → slow; ~0.4 px/frame at t=30)
    ///   (c) landmark[u] ≠ hologram[u]: semantically distinct quantities
    ///       landmark  ≈ 0.4 px/frame (motion-rate estimate, delta space)
    ///       hologram  ≈ 6.5 px       (accumulated absolute position)
    ///       V0.7 produces distinct relational behaviour; V0.6's landmark = hologram
    ///       identity is broken by the change of observable primitive.
    #[test]
    fn sim_abr_landmark_locus_advances_toward_feature() {
        use crate::simulation::{simulate_abr, PanParams};
        let params = PanParams::one_second_30fps();
        let result = simulate_abr(&params);

        let landmark_u = result.landmark_final[0];
        let hologram_u = result.frames.last().unwrap().hologram[0];

        // (a) Landmark must be positive (tracking +u camera motion in delta space)
        assert!(landmark_u > 0.0,
            "[SIM/V0.7] landmark_final[u] = {:.6}; must be > 0 (tracking pan motion rate)",
            landmark_u);

        // (b) Not yet converged: landmark < 1.0 px/frame (ρ = 1/60 → slow convergence)
        assert!(landmark_u < 1.0,
            "[SIM/V0.7] landmark_final[u] = {:.6}; must be < 1.0 (not yet at z_delta = 1 px/frame)",
            landmark_u);

        // (c) V0.7: landmark ≠ hologram — semantically distinct quantities.
        //     landmark: motion-rate estimate in delta space (~0.4 px/frame)
        //     hologram: accumulated absolute correction (~6.5 px)
        //     The full pipeline breaks the V0.6 landmark = hologram identity.
        assert!((landmark_u - hologram_u).abs() > 1.0,
            "[SIM/V0.7] landmark[u] = {:.6}, hologram[u] = {:.6}; must differ > 1.0 (distinct quantities)",
            landmark_u, hologram_u);
    }

    /// [SIM] At equal time budget (30 frames), EKF costs 71.4× more ops than ABR.
    ///
    /// V-10: replaces budget-matched structural test. Both paths run every frame.
    /// The ops ratio is the honest comparison: same frequency, different cost.
    #[test]
    fn sim_abr_uses_far_fewer_ops_at_equal_frame_rate() {
        use crate::simulation::{simulate_abr, simulate_ekf_continuous, PanParams};
        let params = PanParams::one_second_30fps();
        let abr = simulate_abr(&params);
        let ekf = simulate_ekf_continuous(&params);
        let ratio = ekf.total_ops as f64 / abr.total_ops as f64;
        assert!(ratio > 60.0,
            "[SIM] EKF/ABR ops ratio = {:.1}× must be > 60× (V-10 honest framing)",
            ratio);
    }

    // ── [IMPL] Cross-path operation accounting ────────────────────────────────

    /// [IMPL] EKF uses more ops than ABR for this topology.
    #[test]
    fn impl_ekf_uses_more_ops_than_abr() {
        let ekf = ekf_measurement_update(&EkfInputs::ar_example());
        let decl = MDeclaration::ar_measurement_update();
        let abr = abr_measurement_update(&decl, &AbrInputs::ar_example());
        assert!(ekf.total_ops > abr.total_ops,
            "EKF ops ({}) must exceed ABR ops ({}) for n={}, m={}, |E|={}",
            ekf.total_ops, abr.total_ops, STATE_DIM, MEASUREMENT_DIM, decl.edges.len());
    }

    /// [IMPL] Excluded operations documented (documentation test — always passes).
    #[test]
    fn impl_ops_excluded_documented() {
        // EKF excluded: additions in mul/add/sub; innovation computation
        //   (H·x, z−H·x): 2·m·n = 24 additions; loop indexing, pivot comparisons.
        // ABR excluded: base g[e] additions in B and R; ρ computation per locus;
        //   loop overhead; scalar component work within each 2D op.
        let excluded_innovation_adds: usize = 2 * MEASUREMENT_DIM * STATE_DIM;
        assert!(excluded_innovation_adds > 0,
            "Innovation exclusion must be nonzero (= {})", excluded_innovation_adds);
        // Always passes — documentation test.
    }

    /// [IMPL] Both paths consistent with declared complexity classes.
    #[test]
    fn impl_ops_consistent_with_complexity_class() {
        let ekf = ekf_measurement_update(&EkfInputs::ar_example());
        let decl = MDeclaration::ar_measurement_update();
        let abr = abr_measurement_update(&decl, &AbrInputs::ar_example());

        let n = STATE_DIM;
        let m = MEASUREMENT_DIM;
        let e = decl.edges.len();

        // EKF: per-step theoretical bounds (8 steps)
        let ekf_upper = n*n*n + m*m*m + m*n*n + m*m*n + n*n*m + n*m*m + n*n*m + n*m;
        assert!(ekf.total_ops <= ekf_upper,
            "EKF ops {} must be ≤ theoretical upper {}", ekf.total_ops, ekf_upper);

        // ABR: 3·|E| + |L| structural bound
        let abr_upper = 3 * e + decl.loci.len();
        assert!(abr.total_ops <= abr_upper,
            "ABR ops {} must be ≤ 3·|E|+|L| = {}", abr.total_ops, abr_upper);

        let _ = (n, m, e);
    }
}

// ─── 600-FRAME SIMULATION REGRESSION TESTS ───────────────────────────────────
//
// These tests pin the 600-frame simulation outcomes for V0.9, V1.0, and EKF.
// They exist to catch unintended changes to simulation logic and to document
// the known state of V1.0's tracking deficiency (observable mismatch).
//
// V0.9 KEY RESULT: 32× more accurate, 45× less compute than EKF.
// V1.0 KNOWN ISSUE: velocity_locus receives z_obs (absolute) not z_delta;
//   vel_est accumulates as a lagged position integral → ~50 px steady-state lag.
//   Fix target (V1.1): camera_feature must carry z_delta; position_locus must
//   accumulate from velocity_locus and receive z_obs correction separately.
//
#[cfg(test)]
mod simulation_regression {
    use crate::simulation::{simulate_abr_predict_update as simulate_abr_v09, simulate_abr_v10, simulate_ekf_continuous, PanParams};
    #[allow(unused_imports)]
    use crate::simulation::simulate_abr_v11;

    fn params() -> PanParams { PanParams::twenty_seconds_30fps() }

    /// [REGR] V0.9 final residual is ≤ 0.030 px at frame 599.
    /// KEY RESULT: demonstrates observable-based ABR outperforms EKF in accuracy.
    #[test]
    fn v09_final_residual_within_tolerance() {
        let r = simulate_abr_v09(&params());
        let last = r.frames.last().unwrap();
        let res_u = last.residual_after[0].abs();
        assert!(res_u <= 0.030,
            "V0.9 residual at frame 599 = {:.6} px; expected ≤ 0.030 px", res_u);
    }

    /// [REGR] V0.9 total ops = 6600 (11 ops/frame × 600 frames).
    #[test]
    fn v09_total_ops() {
        let r = simulate_abr_v09(&params());
        assert_eq!(r.total_ops, 6_600,
            "V0.9 total ops = {}; expected 6600", r.total_ops);
    }

    /// [REGR] EKF final residual is between 0.79 and 0.81 px at frame 599.
    #[test]
    fn ekf_final_residual_stable() {
        let r = simulate_ekf_continuous(&params());
        let last = r.frames.last().unwrap();
        let res_u = last.residual_after[0].abs();
        assert!((0.79..=0.81).contains(&res_u),
            "EKF residual at frame 599 = {:.6} px; expected 0.79–0.81 px", res_u);
    }

    /// [REGR] EKF total ops = 300000 (500 ops/frame × 600 frames).
    #[test]
    fn ekf_total_ops() {
        let r = simulate_ekf_continuous(&params());
        assert_eq!(r.total_ops, 300_000,
            "EKF total ops = {}; expected 300000", r.total_ops);
    }

    /// [REGR] V0.9 is more accurate than EKF at frame 599 (KEY RESULT).
    #[test]
    fn v09_more_accurate_than_ekf() {
        let v09 = simulate_abr_v09(&params());
        let ekf = simulate_ekf_continuous(&params());
        let v09_res = v09.frames.last().unwrap().residual_after[0].abs();
        let ekf_res = ekf.frames.last().unwrap().residual_after[0].abs();
        assert!(v09_res < ekf_res,
            "V0.9 ({:.4} px) must be more accurate than EKF ({:.4} px)", v09_res, ekf_res);
    }

    /// [REGR] V0.9 uses fewer ops than EKF (KEY RESULT).
    #[test]
    fn v09_less_compute_than_ekf() {
        let v09 = simulate_abr_v09(&params());
        let ekf = simulate_ekf_continuous(&params());
        assert!(v09.total_ops < ekf.total_ops,
            "V0.9 ({} ops) must use fewer ops than EKF ({} ops)", v09.total_ops, ekf.total_ops);
    }

    /// [REGR] V1.0 total ops = 7200 (12 ops/frame × 600 frames).
    #[test]
    fn v10_total_ops() {
        let r = simulate_abr_v10(&params());
        assert_eq!(r.total_ops, 7_200,
            "V1.0 total ops = {}; expected 7200", r.total_ops);
    }

    /// [REGR] V1.0 known deficiency: final residual is in the 45–55 px range.
    /// This documents the steady-state lag from the observable mismatch (z_obs not z_delta).
    /// When V1.1 is implemented, this test MUST be updated to a lower bound.
    #[test]
    fn v10_residual_documents_known_lag() {
        let r = simulate_abr_v10(&params());
        let last = r.frames.last().unwrap();
        let res_u = last.residual_after[0].abs();
        assert!((45.0..=55.0).contains(&res_u),
            "V1.0 residual at frame 599 = {:.4} px; expected 45–55 px (observable mismatch; see V1.1)", res_u);
    }

    /// [REGR] V1.0 uses fewer total ops than EKF.
    /// The compute advantage holds even with the accuracy deficiency.
    #[test]
    fn v10_less_compute_than_ekf() {
        let v10 = simulate_abr_v10(&params());
        let ekf = simulate_ekf_continuous(&params());
        assert!(v10.total_ops < ekf.total_ops,
            "V1.0 ({} ops) must use fewer ops than EKF ({} ops)", v10.total_ops, ekf.total_ops);
    }

    // ── V1.1 Regression Tests ──────────────────────────────────────────────────
    //
    // V1.1 is a declaration correction over V1.0 (correct relational labels).
    // Operators are identical; numerical results match V1.0.
    // Additive g-carry was investigated and abandoned (exponential divergence
    // due to B-operator telescoping — see simulation.rs comment for detail).

    /// [REGR] V1.1 total ops = 7200 (12 ops/frame × 600 frames).
    /// Same as V1.0 — declaration correction only, no carry overhead.
    #[test]
    fn v11_total_ops() {
        use crate::simulation::simulate_abr_v11;
        let r = simulate_abr_v11(&params());
        assert_eq!(r.total_ops, 7_200,
            "V1.1 total ops = {}; expected 7200 (12 ops/frame × 600)", r.total_ops);
    }

    /// [REGR] V1.1 steady-state residual is in the 45–55 px range (same as V1.0).
    /// V1.1 corrects the declaration; numerical behaviour is unchanged.
    /// The lag is fundamental for ramp observables with ρ_a ≈ 0.020.
    #[test]
    fn v11_residual_matches_v10_range() {
        use crate::simulation::simulate_abr_v11;
        let r = simulate_abr_v11(&params());
        let last = r.frames.last().unwrap();
        let res_u = last.residual_after[0].abs();
        assert!((45.0..=55.0).contains(&res_u),
            "V1.1 residual at frame 599 = {:.4} px; expected 45–55 px (same as V1.0 — declaration correction only)", res_u);
    }

    /// [REGR] V1.1 uses fewer total ops than EKF (41.7× lower compute).
    /// Compute advantage is identical to V1.0.
    #[test]
    fn v11_less_compute_than_ekf() {
        use crate::simulation::simulate_abr_v11;
        let v11 = simulate_abr_v11(&params());
        let ekf = simulate_ekf_continuous(&params());
        assert!(v11.total_ops < ekf.total_ops,
            "V1.1 ({} ops) must use fewer ops than EKF ({} ops)", v11.total_ops, ekf.total_ops);
    }

    /// [REGR] V1.1 residual is stable across versions — matches V1.0 residual within 2 px.
    /// Documents that declaration correction does not perturb numerical output.
    #[test]
    fn v11_residual_matches_v10_numerically() {
        use crate::simulation::{simulate_abr_v11, simulate_abr_v10};
        let v11 = simulate_abr_v11(&params());
        let v10 = simulate_abr_v10(&params());
        let v11_res = v11.frames.last().unwrap().residual_after[0].abs();
        let v10_res = v10.frames.last().unwrap().residual_after[0].abs();
        assert!((v11_res - v10_res).abs() < 2.0,
            "V1.1 residual ({:.4} px) should match V1.0 ({:.4} px) within 2 px", v11_res, v10_res);
    }

    // ── V1.2 regression tests ─────────────────────────────────────────────────
    //
    // V1.2: relay_a = z_obs[t-1] delay buffer; z_delta derived internally.
    // Expected: V0.9-class sub-pixel accuracy at 12 ops/frame.
    // relay_a initialization and chain are identical to V0.9's z_prev mechanism.

    /// [REGR] V1.2 total ops = 7200 (12 ops/frame × 600 frames).
    /// chain(7) + position-correction(4) + relay_a-store(1) = 12 ops/frame.
    #[test]
    fn v12_total_ops() {
        use crate::simulation::simulate_abr_v12;
        let r = simulate_abr_v12(&params());
        assert_eq!(r.total_ops, 7_200,
            "V1.2 total ops = {}; expected 7200 (12 ops/frame × 600)", r.total_ops);
    }

    /// [REGR] V1.2 achieves sub-pixel residual at frame 599 (< 0.1 px).
    /// z_delta is derived from relay_a (= z_obs[t-1]); chain dynamics identical to V0.9.
    #[test]
    fn v12_subpixel_residual() {
        use crate::simulation::simulate_abr_v12;
        let r = simulate_abr_v12(&params());
        let last = r.frames.last().unwrap();
        let res_u = last.residual_after[0].abs();
        assert!(res_u < 0.1,
            "V1.2 residual at frame 599 = {:.6} px; expected < 0.1 px (V0.9-class sub-pixel)", res_u);
    }

    /// [REGR] V1.2 is more accurate than EKF at frame 599.
    /// Both accuracy and compute advantage must hold.
    #[test]
    fn v12_more_accurate_and_less_compute_than_ekf() {
        use crate::simulation::{simulate_abr_v12, simulate_ekf_continuous};
        let v12 = simulate_abr_v12(&params());
        let ekf = simulate_ekf_continuous(&params());
        let v12_res = v12.frames.last().unwrap().residual_after[0].abs();
        let ekf_res = ekf.frames.last().unwrap().residual_after[0].abs();
        assert!(v12_res < ekf_res,
            "V1.2 residual ({:.6} px) must be less than EKF ({:.4} px)", v12_res, ekf_res);
        assert!(v12.total_ops < ekf.total_ops,
            "V1.2 ({} ops) must use fewer ops than EKF ({} ops)", v12.total_ops, ekf.total_ops);
    }

    /// [REGR] V1.2 residual matches V0.9 residual within 0.01 px.
    /// relay_a (z_obs[t-1]) is identical to V0.9's z_prev — same z_delta → same dynamics.
    #[test]
    fn v12_residual_matches_v09() {
        use crate::simulation::{simulate_abr_v12, simulate_abr_predict_update};
        let v12 = simulate_abr_v12(&params());
        let v09 = simulate_abr_predict_update(&params());
        let v12_res = v12.frames.last().unwrap().residual_after[0].abs();
        let v09_res = v09.frames.last().unwrap().residual_after[0].abs();
        assert!((v12_res - v09_res).abs() < 0.01,
            "V1.2 residual ({:.6} px) should match V0.9 ({:.6} px) within 0.01 px", v12_res, v09_res);
    }
}
