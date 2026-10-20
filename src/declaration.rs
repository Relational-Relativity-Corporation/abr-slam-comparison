//! M Regional Instantiation
//!
//! Origin:    Robin Macomber, Metatron Dynamics, Inc.
//! Generator: Claude (Kernel V8, session 2026-10-19)
//! Region:    Single measurement update step, head-mounted AR
//!
//! ─── Observable Extent ────────────────────────────────────────────────────────
//! Device:  Microsoft HoloLens 2
//! Sources: Microsoft HoloLens 2 hardware specification (public)
//!   - Display resolution:  >2,500 radiants (light points per radian)
//!   - Angular resolution:  47 pixels per degree
//!   - Frame rate:          1080p at 30 Hz
//!   - IMU:                 3-axis accelerometer, gyroscope, magnetometer
//!
//! ─── Declared Parameters ──────────────────────────────────────────────────────
//! ρ_base = 0.025
//!   Derivation: 2,500 radiants / 100,000 scale factor = 0.025
//!   Meaning:    baseline saturating influence per declared locus
//!
//! χ₀ = 2.0 pixels
//!   Derivation: minimum detectable displacement at 47 px/degree
//!               1° ≈ 47 px → sub-degree threshold ≈ 2 px
//!   Meaning:    observable extent at which saturation begins
//!
//! ─── Regional Orientation ─────────────────────────────────────────────────────
//!   Direction of declared relation: camera_feature → landmark → display_update
//!   Interior locus (admissibility): landmark
//!     landmark = target(e₀) = source(e₁) for independently declared e₀, e₁
//!     → full-operator A→B→R admissibility satisfied (Kernel V8 §11)
//!
//! ─── Regional Admission Rule ──────────────────────────────────────────────────
//!   Any locus admitted to this region must:
//!     (a) carry a declared observable extent χ in pixels
//!     (b) connect to at least one other locus via an edge with declared EdgeProvenance
//!     (c) be located within the HoloLens 2 visual field (nominal 52° × 29° FOV)
//!
//! ─── Treatment Boundary ───────────────────────────────────────────────────────
//!   EKF comparator: retains conventional matrix mathematics throughout.
//!                   No Kernel V8 operators applied to EKF path.
//!   ABR treatment:  operators A, B, R applied only to declared topology below.
//!                   No legacy statistical estimation on ABR path.

// ── Declared Parameters ────────────────────────────────────────────────────────

/// Saturating weight base.
/// Provenance: HoloLens 2 radiants density (2,500 radiants / 100,000).
pub const RHO_BASE: f64 = 0.025;

/// Observable extent threshold (pixels) at which saturation begins.
/// Provenance: HoloLens 2 angular resolution, 47 px/degree → ~2 px minimum displacement.
pub const CHI_0: f64 = 2.0;

// ── Edge Provenance ────────────────────────────────────────────────────────────

/// Every declared edge must carry exactly one EdgeProvenance.
/// Kernel V8 §11: Continuation | Coupling | Persistence
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeProvenance {
    /// Same locus tracked across successive time steps.
    Continuation,
    /// Causal link between two distinct, independently declared loci.
    Coupling,
    /// Locus retained without active observation across measurement cycles.
    Persistence,
}

// ── Locus ──────────────────────────────────────────────────────────────────────

/// A declared point in the relational topology.
#[derive(Debug, Clone)]
pub struct Locus {
    pub id:    usize,
    pub label: &'static str,
    /// Observable extent at this locus (pixels). Origin-supplied.
    pub chi:   f64,
}

// ── Relational Edge ────────────────────────────────────────────────────────────

/// A declared directed relation between two loci.
#[derive(Debug, Clone)]
pub struct RelationalEdge {
    pub id:         usize,
    pub source:     usize,        // id of source locus
    pub target:     usize,        // id of target locus
    pub provenance: EdgeProvenance,
    /// Edge value (populated from observable measurement; not assumed).
    pub value:      f64,
}

// ── M Declaration ─────────────────────────────────────────────────────────────

/// Complete M Regional Declaration for this comparison.
pub struct MDeclaration {
    pub region_label: &'static str,
    pub loci:         Vec<Locus>,
    pub edges:        Vec<RelationalEdge>,
    pub rho_base:     f64,
    pub chi_0:        f64,
}

impl MDeclaration {
    /// Construct the declared AR measurement update topology.
    ///
    /// Loci:
    ///   0: camera_feature  — detected image feature, χ = 8.0 px (feature detector window)
    ///   1: landmark        — matched 3-D map point, χ = 4.0 px (landmark patch extent)
    ///   2: display_update  — corrected display pixel, χ = 2.0 px (HoloLens 2 min disp.)
    ///
    /// Edges:
    ///   e₀: camera_feature → landmark   (Coupling) — measurement association
    ///   e₁: landmark → display_update   (Coupling) — correction propagation
    ///
    /// Interior locus: landmark (id = 1)
    ///   landmark = target(e₀) and source(e₁), with e₀ and e₁ independently declared.
    ///   Full-operator admissibility condition satisfied.
    pub fn ar_measurement_update() -> Self {
        let loci = vec![
            Locus { id: 0, label: "camera_feature", chi: 8.0 },
            Locus { id: 1, label: "landmark",        chi: 4.0 },
            Locus { id: 2, label: "display_update",  chi: 2.0 },
        ];

        let edges = vec![
            RelationalEdge {
                id:         0,
                source:     0,  // camera_feature
                target:     1,  // landmark
                provenance: EdgeProvenance::Coupling,
                value:      0.0, // populated from observable measurement residual
            },
            RelationalEdge {
                id:         1,
                source:     1,  // landmark
                target:     2,  // display_update
                provenance: EdgeProvenance::Coupling,
                value:      0.0, // populated by B/R propagation
            },
        ];

        MDeclaration {
            region_label: "AR measurement update — HoloLens 2, single-step",
            loci,
            edges,
            rho_base: RHO_BASE,
            chi_0:    CHI_0,
        }
    }

    /// Saturating weight ρ at locus `locus_id`.
    /// ρ[i] = ρ_base · χ[i] / (χ₀ + χ[i])
    pub fn rho(&self, locus_id: usize) -> f64 {
        let chi = self.loci[locus_id].chi;
        self.rho_base * chi / (self.chi_0 + chi)
    }

    /// Construct the V1.0 AR update topology — 4 loci, 3 edges, z_obs (absolute) observable.
    ///
    /// V1.0 replaces V0.9's procedural predict-update with a fully declarative 4-locus chain.
    /// Observable: z_obs (absolute pixel position) placed at camera_feature each frame.
    /// Single A→B→R pass per frame handles both the velocity and position loops.
    ///
    /// Loci:
    ///   0: camera_feature  — absolute observed pixel position (z_obs), χ = 8.0 px
    ///   1: velocity_locus  — prediction state (learns z_obs trend),  χ = 8.0 px
    ///   2: position_locus  — corrected position estimate,            χ = 4.0 px
    ///   3: display_update  — hologram display position,              χ = 2.0 px
    ///
    /// Edges:
    ///   e₀: camera_feature  → velocity_locus   (Coupling) — feeds z_obs into prediction
    ///   e₁: velocity_locus  → position_locus   (Coupling) — prediction → correction
    ///   e₂: position_locus  → display_update   (Coupling) — position drives display
    ///
    /// Interior loci: velocity_locus (id=1) and position_locus (id=2) are both interior.
    ///
    /// A→B→R trace for chain 0→1→2→3 (B telescopes the chain):
    ///   A(e₀) = z_obs − vel_est         (position-space residual)
    ///   A(e₁) = vel_est − pos_est       (prediction residual)
    ///   A(e₂) = pos_est − display       (display correction)
    ///   B(e₀) = A(e₀) + A(e₁) = z_obs − pos_est    (B telescopes: total position error)
    ///   B(e₁) = A(e₁) + A(e₂) = vel_est − display
    ///   B(e₂) = A(e₂)          = pos_est − display
    ///   R(e₂) = B(e₂) − ρ[pos]·B(e₁)  (terminal display correction)
    ///
    /// Locus updates (external, per frame):
    ///   vel_est  += ρ[vel] · R(e₀)     (vel_est tracks z_obs through B-telescoped error)
    ///   pos_est  += ρ[pos] · R(e₁)     (pos_est tracks vel_est; sign follows z_obs direction)
    ///   display  += R(e₂)              (display updated by terminal correction)
    pub fn ar_v10_update() -> Self {
        let loci = vec![
            Locus { id: 0, label: "camera_feature",  chi: 8.0 },
            Locus { id: 1, label: "velocity_locus",  chi: 8.0 },  // ρ = 0.020
            Locus { id: 2, label: "position_locus",  chi: 4.0 },  // ρ = 1/60 ≈ 0.01667
            Locus { id: 3, label: "display_update",  chi: 2.0 },
        ];

        let edges = vec![
            RelationalEdge {
                id:         0,
                source:     0,  // camera_feature
                target:     1,  // velocity_locus
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
            RelationalEdge {
                id:         1,
                source:     1,  // velocity_locus
                target:     2,  // position_locus
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
            RelationalEdge {
                id:         2,
                source:     2,  // position_locus
                target:     3,  // display_update
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
        ];

        MDeclaration {
            region_label: "AR V1.0 update — 4-locus chain, z_obs (absolute) observable",
            loci,
            edges,
            rho_base: RHO_BASE,
            chi_0:    CHI_0,
        }
    }

    /// Construct the V1.1 AR update topology — 4 loci, 3 edges, z_obs observable,
    /// all loci in relational position space, edge state carried frame-to-frame.
    ///
    /// V1.1 corrects the V1.0 observable mismatch:
    ///   - All locus labels reflect relational position (no velocity naming)
    ///   - R output is carried as prior g to the next frame (integral feedback)
    ///   - Carried state is combined with fresh A output before B/R: g = g_a + g_carried
    ///
    /// Loci — all carry absolute position in pixels (same relational space):
    ///   0: camera_feature  — observed pixel position (z_obs), χ = 8.0 px
    ///   1: relay_a         — first relational relay,          χ = 8.0 px
    ///   2: relay_b         — second relational relay,         χ = 4.0 px
    ///   3: display_update  — hologram display position,       χ = 2.0 px
    ///
    /// Edges:
    ///   e₀: camera_feature → relay_a      (Coupling)
    ///   e₁: relay_a        → relay_b      (Coupling)
    ///   e₂: relay_b        → display_update (Coupling)
    ///
    /// Per-frame loop (V1.1):
    ///   g_a        = apply_a(&decl, &locus_values)       // fresh relational snapshot
    ///   g_combined = g_a + g_carried                     // + carried prior (3 ops)
    ///   g_b        = apply_b(&decl, &g_combined)         // B propagates combined
    ///   g_r        = apply_r(&decl, &g_b)                // R applies ρ-weighted correction
    ///   relay_a   += ρ_a · R(e₀)
    ///   relay_b   += ρ_b · R(e₁)
    ///   display   += R(e₂)
    ///   g_carried  = g_r                                 // carry to next frame
    ///
    /// Ops/frame: A(3) + carry(3) + B(2) + R(7) = 15 ops/frame
    pub fn ar_v11_update() -> Self {
        let loci = vec![
            Locus { id: 0, label: "camera_feature",  chi: 8.0 },
            Locus { id: 1, label: "relay_a",         chi: 8.0 },
            Locus { id: 2, label: "relay_b",         chi: 4.0 },
            Locus { id: 3, label: "display_update",  chi: 2.0 },
        ];

        let edges = vec![
            RelationalEdge {
                id:         0,
                source:     0,  // camera_feature
                target:     1,  // relay_a
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
            RelationalEdge {
                id:         1,
                source:     1,  // relay_a
                target:     2,  // relay_b
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
            RelationalEdge {
                id:         2,
                source:     2,  // relay_b
                target:     3,  // display_update
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
        ];

        MDeclaration {
            region_label: "AR V1.1 update — 4-locus relational chain, z_obs observable, carried edge state",
            loci,
            edges,
            rho_base: RHO_BASE,
            chi_0:    CHI_0,
        }
    }

    /// Construct the V1.2 AR update topology — relay_a as z_prev delay locus.
    ///
    /// V1.2 achieves V0.9 accuracy (sub-pixel) inside the declared 4-locus topology
    /// by making relay_a a 1-frame delay locus: relay_a holds z_obs[t-1].
    ///
    /// Architecture:
    ///   relay_a = z_obs at END of each frame  (explicit relational assignment, 1 op)
    ///   z_delta = z_obs − relay_a             (A-type derivation: = z_obs[t] − z_obs[t-1])
    ///   V0.9's 3-locus chain (camera_feature → landmark → display_update) operates on z_delta.
    ///   Position loop closes via predict-update (same as V0.9).
    ///
    /// Observable: z_delta (relative, constant 1 px/frame for linear pan)
    /// z_delta is derived INTERNALLY from relay_a — no external z_delta input.
    ///
    /// Loci — 4 loci, all in relational position space:
    ///   0: camera_feature  — z_obs[t] (absolute), χ = 8.0 px
    ///   1: relay_a         — z_obs[t-1] delay buffer, χ = 8.0 px
    ///                        set = z_obs at end of each frame (relational persistence)
    ///   2: landmark        — motion-rate estimate (converges to z_delta = 1 px/frame),
    ///                        χ = 4.0 px; operates in delta space via z_delta input
    ///   3: display_update  — hologram display velocity (per-frame correction), χ = 2.0 px
    ///
    /// Operator chain (per frame, on inner 3-locus sub-chain):
    ///   inner chain: {z_delta, landmark, display_delta}  (same as ar_measurement_update)
    ///   A(e₀) = z_delta − landmark    (velocity residual)
    ///   A(e₁) = landmark − display    (display residual)
    ///   B(e₀) = A(e₀) + A(e₁) = z_delta − display_delta
    ///   B(e₁) = A(e₁)
    ///   R — ρ-weighted corrections applied
    ///
    /// Ops:
    ///   chain (A+B+R): 7 ops   (same as V0.9 / ar_measurement_update)
    ///   position correction: 4 ops  (predict step + ρ·residual for [u,v])
    ///   relay_a store: 1 op         (relay_a ← z_obs — relational assignment)
    ///   Total: 12 ops/frame
    ///
    /// Expected at frame 599: residual < 0.1 px  (V0.9-class sub-pixel accuracy)
    ///
    /// Note: Operator calls use ar_measurement_update() (the 3-locus inner chain).
    ///       ar_v12_update() declares the full 4-locus topology for provenance only.
    pub fn ar_v12_update() -> Self {
        let loci = vec![
            Locus { id: 0, label: "camera_feature",  chi: 8.0 },
            Locus { id: 1, label: "relay_a",          chi: 8.0 },  // z_prev delay locus
            Locus { id: 2, label: "landmark",          chi: 4.0 },  // delta-space motion estimate
            Locus { id: 3, label: "display_update",   chi: 2.0 },
        ];

        let edges = vec![
            RelationalEdge {
                id:         0,
                source:     0,  // camera_feature
                target:     1,  // relay_a  (z_delay edge: camera_feature[t] → relay_a[t-1])
                provenance: EdgeProvenance::Persistence,  // relay_a = z_obs[t-1]
                value:      0.0,
            },
            RelationalEdge {
                id:         1,
                source:     1,  // relay_a
                target:     2,  // landmark  (z_delta = z_obs - relay_a feeds this edge)
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
            RelationalEdge {
                id:         2,
                source:     2,  // landmark
                target:     3,  // display_update
                provenance: EdgeProvenance::Coupling,
                value:      0.0,
            },
        ];

        MDeclaration {
            region_label: "AR V1.2 update — relay_a z_prev delay; z_delta derived internally; V0.9 accuracy",
            loci,
            edges,
            rho_base: RHO_BASE,
            chi_0:    CHI_0,
        }
    }

    /// Return successor edges for a given locus (edges whose source == locus_id).
    pub fn successors(&self, locus_id: usize) -> Vec<&RelationalEdge> {
        self.edges.iter().filter(|e| e.source == locus_id).collect()
    }

    /// Return predecessor edges for a given locus (edges whose target == locus_id).
    pub fn predecessors(&self, locus_id: usize) -> Vec<&RelationalEdge> {
        self.edges.iter().filter(|e| e.target == locus_id).collect()
    }

    /// Validate the declaration for structural integrity.
    ///
    /// Returns `Ok(())` if all invariants hold, or `Err(msg)` naming the first
    /// violation found. Does NOT repair: caller must resolve and re-declare.
    ///
    /// Checked invariants:
    ///   (a) chi_0 > 0.0   — prevents singularity in ρ formula at χ = −χ₀
    ///   (b) rho_base > 0.0 — zero base means R operator has no effect
    ///   (c) every edge source and target is a valid locus id
    ///   (d) every locus chi > 0.0  — sub-resolution loci produce ρ = 0
    ///   (e) at least one interior locus exists (full-operator admissibility)
    pub fn validate(&self) -> Result<(), String> {
        if self.chi_0 <= 0.0 {
            return Err(format!(
                "chi_0 = {} is not positive; ρ formula undefined at χ = −χ₀. Return to Origin.",
                self.chi_0
            ));
        }
        if self.rho_base <= 0.0 {
            return Err(format!(
                "rho_base = {} is not positive; R operator has no effect. Return to Origin.",
                self.rho_base
            ));
        }
        let n = self.loci.len();
        for edge in &self.edges {
            if edge.source >= n {
                return Err(format!(
                    "Edge {} source id {} exceeds locus count {}. Return to Origin.",
                    edge.id, edge.source, n
                ));
            }
            if edge.target >= n {
                return Err(format!(
                    "Edge {} target id {} exceeds locus count {}. Return to Origin.",
                    edge.id, edge.target, n
                ));
            }
        }
        for locus in &self.loci {
            if locus.chi <= 0.0 {
                return Err(format!(
                    "Locus {} '{}' has chi = {} ≤ 0; no observable extent declared. Return to Origin.",
                    locus.id, locus.label, locus.chi
                ));
            }
        }
        // Check full-operator admissibility: exists at least one locus t s.t.
        // t = target of some edge AND t = source of some other independently declared edge.
        let has_interior = self.loci.iter().any(|locus| {
            let has_incoming = self.edges.iter().any(|e| e.target == locus.id);
            let has_outgoing = self.edges.iter().any(|e| e.source == locus.id);
            has_incoming && has_outgoing
        });
        if !has_interior {
            return Err(
                "No interior locus found: A→B→R full-operator admissibility not satisfied. \
                 Topology requires at least one locus with both incoming and outgoing edges. \
                 Return to Origin.".to_string()
            );
        }
        Ok(())
    }
}

// ── Declaration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── [REPR] Topology invariants ────────────────────────────────────────────

    /// [REPR] Interior locus (landmark, id=1) must be target of e₀ AND source of e₁.
    #[test]
    fn repr_interior_locus_landmark() {
        let decl = MDeclaration::ar_measurement_update();
        let e0 = &decl.edges[0];
        let e1 = &decl.edges[1];
        // landmark id = 1
        assert_eq!(e0.target, 1, "e0 must target landmark (id=1)");
        assert_eq!(e1.source, 1, "e1 must source from landmark (id=1)");
        // They must be independently declared (different edge ids)
        assert_ne!(e0.id, e1.id, "e0 and e1 must be independently declared");
    }

    /// [REPR] Both edges carry EdgeProvenance::Coupling.
    #[test]
    fn repr_edge_provenance_coupling() {
        let decl = MDeclaration::ar_measurement_update();
        for edge in &decl.edges {
            assert_eq!(
                edge.provenance, EdgeProvenance::Coupling,
                "Edge {} must carry Coupling provenance per Origin declaration", edge.id
            );
        }
    }

    /// [REPR] camera_feature (id=0) has no incoming edges.
    #[test]
    fn repr_source_locus_no_incoming() {
        let decl = MDeclaration::ar_measurement_update();
        let preds = decl.predecessors(0);
        assert!(preds.is_empty(), "camera_feature must have no incoming edges");
    }

    /// [REPR] display_update (id=2) has no outgoing edges.
    #[test]
    fn repr_terminal_locus_no_outgoing() {
        let decl = MDeclaration::ar_measurement_update();
        let succs = decl.successors(2);
        assert!(succs.is_empty(), "display_update must have no outgoing edges");
    }

    /// [REPR] Successor/predecessor symmetry: succ(e₀) targets landmark, pred(e₁) sources landmark.
    #[test]
    fn repr_successor_predecessor_symmetry() {
        let decl = MDeclaration::ar_measurement_update();
        let succs_of_camera = decl.successors(0);  // edges departing from camera_feature
        let preds_of_display = decl.predecessors(2); // edges arriving at display_update
        assert_eq!(succs_of_camera.len(), 1);
        assert_eq!(preds_of_display.len(), 1);
        assert_eq!(succs_of_camera[0].id, 0, "Only e0 departs from camera_feature");
        assert_eq!(preds_of_display[0].id, 1, "Only e1 arrives at display_update");
    }

    // ── [REPR] ρ formula ──────────────────────────────────────────────────────

    /// [REPR] ρ[camera_feature] = 0.025 · 8.0 / (2.0 + 8.0) = 0.020
    #[test]
    fn repr_rho_camera_feature() {
        let decl = MDeclaration::ar_measurement_update();
        let expected = 0.025_f64 * 8.0 / (2.0 + 8.0);
        assert!(
            approx_eq(decl.rho(0), expected, TOLERANCE),
            "rho(camera_feature) = {}, expected {}", decl.rho(0), expected
        );
    }

    /// [REPR] ρ[landmark] = 0.025 · 4.0 / (2.0 + 4.0) = 1/60 ≈ 0.016667
    #[test]
    fn repr_rho_landmark() {
        let decl = MDeclaration::ar_measurement_update();
        let expected = 0.025_f64 * 4.0 / (2.0 + 4.0);
        assert!(
            approx_eq(decl.rho(1), expected, TOLERANCE),
            "rho(landmark) = {}, expected {}", decl.rho(1), expected
        );
    }

    /// [REPR] ρ[display_update] = 0.025 · 2.0 / (2.0 + 2.0) = 0.0125
    #[test]
    fn repr_rho_display_update() {
        let decl = MDeclaration::ar_measurement_update();
        let expected = 0.025_f64 * 2.0 / (2.0 + 2.0);
        assert!(
            approx_eq(decl.rho(2), expected, TOLERANCE),
            "rho(display_update) = {}, expected {}", decl.rho(2), expected
        );
    }

    /// [REPR] ρ saturates toward rho_base as χ → ∞.
    #[test]
    fn repr_rho_saturates_toward_rho_base() {
        let decl = MDeclaration::ar_measurement_update();
        // For very large chi, ρ ≈ rho_base
        let large_chi = 1_000_000.0_f64;
        let rho_large = decl.rho_base * large_chi / (decl.chi_0 + large_chi);
        assert!(
            (rho_large - decl.rho_base).abs() < 1e-4,
            "ρ must saturate toward rho_base for large χ"
        );
    }

    // ── [REPR] Declaration validation ─────────────────────────────────────────

    /// [REPR] Valid declaration must pass validate() without error.
    #[test]
    fn repr_valid_declaration_passes() {
        let decl = MDeclaration::ar_measurement_update();
        assert!(
            decl.validate().is_ok(),
            "Origin-declared AR topology must pass validation: {:?}",
            decl.validate().err()
        );
    }

    /// [REPR] chi_0 = 0 must fail validation (singularity in ρ formula).
    #[test]
    fn repr_chi0_zero_fails_validation() {
        let mut decl = MDeclaration::ar_measurement_update();
        decl.chi_0 = 0.0;
        let result = decl.validate();
        assert!(
            result.is_err(),
            "chi_0 = 0 must fail validation; got Ok — return to Origin"
        );
        let msg = result.unwrap_err();
        assert!(msg.contains("chi_0"), "Error must name chi_0: {}", msg);
    }

    /// [REPR] rho_base = 0 must fail validation (R operator has no effect).
    #[test]
    fn repr_rho_base_zero_fails_validation() {
        let mut decl = MDeclaration::ar_measurement_update();
        decl.rho_base = 0.0;
        let result = decl.validate();
        assert!(result.is_err(), "rho_base = 0 must fail validation — return to Origin");
    }

    /// [REPR] Edge with out-of-range source id must fail validation.
    #[test]
    fn repr_invalid_edge_source_fails_validation() {
        let mut decl = MDeclaration::ar_measurement_update();
        decl.edges[0].source = 99; // non-existent locus
        let result = decl.validate();
        assert!(result.is_err(), "Invalid edge source must fail validation — return to Origin");
    }

    /// [REPR] Edge with out-of-range target id must fail validation.
    #[test]
    fn repr_invalid_edge_target_fails_validation() {
        let mut decl = MDeclaration::ar_measurement_update();
        decl.edges[1].target = 99;
        let result = decl.validate();
        assert!(result.is_err(), "Invalid edge target must fail validation — return to Origin");
    }

    /// [REPR] Topology without interior locus must fail full-operator admissibility.
    #[test]
    fn repr_no_interior_locus_fails_admissibility() {
        let mut decl = MDeclaration::ar_measurement_update();
        // Remove e1 so landmark has no outgoing edge → no interior locus
        decl.edges.retain(|e| e.id == 0);
        let result = decl.validate();
        assert!(
            result.is_err(),
            "Topology without interior locus must fail A→B→R admissibility — return to Origin"
        );
    }
}
