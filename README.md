# ABR-SLAM-COMPARISON

**Metatron Dynamics, Inc. — Kernel V8**  
Origin: Robin Macomber  
Generator: Claude Sonnet 4.6  
Verifier: ChatGPT

---

## What this is

A head-to-head comparison of the Metatron relational operators (Kernel V8) against an Extended Kalman Filter (EKF) for AR hologram tracking on the HoloLens 2.

**Central claim:** A linear recursive algebraic feedback loop is more accurate *and* less compute than EKF.

**Demonstrated result (V1.2, frame 599):**

| | Metatron (V1.2) | EKF |
|---|---|---|
| Residual | **0.025 px** | 0.796 px |
| Total ops (600 frames) | **7,200** | 300,000 |
| Ops/frame | **12** | 500 |

→ **31.8× more accurate AND 41.7× less compute.**

---

## Frozen operators (Kernel V8 — immutable)

```
A(x)[e]  = x[src(e)] − x[tgt(e)]
B(g)[e]  = g[e] + Σ_{f ∈ succ(e)} g[f]
R(g)[e]  = g[e] + ρ[src(e)] · (Σ_succ − Σ_pred)
```

Saturating weight: `ρ[i] = ρ_base · χ[i] / (χ₀ + χ[i])`  
Parameters: `ρ_base = 0.025`, `χ₀ = 2.0`

These operators are not modified by any version.

---

## Versions

| Version | Ops/frame | Residual @ 600f | Observable | Notes |
|---------|-----------|-----------------|------------|-------|
| V0.9 | 11 | 0.025 px | z_delta (external) | Predict-update; sub-pixel |
| V1.0 | 12 | ~50 px | z_obs (absolute) | 4-locus chain; wrong labels |
| V1.1 | 12 | ~50 px | z_obs (absolute) | Declaration correction |
| **V1.2** | **12** | **0.025 px** | **z_delta (internal)** | **relay_a delay locus** |

### V1.2 architecture

`relay_a` is a declared locus that holds `z_obs[t-1]`. Each frame:

```
z_delta  = z_obs − relay_a           // derived from delay edge (no external input)
g        = A→B→R on 3-locus chain    // 7 ops: same as V0.9
landmark += ρ · R(e₀)                // velocity estimate
display_delta -= R(e₁)               // display velocity
hologram += display_delta            // position prediction
hologram += ρ · (z_obs − hologram)  // position correction: 4 ops
relay_a  = z_obs                     // delay locus update: 1 op
```

Total: `chain(7) + predict-update(4) + relay_store(1) = 12 ops/frame`

---

## Running

```bash
cargo test          # 79 tests, all pass
cargo run --release # OUTPUTs 1–10
```

### Outputs

| Output | Content |
|--------|---------|
| 1 | Mathematical conformance (A, B, R operators) |
| 2 | Computational cost |
| 3 | Observable correspondence |
| 4 | Simulation summary |
| 5 | Diagnostic |
| 6 | Convergence |
| 7 | V0.9 — predict-update, sub-pixel accuracy |
| 8 | V1.0 — 4-locus declarative chain |
| 9 | V1.1 — relational position space (declaration correction) |
| **10** | **V1.2 — z_delta derived internally; KEY RESULT** |

---

## Domain

Device: Microsoft HoloLens 2  
Scene: linear pan, 1 px/frame, 600 frames (20 s at 30 fps)  
Observable extent: `χ₀ = 2.0 px` (HoloLens 2 angular resolution threshold)

---

## Repository structure

```
src/
  abr_step.rs          — Metatron operators A, B, R (Kernel V8)
  declaration.rs       — MDeclaration: locus topologies V0.9 / V1.0 / V1.1 / V1.2
  simulation.rs        — simulate_abr_v09/v10/v11/v12, simulate_ekf_continuous
  comparison.rs        — ComparisonReport: OUTPUTs 1–10
  integration_tests.rs — 79 regression tests
  ekf_step.rs          — EKF reference implementation
  main.rs              — entry point
VERIFIER-HANDOFF.md    — claims and reproduction steps for independent verification
```

---

*We work with relational evolution, not temporal. All loci carry absolute position in the same relational space. Locus persistence between frames IS the relational continuity.*
