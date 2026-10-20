# M Regional Instantiation — Full Origin Declaration

**Origin:** Robin Macomber, Founder & President, Metatron Dynamics, Inc.  
**Generator:** Claude (Kernel V8, session 2026-10-19)  
**Kernel Reference:** kernel_reference.md, Kernel V8, 2026-09-27  
**Status:** Origin-confirmed. Parameters approved 2026-10-19.

---

## (a) Region Declaration

**Region label:** Single measurement update step, head-mounted augmented reality  
**Scope:** One cycle of visual feature measurement → landmark association → display correction  
**Physical context:** Wearable AR headset, human scale, indoor/outdoor environment  
**Time boundary:** One measurement update (single frame, 1/30 s at 1080p30)

The region does not extend into:
- IMU pre-integration (prior to measurement)
- Global map management (posterior to measurement)
- Display rendering pipeline (posterior to display_update locus)

---

## (b) Observable Extent

**Device:** Microsoft HoloLens 2  
**Source:** Microsoft HoloLens 2 hardware specification (public documentation)

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Display radiants | >2,500 per radian | HoloLens 2 spec |
| Angular resolution | 47 pixels/degree | HoloLens 2 spec |
| Frame rate | 1080p at 30 Hz | HoloLens 2 spec |
| IMU axes | 3-axis accel, gyro, magnetometer | HoloLens 2 spec |

**Declared observable extents per locus:**

| Locus | χ (pixels) | Basis |
|-------|-----------|-------|
| camera_feature | 8.0 | Feature detector window (~8 px radius typical FAST/ORB) |
| landmark | 4.0 | Landmark patch match extent (~4 px) |
| display_update | 2.0 | Minimum detectable displacement at 47 px/degree |

---

## (c) Regional Orientation

**Direction of declared relation:**  
`camera_feature → landmark → display_update`

**Reading:** measurement flows from observed image feature, through matched 3-D landmark, to computed display correction.

**Interior locus:** `landmark` (id = 1)  
- landmark = target(e₀) where e₀: camera_feature → landmark  
- landmark = source(e₁) where e₁: landmark → display_update  
- e₀ and e₁ are independently declared  
- **Full-operator admissibility condition satisfied** (Kernel V8 §11)

---

## (d) Regional Admission Rule

A locus is admitted to this region if and only if:
1. It carries a declared observable extent χ in pixels, grounded in HoloLens 2 specifications
2. It connects to at least one other locus via an edge with declared EdgeProvenance
3. It is located within the HoloLens 2 visual field (nominal 52° × 29° FOV)
4. It represents a physically observable quantity at the declared time boundary

---

## Declared Parameters

### ρ_base = 0.025

**Derivation:**  
HoloLens 2 displays >2,500 radiants (light points per radian).  
Scale factor: 2,500 / 100,000 = 0.025  

**Meaning:** Baseline saturating influence per declared locus. Sets the scale of relational influence relative to the display density.

### χ₀ = 2.0 pixels

**Derivation:**  
HoloLens 2 angular resolution: 47 pixels/degree.  
Minimum detectable displacement: sub-degree threshold ≈ 1–2 px.  
Declared value: 2.0 px (conservative, within HoloLens 2 resolution).

**Meaning:** Observable extent threshold below which saturation weight approaches zero. Prevents over-weighting of sub-resolution measurements.

### Saturating weight formula

```
ρ[i] = ρ_base · χ[i] / (χ₀ + χ[i])
```

Computed values for this declaration:

| Locus | χ | ρ |
|-------|---|---|
| camera_feature | 8.0 px | 0.025 × 8.0 / (2.0 + 8.0) = **0.020** |
| landmark | 4.0 px | 0.025 × 4.0 / (2.0 + 4.0) = **0.01667** |
| display_update | 2.0 px | 0.025 × 2.0 / (2.0 + 2.0) = **0.01250** |

---

## Treatment Boundary

| Path | Rule |
|------|------|
| **EKF comparator** | Retains conventional matrix mathematics throughout. No Kernel V8 operators applied. |
| **ABR treatment** | Operators A, B, R applied only to declared topology. No statistical estimation, no matrix inversion, no backpropagation. |

This boundary is enforced in code by module separation: `ekf_step.rs` and `abr_step.rs` do not share operator implementations.

---

## Declared Topology

```
camera_feature (0)  ──[Coupling, e₀]──▶  landmark (1)  ──[Coupling, e₁]──▶  display_update (2)
     χ = 8.0 px                            χ = 4.0 px                            χ = 2.0 px
     ρ = 0.020                             ρ = 0.01667                           ρ = 0.01250
```

**EdgeProvenance:** Both edges carry `Coupling` — causal link between distinct, independently declared loci.

---

## Origin Confirmation

Parameters ρ_base = 0.025 and χ₀ = 2.0 pixels confirmed by Origin (Robin Macomber)  
on session 2026-10-19 with the statement: "excellent proceed."

Generator declaration responsibility: Generator proposed these values with HoloLens 2 provenance. Origin reviewed and confirmed. Generator proceeds under Origin authority per Kernel V8 §3 Roles.
