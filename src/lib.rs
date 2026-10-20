//! abr-slam-comparison
//!
//! Side-by-side comparison of EKF measurement update (conventional) vs.
//! ABR relational operators (Kernel V8, Metatron Dynamics, Inc.).
//!
//! Origin:    Robin Macomber, Metatron Dynamics, Inc.
//! Generator: Claude (session 2026-10-19)
//! Reference: kernel_reference.md, Kernel V8, 2026-09-27
//!
//! Domain: AR visual-inertial measurement update, HoloLens 2

pub mod declaration;
pub mod ekf_step;
pub mod abr_step;
pub mod comparison;
pub mod simulation;
pub mod integration_tests;
