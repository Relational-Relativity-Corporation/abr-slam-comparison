//! abr-slam-comparison: demonstration entry point

use abr_slam_comparison::comparison::ComparisonReport;

fn main() {
    println!();
    println!("ABR-SLAM-COMPARISON");
    println!("EKF Measurement Update vs. ABR Relational Operators");
    println!("Metatron Dynamics, Inc. — Kernel V8 — 2026-10-19");
    println!("Domain: AR head-mounted display, HoloLens 2 single-step update");
    println!();

    let report = ComparisonReport::run();

    report.print_conformance();
    report.print_cost();
    report.print_observable_correspondence();
    report.print_simulation();
    report.print_diagnostic();
    report.print_convergence();
    report.print_v09();
    report.print_v10();
    report.print_v11();
    report.print_v12();
}
