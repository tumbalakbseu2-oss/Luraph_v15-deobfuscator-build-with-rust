//! Standalone Adversarial Verification Test Suite for luraph_deobf_rs::math
//!
//! Stress-tests:
//! 1. Degenerate Bézier curves (coincident points, zero lengths, cusp handles, closed loops)
//! 2. Extreme coordinates (very small subnormals, large numbers, NaN/Inf robustness)
//! 3. RK4 arc-length inversion on high-curvature S-curves (monotonicity, error bounds)
//! 4. Geometric key derivation (determinism, varying resolutions, 148+ byte invariant, folding)

use luraph_deobf_rs::math::bezier::{CubicBezierSegment, SegmentKind};
use luraph_deobf_rs::math::keys::{
    derive_geometric_keys, FrameConfig, Path2DControlPoint, UDim2,
};
use luraph_deobf_rs::math::solver::Path2DSolver;
use luraph_deobf_rs::math::vec2::Vec2;

// =========================================================================
// 1. Degenerate Bézier Curves
// =========================================================================

#[test]
fn test_adv_degenerate_all_points_identical() {
    let pt = Vec2::new(13.37, 42.42);

    // 1. Linear degenerate
    let mut linear = CubicBezierSegment::new_linear(pt, pt);
    assert_eq!(linear.length(), 0.0);
    assert_eq!(linear.eval(0.0), pt);
    assert_eq!(linear.eval(0.5), pt);
    assert_eq!(linear.eval(1.0), pt);
    assert_eq!(linear.tangent(0.5), Vec2::ZERO);

    // 2. Quadratic degenerate
    let mut quad = CubicBezierSegment::new_quadratic(pt, pt, pt);
    assert_eq!(quad.length(), 0.0);
    assert_eq!(quad.eval(0.0), pt);
    assert_eq!(quad.eval(0.5), pt);
    assert_eq!(quad.eval(1.0), pt);
    assert_eq!(quad.tangent(0.5), Vec2::ZERO);

    // 3. Cubic degenerate
    let mut cubic = CubicBezierSegment::new_cubic(pt, pt, pt, pt);
    assert_eq!(cubic.length(), 0.0);
    assert_eq!(cubic.eval(0.0), pt);
    assert_eq!(cubic.eval(0.5), pt);
    assert_eq!(cubic.eval(1.0), pt);
    assert_eq!(cubic.tangent(0.5), Vec2::ZERO);

    // 4. Solver arc length inversion on cubic degenerate
    let solver = Path2DSolver::from_segments(vec![cubic]);
    assert_eq!(solver.get_length(), 0.0);

    for &fraction in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let pos = solver.get_position_on_curve_arc_length(fraction);
        assert_eq!(pos, pt, "Arc evaluation on zero-length cubic must return pt");
        assert!(!pos.x.is_nan() && !pos.y.is_nan());

        let tan = solver.get_tangent_on_curve_arc_length(fraction);
        assert_eq!(tan, Vec2::ZERO);
        assert!(!tan.x.is_nan() && !tan.y.is_nan());
    }
}

#[test]
fn test_adv_coincident_endpoints_loop() {
    // A closed teardrop loop where p0 == p3, but handles extend into space
    let p0 = Vec2::new(50.0, 50.0);
    let p3 = Vec2::new(50.0, 50.0);
    let r0 = Vec2::new(100.0, 200.0);
    let l1 = Vec2::new(-100.0, 200.0);

    let mut seg = CubicBezierSegment::from_handles(p0, r0, l1, p3);
    assert_eq!(seg.kind, SegmentKind::Cubic);

    let len = seg.length();
    assert!(
        len > 100.0,
        "Closed loop must have positive non-zero arc length (got {})",
        len
    );

    // Peak of the loop at t = 0.5
    let mid = seg.eval(0.5);
    // Control points: p0=(50,50), p1=(150,250), p2=(-50,250), p3=(50,50)
    // B(0.5) = 1/8*(50,50) + 3/8*(150,250) + 3/8*(-50,250) + 1/8*(50,50)
    // x = 6.25 + 56.25 - 18.75 + 6.25 = 50.0
    // y = 6.25 + 93.75 + 93.75 + 6.25 = 200.0
    assert!(
        (mid.x - 50.0).abs() < 1e-4,
        "Midpoint x should be 50.0, got {}",
        mid.x
    );
    assert!(
        (mid.y - 200.0).abs() < 1e-4,
        "Midpoint y should be 200.0, got {}",
        mid.y
    );

    // Check arc length solver monotonicity on closed loop
    let solver = Path2DSolver::from_segments(vec![seg]);
    let mut prev_u = -1.0f32;
    for step in 0..=20 {
        let fraction = step as f32 / 20.0;
        let (_, u) = solver.locate_arc(fraction);
        assert!(
            u >= prev_u - 1e-6,
            "Monotonicity violated on loop at fraction {}: prev_u={}, u={}",
            fraction,
            prev_u,
            u
        );
        prev_u = u;
    }
}

#[test]
fn test_adv_collinear_cusp_and_antiparallel_handles() {
    // A cusp curve that reverses direction along x axis
    let p0 = Vec2::new(0.0, 0.0);
    let p1 = Vec2::new(100.0, 0.0);
    let p2 = Vec2::new(-50.0, 0.0);
    let p3 = Vec2::new(50.0, 0.0);

    let mut seg = CubicBezierSegment::new_cubic(p0, p1, p2, p3);
    let len = seg.length();
    assert!(len > 0.0, "Cusp curve must have positive length");

    // RK4 arc_u must not divide by zero or panic even if speed drops near zero
    let u_half = Path2DSolver::arc_u(&seg, len * 0.5);
    assert!(!u_half.is_nan() && !u_half.is_infinite());
    assert!(u_half >= 0.0 && u_half <= 1.5);
}

#[test]
fn test_adv_mixed_chain_zero_and_nonzero_segments() {
    // Segment 0: zero-length
    let s0 = CubicBezierSegment::new_linear(Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.0));
    // Segment 1: length 100
    let s1 = CubicBezierSegment::new_linear(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0));
    // Segment 2: zero-length
    let s2 = CubicBezierSegment::new_linear(Vec2::new(100.0, 0.0), Vec2::new(100.0, 0.0));
    // Segment 3: length 50
    let s3 = CubicBezierSegment::new_linear(Vec2::new(100.0, 0.0), Vec2::new(100.0, 50.0));

    let solver = Path2DSolver::from_segments(vec![s0, s1, s2, s3]);
    assert_eq!(solver.get_length(), 150.0);

    // Arc position at start (fraction 0.0) -> (0, 0)
    let p_start = solver.get_position_on_curve_arc_length(0.0);
    assert!(p_start.approx_eq(Vec2::new(0.0, 0.0), 1e-4));

    // Arc position at 100/150 (fraction ~0.666667) -> (100, 0)
    let p_mid = solver.get_position_on_curve_arc_length(100.0 / 150.0);
    assert!(p_mid.approx_eq(Vec2::new(100.0, 0.0), 1e-3));

    // Arc position at end (fraction 1.0) -> (100, 50)
    let p_end = solver.get_position_on_curve_arc_length(1.0);
    assert!(p_end.approx_eq(Vec2::new(100.0, 50.0), 1e-3));
}

// =========================================================================
// 2. Extreme Coordinates & Numerical Stability
// =========================================================================

#[test]
fn test_adv_vector_subnormals_and_normalization_guards() {
    // Vector with magnitude below epsilon guard (1e-12)
    let tiny = Vec2::new(1e-13, -1e-13);
    assert_eq!(
        tiny.normalize(),
        Vec2::ZERO,
        "Sub-epsilon vector must normalize to ZERO"
    );

    let subnormal = Vec2::new(1e-39, 1e-40);
    assert_eq!(
        subnormal.normalize(),
        Vec2::ZERO,
        "Subnormal vector must normalize to ZERO"
    );

    // Vector with magnitude just above epsilon guard
    let just_above = Vec2::new(2e-12, 0.0);
    let norm_above = just_above.normalize();
    assert!(
        (norm_above.x - 1.0).abs() < 1e-5,
        "Vector above guard must normalize to unit vector (x={})",
        norm_above.x
    );
    assert_eq!(norm_above.y, 0.0);

    // Exact zero vector
    assert_eq!(Vec2::ZERO.normalize(), Vec2::ZERO);
}

#[test]
fn test_adv_negative_and_offset_coordinates() {
    let p0 = Vec2::new(-1000.0, -2000.0);
    let p1 = Vec2::new(-500.0, -1500.0);
    let mut seg = CubicBezierSegment::new_linear(p0, p1);

    let expected_len = ((500.0f32 * 500.0) + (500.0 * 500.0)).sqrt();
    assert!((seg.length() - expected_len).abs() < 1e-3);

    let mid = seg.eval(0.5);
    assert!(mid.approx_eq(Vec2::new(-750.0, -1750.0), 1e-3));
}

#[test]
fn test_adv_out_of_bounds_clamping_and_nan_immunity() {
    let p0 = Vec2::new(0.0, 0.0);
    let p1 = Vec2::new(100.0, 100.0);
    let seg = CubicBezierSegment::new_linear(p0, p1);
    let solver = Path2DSolver::from_segments(vec![seg]);

    // Negative t clamped to 0.0
    let pos_neg = solver.get_position_on_curve(-50.0);
    assert_eq!(pos_neg, p0);

    let pos_neg_arc = solver.get_position_on_curve_arc_length(-50.0);
    assert_eq!(pos_neg_arc, p0);

    // Over-unity t clamped to 1.0
    let pos_pos = solver.get_position_on_curve(100.0);
    assert_eq!(pos_pos, p1);

    let pos_pos_arc = solver.get_position_on_curve_arc_length(100.0);
    assert_eq!(pos_pos_arc, p1);

    // Extreme float values clamp without panics
    let _ = solver.get_position_on_curve(f32::MAX);
    let _ = solver.get_position_on_curve(f32::MIN);
    let _ = solver.get_position_on_curve_arc_length(f32::INFINITY);
    let _ = solver.get_position_on_curve_arc_length(f32::NEG_INFINITY);

    // NaN parameter must not panic
    let _ = solver.get_position_on_curve(f32::NAN);
    let _ = solver.get_position_on_curve_arc_length(f32::NAN);
}

#[test]
fn test_adv_large_finite_coordinates_stability() {
    // Large coordinates within f32 range where distance squared does not overflow to Infinity:
    // (up to ~1e18, because (1e18)^2 = 1e36 < 3.4e38).
    let p0 = Vec2::new(1e5, 1e5);
    let p1 = Vec2::new(2e5, 1e5);
    let p2 = Vec2::new(2e5, 2e5);
    let p3 = Vec2::new(3e5, 2e5);

    let mut seg = CubicBezierSegment::new_cubic(p0, p1, p2, p3);
    let len = seg.length();
    assert!(len > 2e5 && len < 4e5, "Length must be reasonable for 1e5 scale, got {}", len);

    let solver = Path2DSolver::from_segments(vec![seg]);
    let pos_mid = solver.get_position_on_curve_arc_length(0.5);
    assert!(!pos_mid.x.is_nan() && !pos_mid.y.is_nan());
    assert!(pos_mid.x > 1.5e5 && pos_mid.x < 2.5e5);
}

#[test]
fn test_adv_extreme_1e20_coordinates() {
    let p0 = Vec2::new(1e20, 1e20);
    let p1 = Vec2::new(2e20, 1e20);
    let p2 = Vec2::new(2e20, 2e20);
    let p3 = Vec2::new(3e20, 2e20);

    let mut seg = CubicBezierSegment::new_cubic(p0, p1, p2, p3);
    // Let's verify whether compute_length terminates without hanging
    let len = seg.length();
    // In f32, (1e20)^2 overflows to Inf
    assert!(len.is_infinite() || len.is_nan() || len > 1e20);
}

// =========================================================================
// 3. RK4 Arc-Length Inversion Rigor & Error Bounds
// =========================================================================

#[test]
fn test_adv_rk4_monotonicity_and_error_bounds_high_curvature_s_curve() {
    // Extreme S-curve with steep control handles
    let p0 = Vec2::new(0.0, 0.0);
    let r0 = Vec2::new(20.0, 300.0);
    let l1 = Vec2::new(-20.0, -300.0);
    let p3 = Vec2::new(100.0, 0.0);

    let mut seg = CubicBezierSegment::from_handles(p0, r0, l1, p3);
    let total_len = seg.length();
    println!("Measured steep S-curve total_len = {}", total_len);
    assert!(total_len > 100.0, "Steep S-curve length must be greater than chord length (100.0), got {}", total_len);

    let solver = Path2DSolver::from_segments(vec![seg]);

    // 1. Verify strict monotonicity across 100 evaluation points
    let num_samples = 100;
    let mut prev_u = -1.0f32;
    let mut u_samples = Vec::with_capacity(num_samples + 1);

    for i in 0..=num_samples {
        let fraction = i as f32 / num_samples as f32;
        let (_, u) = solver.locate_arc(fraction);
        assert!(
            u >= prev_u - 1e-6,
            "RK4 inversion violated monotonicity: prev_u={}, u={} at fraction={}",
            prev_u,
            u,
            fraction
        );
        prev_u = u;
        u_samples.push(u);
    }

    // 2. Verify error bounds: compare RK4 inverted u against numerical arc-length integration
    // We compute length from 0 to u using fine Simpson's rule (1000 sub-steps)
    let simpson_arc_length = |u_target: f32| -> f32 {
        if u_target <= 0.0 {
            return 0.0;
        }
        let steps = 500;
        let dt = u_target / steps as f32;
        let speed = |u: f32| -> f32 { seg.tangent(u).length() };

        let mut sum = speed(0.0) + speed(u_target);
        for j in 1..steps {
            let t = j as f32 * dt;
            let weight = if j % 2 == 0 { 2.0 } else { 4.0 };
            sum += speed(t) * weight;
        }
        sum * (dt / 3.0)
    };

    // Evaluate error at test fractions
    for &fraction in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let (_, u) = solver.locate_arc(fraction);
        let actual_s = simpson_arc_length(u);
        let target_s = fraction * total_len;
        let relative_error = (actual_s - target_s).abs() / total_len;

        assert!(
            relative_error < 0.015,
            "RK4 arc-length error exceeded 1.5% at fraction {}: target_s={}, actual_s={}, rel_err={}",
            fraction,
            target_s,
            actual_s,
            relative_error
        );
    }
}

#[test]
fn test_adv_rk4_extreme_hairpin_turn() {
    // Hairpin curve that doubles back almost completely
    let p0 = Vec2::new(0.0, 0.0);
    let p1 = Vec2::new(10.0, 500.0);
    let p2 = Vec2::new(15.0, 500.0);
    let p3 = Vec2::new(25.0, 0.0);

    let seg = CubicBezierSegment::new_cubic(p0, p1, p2, p3);
    let solver = Path2DSolver::from_segments(vec![seg]);

    let (idx, u_mid) = solver.locate_arc(0.5);
    assert_eq!(idx, 0);
    assert!(
        (u_mid - 0.5).abs() < 0.05,
        "Hairpin apex should be near u=0.5 (got {})",
        u_mid
    );
}

// =========================================================================
// 4. Geometric Key Derivation Rigor
// =========================================================================

#[test]
fn test_adv_geometric_keys_determinism_and_length() {
    let frame = FrameConfig::new(800.0, 600.0);
    let cp1 = Path2DControlPoint::new(
        UDim2::new(0.1, 10.0, 0.2, 5.0),
        UDim2::new(0.0, 0.0, 0.0, 0.0),
        UDim2::new(0.1, 20.0, 0.0, -10.0),
    );
    let cp2 = Path2DControlPoint::new(
        UDim2::new(0.5, -5.0, 0.7, 15.0),
        UDim2::new(0.0, -15.0, 0.1, 5.0),
        UDim2::new(0.0, 0.0, 0.0, 0.0),
    );
    let points = [cp1, cp2];

    let base = derive_geometric_keys(&points, frame);
    assert!(
        base.bytes.len() >= 148,
        "Key material must be >= 148 bytes (got {})",
        base.bytes.len()
    );

    // Determinism test: 50 repeated invocations must yield identical byte sequences
    for i in 0..50 {
        let current = derive_geometric_keys(&points, frame);
        assert_eq!(
            base.bytes, current.bytes,
            "Key derivation non-deterministic on iteration {}",
            i
        );
    }
}

#[test]
fn test_adv_geometric_keys_varying_frame_resolutions() {
    let cp1 = Path2DControlPoint::new(
        UDim2::new(0.2, 0.0, 0.3, 0.0),
        UDim2::default(),
        UDim2::new(0.1, 10.0, 0.0, 0.0),
    );
    let cp2 = Path2DControlPoint::new(
        UDim2::new(0.8, 0.0, 0.7, 0.0),
        UDim2::new(-0.1, -10.0, 0.0, 0.0),
        UDim2::default(),
    );
    let points = [cp1, cp2];

    let frames = [
        FrameConfig::new(1920.0, 1080.0), // 16:9 1080p
        FrameConfig::new(2560.0, 1440.0), // 1440p
        FrameConfig::new(3840.0, 2160.0), // 4K
        FrameConfig::new(800.0, 600.0),   // 4:3 SVGA
        FrameConfig::new(1.0, 1.0),       // Unit frame
        FrameConfig::new(10000.0, 500.0), // Extreme aspect ratio
    ];

    let mut key_outputs = Vec::new();

    for &f in &frames {
        let mat = derive_geometric_keys(&points, f);
        assert!(
            mat.bytes.len() >= 148,
            "Key material length for frame {}x{} must be >= 148 (got {})",
            f.width,
            f.height,
            mat.bytes.len()
        );
        key_outputs.push(mat.bytes);
    }

    // Different frame sizes must produce distinct canary key material
    for i in 0..key_outputs.len() {
        for j in (i + 1)..key_outputs.len() {
            assert_ne!(
                key_outputs[i], key_outputs[j],
                "Key material must differentiate distinct frame resolutions ({:?} vs {:?})",
                frames[i], frames[j]
            );
        }
    }
}

#[test]
fn test_adv_geometric_keys_edge_cases_and_fallbacks() {
    let frame = FrameConfig::new(100.0, 100.0);

    // Empty control points
    let empty = derive_geometric_keys(&[], frame);
    assert_eq!(empty.bytes.len(), 148);
    assert_eq!(empty.bytes, vec![0u8; 148]);

    // Single control point
    let single = derive_geometric_keys(
        &[Path2DControlPoint::new(
            UDim2::default(),
            UDim2::default(),
            UDim2::default(),
        )],
        frame,
    );
    assert_eq!(single.bytes.len(), 148);
    assert_eq!(single.bytes, vec![0u8; 148]);

    // All zero control points (collinear at 0, 0)
    let zeros = [
        Path2DControlPoint::default(),
        Path2DControlPoint::default(),
    ];
    let zero_keys = derive_geometric_keys(&zeros, frame);
    assert!(zero_keys.bytes.len() >= 148);
    // Must not contain NaNs or crash
    for chunk in zero_keys.bytes.chunks(4) {
        if chunk.len() == 4 {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            assert!(!val.is_nan(), "Key bytes must not decode to NaN");
        }
    }

    // Key folding tests
    let folded_32 = zero_keys.fold(32);
    assert_eq!(folded_32.len(), 32);
    let folded_128 = zero_keys.fold(128);
    assert_eq!(folded_128.len(), 128);
}

#[test]
fn test_adv_concurrent_multithreaded_key_derivation() {
    use std::sync::Arc;
    use std::thread;

    let frame = FrameConfig::new(1920.0, 1080.0);
    let cp1 = Path2DControlPoint::new(
        UDim2::new(0.1, 10.0, 0.2, 5.0),
        UDim2::default(),
        UDim2::new(0.1, 20.0, 0.0, -10.0),
    );
    let cp2 = Path2DControlPoint::new(
        UDim2::new(0.5, -5.0, 0.7, 15.0),
        UDim2::new(0.0, -15.0, 0.1, 5.0),
        UDim2::default(),
    );
    let points = Arc::new(vec![cp1, cp2]);
    let reference_keys = derive_geometric_keys(&points, frame);

    let mut handles = Vec::new();
    for _ in 0..16 {
        let pts = Arc::clone(&points);
        let expected = reference_keys.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let keys = derive_geometric_keys(&pts, frame);
                assert_eq!(keys, expected);
            }
        }));
    }

    for h in handles {
        h.join().expect("Worker thread panicked");
    }
}
