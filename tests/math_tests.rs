//! Test suite for Path2D cubic Bézier curve calculation, arc length inversion, and geometric key derivation.

use luraph_deobf_rs::math::bezier::{CubicBezierSegment, SegmentKind};
use luraph_deobf_rs::math::keys::{
    derive_geometric_keys, FrameConfig, Path2DControlPoint, UDim2,
};
use luraph_deobf_rs::math::solver::Path2DSolver;
use luraph_deobf_rs::math::vec2::Vec2;

#[test]
fn test_t1_vec2_arithmetic_and_precision() {
    let a = Vec2::new(3.0, 4.0);
    let b = Vec2::new(1.0, 2.0);

    assert_eq!(a.add(b), Vec2::new(4.0, 6.0));
    assert_eq!(a.sub(b), Vec2::new(2.0, 2.0));
    assert_eq!(a.mul_scalar(2.0), Vec2::new(6.0, 8.0));
    assert_eq!(a.div_scalar(2.0), Vec2::new(1.5, 2.0));
    assert_eq!(a.dot(b), 3.0 * 1.0 + 4.0 * 2.0); // 11.0
    assert_eq!(a.length_sq(), 25.0);
    assert_eq!(a.length(), 5.0);
    assert_eq!(a.distance(Vec2::new(0.0, 0.0)), 5.0);

    let norm = a.normalize();
    assert!(norm.approx_eq(Vec2::new(0.6, 0.8), 1e-6));

    // Zero vector normalization guard
    let zero = Vec2::ZERO.normalize();
    assert_eq!(zero, Vec2::ZERO);
}

#[test]
fn test_t2_linear_segment_degenerate_and_arc_inversion() {
    let p0 = Vec2::new(10.0, 20.0);
    let p1 = Vec2::new(110.0, 20.0);
    let mut seg = CubicBezierSegment::new_linear(p0, p1);

    assert_eq!(seg.kind, SegmentKind::Linear);
    assert_eq!(seg.c1, Vec2::new(100.0, 0.0));
    assert_eq!(seg.c2, Vec2::ZERO);
    assert_eq!(seg.c3, Vec2::ZERO);

    let len = seg.length();
    assert!((len - 100.0).abs() < 1e-5);

    let mid = seg.eval(0.5);
    assert!(mid.approx_eq(Vec2::new(60.0, 20.0), 1e-5));

    let tan = seg.tangent(0.5);
    assert_eq!(tan, Vec2::new(100.0, 0.0));

    let solver = Path2DSolver::from_segments(vec![seg]);
    let (idx, u) = solver.locate_arc(0.5);
    assert_eq!(idx, 0);
    assert!((u - 0.5).abs() < 1e-5);
}

#[test]
fn test_t3_quadratic_segment_with_single_handle() {
    let p0 = Vec2::new(0.0, 0.0);
    let r0 = Vec2::new(50.0, 100.0);
    let l1 = Vec2::ZERO;
    let p3 = Vec2::new(100.0, 0.0);

    let seg = CubicBezierSegment::from_handles(p0, r0, l1, p3);
    assert_eq!(seg.kind, SegmentKind::Quadratic);
    assert_eq!(seg.c1, Vec2::new(100.0, 200.0));
    assert_eq!(seg.c2, Vec2::new(0.0, -200.0));

    // Midpoint evaluation
    let mid = seg.eval(0.5);
    assert!(mid.approx_eq(Vec2::new(50.0, 50.0), 1e-5));

    // Tangent at peak: horizontal (y-derivative is 0)
    let tan = seg.tangent(0.5);
    assert_eq!(tan.x, 100.0);
    assert!((tan.y).abs() < 1e-5);
}

#[test]
fn test_t4_cubic_s_curve_and_adaptive_bisection_convergence() {
    let p0 = Vec2::new(0.0, 0.0);
    let r0 = Vec2::new(0.0, 100.0);
    let l1 = Vec2::new(0.0, -100.0);
    let p3 = Vec2::new(200.0, 0.0);

    let mut seg = CubicBezierSegment::from_handles(p0, r0, l1, p3);
    assert_eq!(seg.kind, SegmentKind::Cubic);

    let len = seg.length();
    assert!(len > 200.0, "S-curve length must exceed chord distance");

    // Inflection point symmetry at u = 0.5
    let mid = seg.eval(0.5);
    assert!((mid.x - 100.0).abs() < 1e-4);
    assert!((mid.y - 0.0).abs() < 1e-4);
}

#[test]
fn test_t5_rk4_arc_length_inversion_accuracy_and_monotonicity() {
    let p0 = Vec2::new(0.0, 0.0);
    let r0 = Vec2::new(0.0, 100.0);
    let l1 = Vec2::new(0.0, -100.0);
    let p3 = Vec2::new(200.0, 0.0);

    let seg = CubicBezierSegment::from_handles(p0, r0, l1, p3);
    let solver = Path2DSolver::from_segments(vec![seg]);

    let pos_mid = solver.get_position_on_curve_arc_length(0.5);
    assert!((pos_mid.x - 100.0).abs() < 1e-2);
    assert!((pos_mid.y - 0.0).abs() < 1e-2);

    // Monotonicity check
    let mut prev_u = -1.0f32;
    for step in 0..=10 {
        let fraction = step as f32 / 10.0;
        let (_, u) = solver.locate_arc(fraction);
        assert!(u >= prev_u, "Arc parameter u must be monotonic");
        prev_u = u;
    }
}

#[test]
fn test_t6_degenerate_zero_length_and_collinear_segments() {
    let pt = Vec2::new(42.0, 42.0);
    let mut seg = CubicBezierSegment::new_linear(pt, pt);
    let len = seg.length();
    assert_eq!(len, 0.0);

    let solver = Path2DSolver::from_segments(vec![seg]);
    let pos = solver.get_position_on_curve_arc_length(0.5);
    assert_eq!(pos, pt);
    assert!(!pos.x.is_nan() && !pos.y.is_nan());
}

#[test]
fn test_t7_multi_segment_chaining_across_multiple_control_points() {
    let pts = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 0.0),
        Vec2::new(10.0, 20.0),
        Vec2::new(30.0, 20.0),
    ];
    let solver = Path2DSolver::new(&pts);
    assert_eq!(solver.segments.len(), 3);
    assert_eq!(solver.get_length(), 10.0 + 20.0 + 20.0); // 50.0

    // Arc length continuity: at s = 10.0 (fraction 0.2)
    let p_seg1_end = solver.get_position_on_curve_arc_length(0.2);
    assert!(p_seg1_end.approx_eq(Vec2::new(10.0, 0.0), 1e-4));

    // Arc length continuity: at s = 30.0 (fraction 0.6)
    let p_seg2_end = solver.get_position_on_curve_arc_length(0.6);
    assert!(p_seg2_end.approx_eq(Vec2::new(10.0, 20.0), 1e-4));
}

#[test]
fn test_t8_udim2_resolution_and_geometric_key_derivation() {
    let frame = FrameConfig::new(259.0, 198.0);
    let p0 = Path2DControlPoint::new(
        UDim2::new(0.125, 8.0, 0.5, 7.0),
        UDim2::default(),
        UDim2::new(0.125, -7.0, 0.0, -4.0),
    );
    let p1 = Path2DControlPoint::new(
        UDim2::new(0.5, 0.0, 0.8, 10.0),
        UDim2::new(0.0, 7.0, 0.0, 2.0),
        UDim2::default(),
    );

    let keys1 = derive_geometric_keys(&[p0, p1], frame);
    let keys2 = derive_geometric_keys(&[p0, p1], frame);

    assert!(keys1.bytes.len() >= 148);
    // Deterministic key generation
    assert_eq!(keys1.bytes, keys2.bytes);

    // Key folding produces desired digest length
    let folded_16 = keys1.fold(16);
    let folded_32 = keys1.fold(32);
    assert_eq!(folded_16.len(), 16);
    assert_eq!(folded_32.len(), 32);
}
