//! Geometric key derivation from Roblox Path2D control points and frame dimensions.

use super::bezier::CubicBezierSegment;
use super::solver::Path2DSolver;
use super::vec2::Vec2;

/// Roblox UDim2 coordinate: (ScaleX, OffsetX, ScaleY, OffsetY).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct UDim2 {
    pub scale_x: f32,
    pub offset_x: f32,
    pub scale_y: f32,
    pub offset_y: f32,
}

impl UDim2 {
    pub const fn new(sx: f32, ox: f32, sy: f32, oy: f32) -> Self {
        Self {
            scale_x: sx,
            offset_x: ox,
            scale_y: sy,
            offset_y: oy,
        }
    }

    /// Resolves UDim2 coordinates into absolute screen pixel coordinates.
    #[inline]
    pub fn resolve(&self, width: f32, height: f32) -> Vec2 {
        Vec2::new(
            (self.scale_x * width) + self.offset_x,
            (self.scale_y * height) + self.offset_y,
        )
    }
}

/// A Path2D control point with position and left/right tangent handles in UDim2.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Path2DControlPoint {
    pub position: UDim2,
    pub left_tangent: UDim2,
    pub right_tangent: UDim2,
}

impl Path2DControlPoint {
    pub const fn new(pos: UDim2, left: UDim2, right: UDim2) -> Self {
        Self {
            position: pos,
            left_tangent: left,
            right_tangent: right,
        }
    }
}

/// Frame pixel dimensions configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameConfig {
    pub width: f32,
    pub height: f32,
}

impl FrameConfig {
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

/// Generated geometric canary key material.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometricKeyMaterial {
    pub bytes: Vec<u8>,
}

impl GeometricKeyMaterial {
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Folds the 148+ byte buffer into a fixed-size cryptographic seed.
    pub fn fold(&self, output_len: usize) -> Vec<u8> {
        let mut out = vec![0u8; output_len];
        for (i, &b) in self.bytes.iter().enumerate() {
            out[i % output_len] ^= b.wrapping_add((i as u8).wrapping_mul(31));
        }
        out
    }
}

/// Derives geometric canary key material (148+ bytes) by resolving control points
/// against frame dimensions and executing canonical Path2D geometric queries.
pub fn derive_geometric_keys(
    control_points: &[Path2DControlPoint],
    frame: FrameConfig,
) -> GeometricKeyMaterial {
    if control_points.len() < 2 {
        return GeometricKeyMaterial {
            bytes: vec![0u8; 148],
        };
    }

    // 1. Resolve points and handles in pixel space
    let mut resolved_pts = Vec::with_capacity(control_points.len());
    for cp in control_points {
        let p = cp.position.resolve(frame.width, frame.height);
        let l = cp.left_tangent.resolve(frame.width, frame.height);
        let r = cp.right_tangent.resolve(frame.width, frame.height);
        resolved_pts.push((p, l, r));
    }

    // 2. Build segments
    let mut segments = Vec::with_capacity(resolved_pts.len() - 1);
    for i in 0..resolved_pts.len() - 1 {
        let (p0, _, r0) = resolved_pts[i];
        let (p3, l1, _) = resolved_pts[i + 1];
        segments.push(CubicBezierSegment::from_handles(p0, r0, l1, p3));
    }
    let solver = Path2DSolver::from_segments(segments);

    let mut buf = Vec::with_capacity(160);

    // 3. Serialize total curve length (4 bytes IEEE-754 BE)
    let len = solver.get_length();
    buf.extend_from_slice(&len.to_be_bytes());

    // 4. Sample canonical fractions (matching Luarmor canary schedule)
    const FRACTIONS: [f32; 9] = [
        0.0,
        0.125,
        0.25,
        0.375,
        0.44444445,
        0.5,
        0.6666667,
        0.75,
        1.0,
    ];

    for &t in &FRACTIONS {
        let pos = solver.get_position_on_curve(t);
        // Relative to frame size in Scale as Roblox does
        let sx = pos.x / frame.width;
        let sy = pos.y / frame.height;
        buf.extend_from_slice(&sx.to_be_bytes());
        buf.extend_from_slice(&sy.to_le_bytes());

        let tan = solver.get_tangent_on_curve(t);
        buf.extend_from_slice(&tan.x.to_le_bytes());
        buf.extend_from_slice(&tan.y.to_be_bytes());
    }

    for &t in &FRACTIONS {
        let pos = solver.get_position_on_curve_arc_length(t);
        let sx = pos.x / frame.width;
        let sy = pos.y / frame.height;
        buf.extend_from_slice(&sx.to_be_bytes());
        buf.extend_from_slice(&sy.to_le_bytes());

        let tan = solver.get_tangent_on_curve_arc_length(t);
        buf.extend_from_slice(&tan.x.to_le_bytes());
        buf.extend_from_slice(&tan.y.to_be_bytes());
    }

    // Guarantee minimum 148 bytes
    if buf.len() < 148 {
        buf.resize(148, 0x5a);
    }

    GeometricKeyMaterial { bytes: buf }
}
