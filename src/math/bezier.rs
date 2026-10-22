//! Cubic Bézier segment evaluation, Horner power basis form, and adaptive bisection length.

use super::vec2::Vec2;

/// Segment degree classification matching Roblox Path2D engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    /// Linear segment (degree 1, 2 control points).
    Linear = 2,
    /// Quadratic Bézier (degree 2, 3 control points, 1 active handle).
    Quadratic = 3,
    /// Cubic Bézier (degree 3, 4 control points, 2 active handles).
    Cubic = 4,
}

/// A Bézier curve segment parameterized by power basis polynomial coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubicBezierSegment {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
    pub p3: Vec2,
    pub kind: SegmentKind,
    pub c1: Vec2,
    pub c2: Vec2,
    pub c3: Vec2,
    pub cached_length: Option<f32>,
}

impl CubicBezierSegment {
    /// Constructs a linear segment between p0 and p1.
    pub fn new_linear(p0: Vec2, p1: Vec2) -> Self {
        let c1 = Vec2::new(p1.x - p0.x, p1.y - p0.y);
        Self {
            p0,
            p1,
            p2: p1,
            p3: p1,
            kind: SegmentKind::Linear,
            c1,
            c2: Vec2::ZERO,
            c3: Vec2::ZERO,
            cached_length: None,
        }
    }

    /// Constructs a quadratic Bézier segment between p0, p1, and p2.
    pub fn new_quadratic(p0: Vec2, p1: Vec2, p2: Vec2) -> Self {
        let c1 = Vec2::new((p1.x - p0.x) * 2.0, (p1.y - p0.y) * 2.0);
        let c2 = Vec2::new((p0.x - (p1.x * 2.0)) + p2.x, (p0.y - (p1.y * 2.0)) + p2.y);
        Self {
            p0,
            p1,
            p2,
            p3: p2,
            kind: SegmentKind::Quadratic,
            c1,
            c2,
            c3: Vec2::ZERO,
            cached_length: None,
        }
    }

    /// Constructs a cubic Bézier segment between p0, p1, p2, and p3.
    pub fn new_cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Self {
        let c1 = Vec2::new((p1.x - p0.x) * 3.0, (p1.y - p0.y) * 3.0);
        let c2 = Vec2::new(
            ((p0.x - (p1.x * 2.0)) + p2.x) * 3.0,
            ((p0.y - (p1.y * 2.0)) + p2.y) * 3.0,
        );
        let c3 = Vec2::new(
            ((p3.x - p0.x) + (p1.x * 3.0)) - (p2.x * 3.0),
            ((p3.y - p0.y) + (p1.y * 3.0)) - (p2.y * 3.0),
        );
        Self {
            p0,
            p1,
            p2,
            p3,
            kind: SegmentKind::Cubic,
            c1,
            c2,
            c3,
            cached_length: None,
        }
    }

    /// Constructs a segment from endpoints and optional tangent handles.
    pub fn from_handles(p0: Vec2, right_handle: Vec2, left_handle: Vec2, p3: Vec2) -> Self {
        let has_right = right_handle.x != 0.0 || right_handle.y != 0.0;
        let has_left = left_handle.x != 0.0 || left_handle.y != 0.0;

        match (has_right, has_left) {
            (false, false) => Self::new_linear(p0, p3),
            (true, false) => Self::new_quadratic(
                p0,
                Vec2::new(p0.x + right_handle.x, p0.y + right_handle.y),
                p3,
            ),
            (false, true) => Self::new_quadratic(
                p0,
                Vec2::new(p3.x + left_handle.x, p3.y + left_handle.y),
                p3,
            ),
            (true, true) => Self::new_cubic(
                p0,
                Vec2::new(p0.x + right_handle.x, p0.y + right_handle.y),
                Vec2::new(p3.x + left_handle.x, p3.y + left_handle.y),
                p3,
            ),
        }
    }

    /// Relative evaluation B(u) - P0 to eliminate catastrophic numerical cancellation.
    #[inline]
    pub fn eval0(&self, u: f32) -> Vec2 {
        match self.kind {
            SegmentKind::Linear => Vec2::new(self.c1.x * u, self.c1.y * u),
            SegmentKind::Quadratic => Vec2::new(
                (self.c1.x + (self.c2.x * u)) * u,
                (self.c1.y + (self.c2.y * u)) * u,
            ),
            SegmentKind::Cubic => {
                let ix = self.c2.x + (self.c3.x * u);
                let iy = self.c2.y + (self.c3.y * u);
                Vec2::new((self.c1.x + (ix * u)) * u, (self.c1.y + (iy * u)) * u)
            }
        }
    }

    /// Absolute position on segment B(u) = P0 + eval0(u).
    #[inline]
    pub fn eval(&self, u: f32) -> Vec2 {
        let rel = self.eval0(u);
        Vec2::new(self.p0.x + rel.x, self.p0.y + rel.y)
    }

    /// First derivative tangent vector B'(u) in Horner form.
    #[inline]
    pub fn tangent(&self, u: f32) -> Vec2 {
        match self.kind {
            SegmentKind::Linear => self.c1,
            SegmentKind::Quadratic => {
                let k = u * 2.0;
                Vec2::new(self.c1.x + (self.c2.x * k), self.c1.y + (self.c2.y * k))
            }
            SegmentKind::Cubic => {
                let k = 3.0 * u;
                Vec2::new(
                    self.c1.x + (u * ((self.c2.x * 2.0) + (self.c3.x * k))),
                    self.c1.y + (u * ((self.c2.y * 2.0) + (self.c3.y * k))),
                )
            }
        }
    }

    /// Computes adaptive bisection segment length with TOL = 1e-5 and max depth 30.
    pub fn compute_length(&self) -> f32 {
        if self.kind == SegmentKind::Linear {
            return self.p0.distance(self.p1);
        }

        const TOL: f32 = 1e-5;
        const MAX_DEPTH: u32 = 30;

        fn rec(
            seg: &CubicBezierSegment,
            t0: f32,
            t1: f32,
            p0: Vec2,
            p1: Vec2,
            depth: u32,
        ) -> f32 {
            let tm = (t0 + t1) * 0.5;
            let pm = seg.eval0(tm);
            let chord_a = p0.distance(p1);
            let half_chords_b = p0.distance(pm) + pm.distance(p1);

            if depth >= MAX_DEPTH || (depth >= 2 && (half_chords_b - chord_a) < TOL) {
                return half_chords_b;
            }
            rec(seg, t0, tm, p0, pm, depth + 1) + rec(seg, tm, t1, pm, p1, depth + 1)
        }

        let p0 = self.eval0(0.0);
        let p1 = self.eval0(1.0);
        rec(self, 0.0, 1.0, p0, p1, 0)
    }

    /// Retrieves cached length or calculates and stores it.
    pub fn length(&mut self) -> f32 {
        if let Some(len) = self.cached_length {
            len
        } else {
            let len = self.compute_length();
            self.cached_length = Some(len);
            len
        }
    }
}
