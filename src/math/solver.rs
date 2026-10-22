//! Multi-segment Path2D curve chaining and 32-step Runge-Kutta 4 (RK4) arc-length solver.

use super::bezier::{CubicBezierSegment, SegmentKind};
use super::vec2::Vec2;

/// Multi-segment Path2D Bézier curve solver.
#[derive(Clone, Debug, Default)]
pub struct Path2DSolver {
    pub segments: Vec<CubicBezierSegment>,
    pub segment_lengths: Vec<f32>,
    pub total_length: f32,
}

impl Path2DSolver {
    /// Constructs a new solver by chaining contiguous points with linear segments.
    pub fn new(control_points: &[Vec2]) -> Self {
        if control_points.len() < 2 {
            return Self::default();
        }

        let mut segments = Vec::with_capacity(control_points.len() - 1);
        for i in 0..control_points.len() - 1 {
            segments.push(CubicBezierSegment::new_linear(
                control_points[i],
                control_points[i + 1],
            ));
        }
        Self::from_segments(segments)
    }

    /// Constructs a solver from pre-built Bézier segments and computes total arc length.
    pub fn from_segments(mut segments: Vec<CubicBezierSegment>) -> Self {
        let mut lengths = Vec::with_capacity(segments.len());
        let mut total = 0.0f32;
        for seg in &mut segments {
            let l = seg.length();
            lengths.push(l);
            total += l;
        }
        Self {
            segments,
            segment_lengths: lengths,
            total_length: total,
        }
    }

    /// Returns the total cumulative arc length of all segments in f32.
    pub fn get_length(&self) -> f32 {
        self.total_length
    }

    /// Maps normalized curve parameter t in [0.0, 1.0] to (segment_index, local_u).
    pub fn locate(&self, t: f32) -> (usize, f32) {
        if self.segments.is_empty() {
            return (0, 0.0);
        }
        let t = t.clamp(0.0, 1.0);
        let n = self.segments.len() as f32;
        let x = t * n;
        let k = (x.floor() as usize).min(self.segments.len() - 1);
        let u = x - (k as f32);
        (k, u)
    }

    /// Solves for local curve parameter u given distance s via 32-step Runge-Kutta 4:
    /// du/ds = 1 / ||B'(u)||
    pub fn arc_u(seg: &CubicBezierSegment, s: f32) -> f32 {
        let d2x = seg.c2.x * 2.0;
        let d2y = seg.c2.y * 2.0;
        let d3x = if seg.kind == SegmentKind::Cubic {
            seg.c3.x * 3.0
        } else {
            0.0
        };
        let d3y = if seg.kind == SegmentKind::Cubic {
            seg.c3.y * 3.0
        } else {
            0.0
        };

        let inv_speed = |u: f32| -> f32 {
            let (dx, dy) = if seg.kind == SegmentKind::Quadratic {
                (seg.c1.x + (d2x * u), seg.c1.y + (d2y * u))
            } else {
                (
                    seg.c1.x + (u * (d2x + (u * d3x))),
                    seg.c1.y + (u * (d2y + (u * d3y))),
                )
            };
            let speed_sq = (dx * dx) + (dy * dy);
            let speed = speed_sq.sqrt();
            if speed < 1e-12 {
                1e-12
            } else {
                speed
            }
        };

        const SIXTH: f32 = 1.0 / 6.0;
        let h = s / 32.0;
        let mut u = 0.0f32;

        for _ in 0..32 {
            let k1 = h / inv_speed(u);
            let k2 = h / inv_speed(u + (k1 * 0.5));
            let k3 = h / inv_speed(u + (k2 * 0.5));
            let k4 = h / inv_speed(u + k3);
            u += ((k1 + (2.0 * (k2 + k3))) + k4) * SIXTH;
        }
        u
    }

    /// Maps normalized arc-length fraction t in [0.0, 1.0] to (segment_index, local_u).
    pub fn locate_arc(&self, t: f32) -> (usize, f32) {
        if self.segments.is_empty() {
            return (0, 0.0);
        }
        let t = t.clamp(0.0, 1.0);
        let mut s = t * self.total_length;

        for (i, seg) in self.segments.iter().enumerate() {
            let len = self.segment_lengths[i];
            if s <= len {
                let u = if seg.kind == SegmentKind::Linear {
                    if len > 1e-12 {
                        s / len
                    } else {
                        0.0
                    }
                } else {
                    Self::arc_u(seg, s)
                };
                return (i, u);
            }
            s -= len;
        }

        (self.segments.len() - 1, 1.0)
    }

    /// Evaluates absolute 2D position at normalized curve parameter t in [0.0, 1.0].
    pub fn get_position_on_curve(&self, t: f32) -> Vec2 {
        if self.segments.is_empty() {
            return Vec2::ZERO;
        }
        let (k, u) = self.locate(t);
        self.segments[k].eval(u)
    }

    /// Evaluates 2D tangent vector at normalized curve parameter t in [0.0, 1.0].
    pub fn get_tangent_on_curve(&self, t: f32) -> Vec2 {
        if self.segments.is_empty() {
            return Vec2::ZERO;
        }
        let (k, u) = self.locate(t);
        self.segments[k].tangent(u)
    }

    /// Evaluates absolute 2D position at arc length fraction target_len_fraction in [0.0, 1.0].
    pub fn get_position_on_curve_arc_length(&self, target_len_fraction: f32) -> Vec2 {
        if self.segments.is_empty() {
            return Vec2::ZERO;
        }
        let (k, u) = self.locate_arc(target_len_fraction);
        self.segments[k].eval(u)
    }

    /// Evaluates 2D tangent vector at arc length fraction target_len_fraction in [0.0, 1.0].
    pub fn get_tangent_on_curve_arc_length(&self, target_len_fraction: f32) -> Vec2 {
        if self.segments.is_empty() {
            return Vec2::ZERO;
        }
        let (k, u) = self.locate_arc(target_len_fraction);
        self.segments[k].tangent(u)
    }
}
