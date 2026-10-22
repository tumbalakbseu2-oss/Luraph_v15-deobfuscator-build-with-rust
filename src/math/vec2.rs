//! IEEE-754 binary32 2D vector primitives for Roblox Path2D curve evaluation.

use std::ops::{Add, Div, Mul, Sub};

/// 2D single-precision float vector matching Roblox C++ engine binary32 semantics.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    /// Zero vector constant.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Constructs a new 2D vector.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Vector addition in f32.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }

    /// Vector subtraction in f32.
    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }

    /// Scalar multiplication in f32.
    #[inline]
    pub fn mul_scalar(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }

    /// Scalar division in f32.
    #[inline]
    pub fn div_scalar(self, scalar: f32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }

    /// Dot product in f32.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y)
    }

    /// Squared Euclidean length in f32.
    #[inline]
    pub fn length_sq(self) -> f32 {
        (self.x * self.x) + (self.y * self.y)
    }

    /// Euclidean length in f32.
    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Euclidean distance between two points in f32.
    #[inline]
    pub fn distance(self, rhs: Self) -> f32 {
        self.sub(rhs).length()
    }

    /// Normalizes vector with an epsilon guard (> 1e-12) to avoid NaN.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-12 {
            self.div_scalar(len)
        } else {
            Self::ZERO
        }
    }

    /// Checks approximate equality within epsilon tolerance.
    #[inline]
    pub fn approx_eq(self, rhs: Self, eps: f32) -> bool {
        (self.x - rhs.x).abs() <= eps && (self.y - rhs.y).abs() <= eps
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Vec2::add(self, rhs)
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Vec2::sub(self, rhs)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Vec2::mul_scalar(self, rhs)
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        Vec2::div_scalar(self, rhs)
    }
}
