//! Offline Path2D cubic Bézier curve calculation and geometric key derivation.

pub mod bezier;
pub mod keys;
pub mod solver;
pub mod vec2;

pub use bezier::{CubicBezierSegment, SegmentKind};
pub use keys::{
    derive_geometric_keys, FrameConfig, GeometricKeyMaterial, Path2DControlPoint, UDim2,
};
pub use solver::Path2DSolver;
pub use vec2::Vec2;
