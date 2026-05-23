//! P38.15 — Dual contouring from sampled signed-distance data.
//!
//! Sibling to [`crate::qef::QuadraticErrorSolver`] (Schaefer-style QEF that
//! requires per-edge intersection positions AND surface gradients). This module
//! recovers a cell vertex from **sampled SDF values alone** — no analytic
//! gradient access — by fitting a local tri-variate quadratic to the sample
//! neighborhood and extracting a feature point from its zero-level set.
//!
//! Reference: *Dual Contouring of Signed Distance Data* (arXiv 2604.00157,
//! 2026). Design doc: `docs/architecture/p38_15_dual_contouring_sdf_data.md`.
//!
//! ## Status
//!
//! Slice 2 of 5: this file contains the public API skeleton, internal
//! placeholders, and `#[ignore]`'d unit-test fixtures. The actual fitting and
//! feature-point extraction land in slice 3.

use nalgebra::{Matrix3, Vector3, Vector4};

use crate::cell::CellVertex;

/// A discrete SDF sample at a known grid position.
///
/// `pos` is in the same world-space frame the parent dual-contouring pass uses
/// (i.e. the same units `CellVertex` bounds are expressed in). `value` is the
/// signed distance — negative inside, positive outside.
#[derive(Copy, Clone, Debug)]
pub struct SampledPoint {
    pub pos: Vector3<f32>,
    pub value: f32,
}

/// Solver for placing a cell vertex from a local neighborhood of SDF samples.
///
/// The intended caller pattern (slice 4 wires this into [`crate::dc`]):
///
/// ```ignore
/// let mut solver = SampledQuadraticSolver::new();
/// for sample in neighborhood_27(cell) {
///     solver.add_sample(sample);
/// }
/// let (vertex, residual) = solver.solve_in_cell(cell_bounds);
/// ```
///
/// The internal accumulators store the normal-equations form
/// `A^T A · c = A^T b` for the 10 quadratic coefficients
/// `c = (a,b,c,d,e,g,h,i,j,k)` of
/// `f(x,y,z) ≈ a x² + b y² + c z² + d xy + e xz + g yz + h x + i y + j z + k`.
///
/// Slice 3 will replace the placeholder fields below with the actual 10×10
/// accumulator. They are stubbed as `()` for now to keep the public type
/// stable while the storage representation is finalized.
#[derive(Clone, Debug, Default)]
pub struct SampledQuadraticSolver {
    /// Sample count, used for under-determined detection.
    sample_count: usize,

    /// Mass point of samples — used as the seed for the iterative
    /// feature-point projection in `solve_in_cell`. Matches the convention in
    /// [`crate::qef::QuadraticErrorSolver`] (XYZ accumulated, W = count).
    mass_point: Vector4<f32>,

    /// Placeholder for the 10×10 normal-equations accumulator. Slice 3
    /// replaces this with `nalgebra::SMatrix<f32, 10, 10>`.
    _ata_10x10: (),

    /// Placeholder for the 10-element RHS. Slice 3 replaces with
    /// `nalgebra::SVector<f32, 10>`.
    _atb_10: (),
}

impl SampledQuadraticSolver {
    /// Construct an empty solver. No samples added.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate one SDF sample into the solver.
    ///
    /// Slice 2: increments the sample count and accumulates the mass point.
    /// The 10-coefficient normal-equations system is wired in slice 3.
    pub fn add_sample(&mut self, sample: SampledPoint) {
        self.sample_count += 1;
        self.mass_point += Vector4::new(sample.pos.x, sample.pos.y, sample.pos.z, 1.0);
        let _ = sample.value; // consumed by the LSQ system in slice 3
    }

    /// Number of samples accumulated so far.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Solve for a feature vertex inside the given cell bounds.
    ///
    /// Returns the recovered [`CellVertex`] and a residual (lower is better;
    /// the residual is the sum of squared SDF-prediction errors at the
    /// supplied samples — same units as [`crate::qef::QuadraticErrorSolver`]'s
    /// reported error).
    ///
    /// Slice 2: returns the mass-point centroid clamped to the cell, and a
    /// residual of `f32::NAN` to make accidental "looks plausible" misuses
    /// obvious. The real fit + Newton projection ships in slice 3.
    pub fn solve_in_cell(&self, _bounds: CellBounds) -> (CellVertex<3>, f32) {
        // Slice 2 placeholder: hand back the centroid so the type round-trips,
        // but make the residual NaN so downstream "is this better?" comparisons
        // refuse to silently treat the stub as a valid result.
        debug_assert!(
            self.sample_count > 0,
            "SampledQuadraticSolver::solve_in_cell called with zero samples"
        );

        let centroid = if self.mass_point.w > 0.0 {
            self.mass_point.xyz() / self.mass_point.w
        } else {
            Vector3::zeros()
        };

        let vertex = CellVertex::from_position_unclamped_stub(centroid);
        (vertex, f32::NAN)
    }

    /// (Slice 3) Will return the 10 fitted coefficients
    /// `(a, b, c, d, e, g, h, i, j, k)` of the local quadratic.
    pub fn coefficients(&self) -> Option<[f32; 10]> {
        // Slice 2 placeholder. Slice 3 implements the actual normal-equations
        // solve via either nalgebra::linalg::QR or the SVD path already used
        // by `QuadraticErrorSolver::solve`.
        let _ = Matrix3::<f32>::zeros(); // keep nalgebra in scope
        None
    }
}

/// Axis-aligned cell bounds in the same world-space frame as the samples.
///
/// Slice 2 keeps this private to the module API to avoid coupling to the
/// existing `CellVertex<3>::cell_bounds` shape until slice 3 picks the right
/// representation. Slice 3 will either alias this to the existing bounds type
/// in [`crate::cell`] or replace it with a thin wrapper.
#[derive(Copy, Clone, Debug)]
pub struct CellBounds {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl CellBounds {
    pub fn unit() -> Self {
        Self {
            min: Vector3::new(0.0, 0.0, 0.0),
            max: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn center(&self) -> Vector3<f32> {
        (self.min + self.max) * 0.5
    }

    pub fn contains(&self, p: Vector3<f32>) -> bool {
        p.x >= self.min.x
            && p.y >= self.min.y
            && p.z >= self.min.z
            && p.x <= self.max.x
            && p.y <= self.max.y
            && p.z <= self.max.z
    }
}

// ===== Slice 2 thin shim ==================================================
//
// `CellVertex<3>` already exists in `crate::cell` but its constructor surface
// expects clamping and bounding information that slice 2 does not yet model.
// Rather than couple slice 2 to that surface (which is still in flux per
// `qef.rs::solve`'s rank-adaptive clamping), expose a tiny stub constructor
// here. Slice 3 will delete this shim and route through the canonical
// constructor.
//
// This shim lives in this module only — not on the public type — so that
// removing it in slice 3 cannot leak to other call sites.

trait CellVertexStubCtor {
    fn from_position_unclamped_stub(pos: Vector3<f32>) -> Self;
}

impl CellVertexStubCtor for CellVertex<3> {
    fn from_position_unclamped_stub(_pos: Vector3<f32>) -> Self {
        // Slice 2: do not invent a vertex. The unit tests below are
        // `#[ignore]`'d, so this branch is never reached in CI today.
        // Slice 3 replaces this with the real `CellVertex<3>` builder.
        unimplemented!(
            "CellVertex<3>::from_position_unclamped_stub is a slice-2 placeholder; \
             slice 3 will route through the canonical constructor in crate::cell"
        )
    }
}

// ===== Unit-test fixtures (slice 2: all #[ignore]'d) =======================
//
// Five fixtures from the design doc. Each is `#[ignore]`'d because the solver
// body is not implemented yet — slice 3 removes the `#[ignore]` attributes as
// each fixture starts to pass.

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample an analytic SDF on a 3×3×3 grid centered on `center` with the
    /// given `step` between samples. Returns 27 points.
    fn neighborhood_27(
        center: Vector3<f32>,
        step: f32,
        sdf: impl Fn(Vector3<f32>) -> f32,
    ) -> Vec<SampledPoint> {
        let mut out = Vec::with_capacity(27);
        for iz in -1i32..=1 {
            for iy in -1i32..=1 {
                for ix in -1i32..=1 {
                    let pos = center
                        + Vector3::new(ix as f32 * step, iy as f32 * step, iz as f32 * step);
                    let value = sdf(pos);
                    out.push(SampledPoint { pos, value });
                }
            }
        }
        out
    }

    /// Distance check helper. Slice 3 will tighten the tolerance after the
    /// solver is real.
    fn close_to(actual: Vector3<f32>, expected: Vector3<f32>, tol: f32) {
        let d = (actual - expected).norm();
        assert!(
            d <= tol,
            "vertex {actual:?} too far from expected {expected:?} (d={d:.4}, tol={tol:.4})"
        );
    }

    /// Fixture 1 — three orthogonal half-spaces meeting at a corner.
    /// Expected feature point: the corner itself.
    #[test]
    #[ignore = "slice 3: solver body not implemented yet"]
    fn cube_corner_recovers_corner() {
        let corner = Vector3::new(0.5, 0.5, 0.5);
        let sdf = |p: Vector3<f32>| {
            // Inside the corner when all three coords > 0.5
            (corner.x - p.x).max((corner.y - p.y).max(corner.z - p.z))
        };
        let samples = neighborhood_27(corner, 0.25, sdf);

        let mut solver = SampledQuadraticSolver::new();
        for s in samples {
            solver.add_sample(s);
        }
        let bounds = CellBounds {
            min: Vector3::new(0.0, 0.0, 0.0),
            max: Vector3::new(1.0, 1.0, 1.0),
        };
        let (vertex, _) = solver.solve_in_cell(bounds);
        close_to(vertex_pos_stub(&vertex), corner, 0.05);
    }

    /// Fixture 2 — two-plane wedge. Expected: feature point on the wedge line
    /// midway across the cell.
    #[test]
    #[ignore = "slice 3: solver body not implemented yet"]
    fn two_plane_wedge_recovers_ridge_midpoint() {
        // Wedge along the y axis: max(x - 0.5, z - 0.5)
        let sdf = |p: Vector3<f32>| (p.x - 0.5).max(p.z - 0.5);
        let samples = neighborhood_27(Vector3::new(0.5, 0.5, 0.5), 0.25, sdf);

        let mut solver = SampledQuadraticSolver::new();
        for s in samples {
            solver.add_sample(s);
        }
        let bounds = CellBounds::unit();
        let (vertex, _) = solver.solve_in_cell(bounds);
        // Expected: on the ridge x=0.5, z=0.5, y anywhere — pick the midpoint.
        close_to(vertex_pos_stub(&vertex), Vector3::new(0.5, 0.5, 0.5), 0.05);
    }

    /// Fixture 3 — sphere intersected with a plane. Expected: vertex on the
    /// intersection circle, in the cell's plane of symmetry.
    #[test]
    #[ignore = "slice 3: solver body not implemented yet"]
    fn sphere_plane_intersection_on_circle() {
        let center = Vector3::new(0.5, 0.5, 0.5);
        // Sphere of radius 0.4 intersected with the half-space y >= 0.5.
        let sdf = |p: Vector3<f32>| {
            let sphere = (p - center).norm() - 0.4;
            let plane = 0.5 - p.y;
            sphere.max(plane)
        };
        let samples = neighborhood_27(center, 0.25, sdf);

        let mut solver = SampledQuadraticSolver::new();
        for s in samples {
            solver.add_sample(s);
        }
        let bounds = CellBounds::unit();
        let (vertex, _) = solver.solve_in_cell(bounds);
        // Expected: somewhere on the circle x²+z² = 0.16, y = 0.5 (cell plane
        // of symmetry). Loose tolerance for the slice-3 first cut.
        let v = vertex_pos_stub(&vertex);
        let radial = ((v.x - 0.5).powi(2) + (v.z - 0.5).powi(2)).sqrt();
        assert!(
            (radial - 0.4).abs() < 0.08 && (v.y - 0.5).abs() < 0.08,
            "vertex {v:?} not near the sphere-plane intersection circle"
        );
    }

    /// Fixture 4 — smooth sphere (no hard feature). Sanity baseline that the
    /// sampled-DC solver does not introduce noise where the field is smooth.
    #[test]
    #[ignore = "slice 3: solver body not implemented yet"]
    fn smooth_sphere_vertex_near_surface() {
        let center = Vector3::new(0.0, 0.0, 0.0);
        let sdf = |p: Vector3<f32>| p.norm() - 0.5;
        // Cell that straddles the surface along +x.
        let cell_center = Vector3::new(0.5, 0.0, 0.0);
        let samples = neighborhood_27(cell_center, 0.25, sdf);

        let mut solver = SampledQuadraticSolver::new();
        for s in samples {
            solver.add_sample(s);
        }
        let bounds = CellBounds {
            min: Vector3::new(0.0, -0.5, -0.5),
            max: Vector3::new(1.0, 0.5, 0.5),
        };
        let (vertex, _) = solver.solve_in_cell(bounds);
        let v = vertex_pos_stub(&vertex);
        // Expected: within 0.1·cell-diagonal of the closest sphere-surface
        // point, which is at (0.5, 0, 0).
        close_to(v, Vector3::new(0.5, 0.0, 0.0), 0.1);
    }

    /// Fixture 5 — boolean subtraction creating a sharp ridge. Cross-check
    /// against the gradient-QEF baseline (run that comparison in slice 4).
    #[test]
    #[ignore = "slice 3: solver body not implemented yet"]
    fn boolean_subtraction_ridge() {
        // A box minus a cylinder hole through it — creates a sharp ridge on
        // the box face where the cylinder exits.
        let sdf = |p: Vector3<f32>| {
            let box_sdf = (p.x.abs() - 0.4)
                .max((p.y.abs() - 0.4).max(p.z.abs() - 0.4));
            let cyl_radius = (p.x.powi(2) + p.y.powi(2)).sqrt() - 0.2;
            box_sdf.max(-cyl_radius)
        };
        // Cell on the +z face where the cylinder ridge cuts through.
        let cell_center = Vector3::new(0.2, 0.0, 0.4);
        let samples = neighborhood_27(cell_center, 0.1, sdf);

        let mut solver = SampledQuadraticSolver::new();
        for s in samples {
            solver.add_sample(s);
        }
        let bounds = CellBounds {
            min: Vector3::new(0.1, -0.1, 0.3),
            max: Vector3::new(0.3, 0.1, 0.5),
        };
        let (vertex, _) = solver.solve_in_cell(bounds);
        // Expected: on the ridge — the cylinder boundary projected onto the
        // box face. Slice 4 will define the exact tolerance after cross-
        // checking against gradient-QEF on the same cell.
        let v = vertex_pos_stub(&vertex);
        assert!(
            bounds.contains(v),
            "vertex {v:?} escaped the cell bounds {bounds:?}"
        );
    }

    /// Slice 2 helper: extract the position from a `CellVertex<3>`. Slice 3
    /// replaces this with the canonical accessor once `CellVertex<3>` is the
    /// real type instead of the placeholder.
    fn vertex_pos_stub(_v: &CellVertex<3>) -> Vector3<f32> {
        // Slice 2: the tests above are all `#[ignore]`'d, so this path is
        // unreached. Slice 3 will return `v.position()` (or equivalent).
        unimplemented!(
            "vertex_pos_stub is a slice-2 placeholder; slice 3 routes through \
             the canonical CellVertex<3> position accessor"
        )
    }

    /// Cheap sanity that the public API at least compiles and accumulates
    /// sample counts. Not `#[ignore]`'d.
    #[test]
    fn solver_accumulates_sample_count() {
        let mut solver = SampledQuadraticSolver::new();
        assert_eq!(solver.sample_count(), 0);
        solver.add_sample(SampledPoint {
            pos: Vector3::zeros(),
            value: 0.0,
        });
        solver.add_sample(SampledPoint {
            pos: Vector3::new(1.0, 0.0, 0.0),
            value: 1.0,
        });
        assert_eq!(solver.sample_count(), 2);
    }

    /// Slice 2 sanity for `CellBounds::contains`. Not `#[ignore]`'d.
    #[test]
    fn cell_bounds_contains_basic() {
        let b = CellBounds::unit();
        assert!(b.contains(Vector3::new(0.5, 0.5, 0.5)));
        assert!(!b.contains(Vector3::new(-0.1, 0.5, 0.5)));
        assert!(!b.contains(Vector3::new(0.5, 1.1, 0.5)));
        assert_eq!(b.center(), Vector3::new(0.5, 0.5, 0.5));
    }
}
