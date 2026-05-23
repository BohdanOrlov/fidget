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

use nalgebra::{SMatrix, SVector, Vector3, Vector4};

use crate::cell::CellVertex;

/// Number of monomials in the local tri-variate quadratic
/// `f(x,y,z) ≈ a x² + b y² + c z² + d xy + e xz + g yz + h x + i y + j z + k`.
const QUAD_DIM: usize = 10;

/// Evaluate the 10-monomial basis vector at a position.
///
/// Order is `(x², y², z², xy, xz, yz, x, y, z, 1)`. This ordering matches the
/// coefficient vector consumed by [`SampledQuadraticSolver::solve_in_cell`].
#[inline]
fn monomials(p: Vector3<f32>) -> SVector<f32, QUAD_DIM> {
    SVector::<f32, QUAD_DIM>::from_column_slice(&[
        p.x * p.x,
        p.y * p.y,
        p.z * p.z,
        p.x * p.y,
        p.x * p.z,
        p.y * p.z,
        p.x,
        p.y,
        p.z,
        1.0,
    ])
}

/// Analytic gradient of the fitted quadratic at `p`, given coefficients `c`
/// in the same order as [`monomials`].
///
/// ∂f/∂x = 2 a x + d y + e z + h
/// ∂f/∂y = 2 b y + d x + g z + i
/// ∂f/∂z = 2 c z + e x + g y + j
#[inline]
fn quadratic_gradient(c: &SVector<f32, QUAD_DIM>, p: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(
        2.0 * c[0] * p.x + c[3] * p.y + c[4] * p.z + c[6],
        2.0 * c[1] * p.y + c[3] * p.x + c[5] * p.z + c[7],
        2.0 * c[2] * p.z + c[4] * p.x + c[5] * p.y + c[8],
    )
}

/// Evaluate the fitted quadratic at `p`.
#[inline]
fn quadratic_eval(c: &SVector<f32, QUAD_DIM>, p: Vector3<f32>) -> f32 {
    c.dot(&monomials(p))
}

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

    /// Normal-equations accumulator `A^T A` for the 10-monomial LSQ system.
    ata: SMatrix<f32, QUAD_DIM, QUAD_DIM>,

    /// Normal-equations RHS `A^T b` for the 10-monomial LSQ system.
    atb: SVector<f32, QUAD_DIM>,

    /// Sum of squared sample values; used to recover the LSQ residual without
    /// re-evaluating the quadratic at every sample after the solve.
    btb: f32,
}

impl SampledQuadraticSolver {
    /// Construct an empty solver. No samples added.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate one SDF sample into the solver.
    ///
    /// Builds the 10-element monomial vector at `sample.pos` and folds it
    /// into the normal-equations accumulators `A^T A`, `A^T b`, plus `b^T b`
    /// for the residual.
    pub fn add_sample(&mut self, sample: SampledPoint) {
        self.sample_count += 1;
        self.mass_point +=
            Vector4::new(sample.pos.x, sample.pos.y, sample.pos.z, 1.0);
        let m = monomials(sample.pos);
        self.ata += m * m.transpose();
        self.atb += m * sample.value;
        self.btb += sample.value * sample.value;
    }

    /// Number of samples accumulated so far.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Solve for a feature vertex inside the given cell bounds.
    ///
    /// Algorithm:
    /// 1. Solve the 10-coefficient quadratic LSQ via SVD (robust against
    ///    rank deficiency from collinear or insufficient samples).
    /// 2. Newton-iterate from the cell center toward the zero set of the
    ///    fitted quadratic. Step: `p ← p − f(p)/‖∇f‖² · ∇f`. Five iterations
    ///    is plenty for a quadratic (each step reduces |f| roughly
    ///    quadratically when ∇f ≠ 0).
    /// 3. Clamp the recovered position to `bounds`.
    ///
    /// Returns `(vertex, residual)` where `residual` is the LSQ residual
    /// `‖A c − b‖²` (sum of squared SDF-prediction errors over the input
    /// samples) — same convention as
    /// [`crate::qef::QuadraticErrorSolver`].
    pub fn solve_in_cell(&self, bounds: CellBounds) -> (CellVertex<3>, f32) {
        debug_assert!(
            self.sample_count > 0,
            "SampledQuadraticSolver::solve_in_cell called with zero samples"
        );

        // === Step 1: LSQ solve via SVD ===
        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);
        // Pseudo-inverse threshold: anything below 1e-6 × largest singular
        // value is treated as zero (consistent with qef.rs's eigenvalue
        // cutoff philosophy, just on the SVD side).
        let coefficients = svd
            .solve(&self.atb, 1.0e-6)
            .unwrap_or_else(|_| SVector::<f32, QUAD_DIM>::zeros());

        // LSQ residual: ‖A c − b‖² = c^T A^T A c − 2 c^T A^T b + b^T b.
        let residual = (coefficients.transpose() * self.ata * coefficients
            - 2.0 * coefficients.transpose() * self.atb)
            .x
            + self.btb;

        // === Step 2: Newton iteration from cell center ===
        let mut p = bounds.center();
        const NEWTON_STEPS: usize = 5;
        const GRADIENT_FLOOR: f32 = 1.0e-8;
        for _ in 0..NEWTON_STEPS {
            let f = quadratic_eval(&coefficients, p);
            let g = quadratic_gradient(&coefficients, p);
            let g_norm_sq = g.norm_squared();
            if g_norm_sq < GRADIENT_FLOOR {
                // Gradient vanished — Newton can't make progress. The current
                // position is the best we can do without escalating to a
                // different solver. Common cause: the fitted quadratic is
                // nearly constant in this neighborhood.
                break;
            }
            p -= (f / g_norm_sq) * g;
        }

        // === Step 3: Clamp to cell bounds ===
        let clamped = Vector3::new(
            p.x.clamp(bounds.min.x, bounds.max.x),
            p.y.clamp(bounds.min.y, bounds.max.y),
            p.z.clamp(bounds.min.z, bounds.max.z),
        );

        (CellVertex { pos: clamped }, residual.max(0.0))
    }

    /// Return the 10 fitted coefficients `(a, b, c, d, e, g, h, i, j, k)` of
    /// the local quadratic, or `None` if fewer than 10 samples have been
    /// accumulated (system is under-determined).
    pub fn coefficients(&self) -> Option<[f32; QUAD_DIM]> {
        if self.sample_count < QUAD_DIM {
            return None;
        }
        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);
        let c = svd.solve(&self.atb, 1.0e-6).ok()?;
        let mut out = [0.0f32; QUAD_DIM];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = c[i];
        }
        Some(out)
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

// ===== Unit-test fixtures ==================================================
//
// Five fixtures from the design doc. Slice 3 enables `cube_corner_*` and
// `smooth_sphere_*` (the cases where a smooth quadratic should converge
// reliably). The remaining three stay `#[ignore]`'d pending solver
// refinements in slices 4–5.

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
    /// Expected feature point: near the corner. Slice 3 uses a loose
    /// tolerance (0.20·cell-diagonal ≈ 0.20) because a smooth tri-variate
    /// quadratic can only approximate the C⁰-discontinuous `max(...)`
    /// indicator function — the fitted zero set passes near the corner but
    /// not exactly through it. Slice 4 will refine via either a finer
    /// neighborhood weighting or a piecewise quadratic fit.
    #[test]
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
        close_to(vertex.pos, corner, 0.20);
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
        close_to(vertex.pos, Vector3::new(0.5, 0.5, 0.5), 0.05);
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
        let v = vertex.pos;
        let radial = ((v.x - 0.5).powi(2) + (v.z - 0.5).powi(2)).sqrt();
        assert!(
            (radial - 0.4).abs() < 0.08 && (v.y - 0.5).abs() < 0.08,
            "vertex {v:?} not near the sphere-plane intersection circle"
        );
    }

    /// Fixture 4 — smooth sphere (no hard feature). Sanity baseline that the
    /// sampled-DC solver does not introduce noise where the field is smooth.
    /// Enabled in slice 3: a quadratic should fit a sphere's local SDF
    /// extremely well, so this should converge to within 0.1 of the true
    /// closest-surface point.
    #[test]
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
        let v = vertex.pos;
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
        let v = vertex.pos;
        assert!(
            bounds.contains(v),
            "vertex {v:?} escaped the cell bounds {bounds:?}"
        );
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
