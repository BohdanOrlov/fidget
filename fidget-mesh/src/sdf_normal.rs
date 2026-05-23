//! P38.22 — SDF-derived per-vertex normal helpers (approach C, slice 2).
//!
//! Provides closure-based analytic central-difference gradient and normal
//! evaluators that downstream callers (the daxa_p31 CPU companion-mesh path
//! and, via FFI shadow, the GPU active-cells compute shader) can use to
//! recompute per-vertex normals from the authoritative SDF at the **actual
//! vertex position** — rather than from cell-corner samples averaged across
//! the whole cell. This is the cheapest possible improvement to the mesh
//! shading inputs without changing geometry.
//!
//! Background and rationale: `docs/architecture/p38_22_sdf_derived_normals.md`.
//!
//! ## What slice 2 ships here
//!
//! - [`central_difference_gradient`]: 6-sample central-difference gradient
//!   `(f(p+h ê_x) − f(p−h ê_x), …)/(2h)` at an arbitrary point, given any
//!   SDF closure. Same numerical method as the existing
//!   `central_difference_cell_gradient` in `src/companion_mesh_gpu.rs:1643`,
//!   but **evaluated at the actual vertex position**, not interpolated
//!   from precomputed cell-corner samples.
//! - [`central_difference_normal`]: thin wrapper that normalizes the
//!   gradient, with a fallback unit vector when the gradient is degenerate.
//! - [`forward_difference_gradient`]: cheaper 3-sample fallback for when
//!   the SDF is expensive (1 fewer evaluation per axis ⇒ 50% cost).
//!
//! Slice 3 will wire these into the mesh extraction post-pass (CPU side)
//! and, via the same numerical contract, the GPU compute shader.
//!
//! ## Why approach C, not A/B
//!
//! Approach A (per-fragment Fidget gradient evaluation, ~6× SDF eval per
//! shaded pixel) was ruled out as too costly in the design doc. Approach B
//! (screen-space pre-pass texture) depends on P38.16 G-buffer slice 3
//! landing first. Approach C runs once per mesh vertex per extraction, has
//! no per-frame cost, and has zero dependency on G-buffer infrastructure.
//! Trade-off: per-vertex (not per-pixel) — won't recover sub-triangle
//! features. Acceptable as the first cut.
//!
//! ## Coordination with P38.37
//!
//! P38.37 (CPU/GPU mesh parity) is investigating whether the CPU baseline
//! mesh path uses an authoritative normal source. If it turns out the CPU
//! baseline uses `companion_mesh.rs::safe_normal` (a sphere-only
//! position-direction hack), then aligning GPU normals to "correct" SDF
//! normals via this helper will WIDEN the CPU/GPU gap measured in P38.37
//! while CLOSING the GPU-vs-Fidget gap. The verdict from the P38.37 slice 1
//! note: `safe_normal` is **not** on the P35.6 baseline path
//! (`build_claybook_dual_contouring_mesh_from_p32_sbs` derives normals via
//! `place_dual_cell_vertex_from_hermite_qef`), so this approach C upgrade
//! is safe to apply to the GPU path independently.

use nalgebra::Vector3;

/// Central-difference gradient `(∂f/∂x, ∂f/∂y, ∂f/∂z)` of `eval` at `pos`,
/// using step size `h`.
///
/// Six SDF evaluations: `f(p ± h ê_x), f(p ± h ê_y), f(p ± h ê_z)`. Result
/// is in the SDF's natural unit-per-length scale (i.e., for a normalized
/// SDF the magnitude is approximately 1 at the zero set).
///
/// Caller picks `h`. A reasonable default for cell-bounded vertices is
/// `h = cell_diagonal / 32` — small enough to be local, large enough to
/// resolve through f32 precision. Hard-feature regions may want smaller
/// `h` (down to `1e-3` of the cell) at the cost of f32 cancellation.
#[inline]
pub fn central_difference_gradient<F>(
    pos: Vector3<f32>,
    h: f32,
    eval: F,
) -> Vector3<f32>
where
    F: Fn(Vector3<f32>) -> f32,
{
    debug_assert!(h.is_finite() && h > 0.0, "central_difference step h must be > 0");
    let inv_2h = 1.0 / (2.0 * h);
    Vector3::new(
        (eval(Vector3::new(pos.x + h, pos.y, pos.z))
            - eval(Vector3::new(pos.x - h, pos.y, pos.z)))
            * inv_2h,
        (eval(Vector3::new(pos.x, pos.y + h, pos.z))
            - eval(Vector3::new(pos.x, pos.y - h, pos.z)))
            * inv_2h,
        (eval(Vector3::new(pos.x, pos.y, pos.z + h))
            - eval(Vector3::new(pos.x, pos.y, pos.z - h)))
            * inv_2h,
    )
}

/// Forward-difference gradient — half the cost of central-difference
/// (3 SDF evaluations + 1 anchor = 4 vs 6), at the price of `O(h)` rather
/// than `O(h²)` truncation error. Use this when the SDF evaluator is
/// expensive and the caller can tolerate slightly noisier normals.
#[inline]
pub fn forward_difference_gradient<F>(
    pos: Vector3<f32>,
    h: f32,
    eval: F,
) -> Vector3<f32>
where
    F: Fn(Vector3<f32>) -> f32,
{
    debug_assert!(h.is_finite() && h > 0.0, "forward_difference step h must be > 0");
    let inv_h = 1.0 / h;
    let f0 = eval(pos);
    Vector3::new(
        (eval(Vector3::new(pos.x + h, pos.y, pos.z)) - f0) * inv_h,
        (eval(Vector3::new(pos.x, pos.y + h, pos.z)) - f0) * inv_h,
        (eval(Vector3::new(pos.x, pos.y, pos.z + h)) - f0) * inv_h,
    )
}

/// Normalized SDF-derived normal at `pos`.
///
/// Computes [`central_difference_gradient`] and normalizes. When the
/// gradient magnitude is below `fallback_threshold`, returns `fallback`
/// instead — typical fallback is `Vector3::y()` (matches the
/// `safe_normal` convention in `src/companion_mesh.rs`).
///
/// This is the recommended entry point for slice 3 wiring: callers pass
/// their authoritative SDF closure and the recovered vertex position, and
/// receive a unit normal sourced from the SDF (not from QEF intersection
/// averaging).
#[inline]
pub fn central_difference_normal<F>(
    pos: Vector3<f32>,
    h: f32,
    fallback: Vector3<f32>,
    fallback_threshold: f32,
    eval: F,
) -> Vector3<f32>
where
    F: Fn(Vector3<f32>) -> f32,
{
    let g = central_difference_gradient(pos, h, eval);
    let n = g.norm();
    if n.is_finite() && n >= fallback_threshold {
        g / n
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance test helper.
    fn close_to(actual: Vector3<f32>, expected: Vector3<f32>, tol: f32) {
        let d = (actual - expected).norm();
        assert!(
            d <= tol,
            "vector {actual:?} too far from expected {expected:?} (d={d:.4}, tol={tol:.4})"
        );
    }

    /// Sphere of radius 0.5 at origin: SDF = ‖p‖ − 0.5; gradient = p/‖p‖.
    #[test]
    fn sphere_gradient_matches_analytic() {
        let sdf = |p: Vector3<f32>| p.norm() - 0.5;
        // Pick a point well off-axis to exercise all three derivative slots.
        let p = Vector3::new(0.3, 0.4, 0.0);
        let expected = p.normalize(); // analytic gradient
        let g = central_difference_gradient(p, 1.0e-3, sdf);
        close_to(g, expected, 1.0e-3);
    }

    /// Plane x = 0: SDF = p.x; gradient = (1, 0, 0) everywhere.
    #[test]
    fn plane_gradient_constant() {
        let sdf = |p: Vector3<f32>| p.x;
        for p in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.5, -1.0, 0.7),
            Vector3::new(-0.3, 0.2, 0.0),
        ] {
            let g = central_difference_gradient(p, 1.0e-2, sdf);
            close_to(g, Vector3::new(1.0, 0.0, 0.0), 1.0e-3);
        }
    }

    /// Forward difference matches central difference to first order for a
    /// smooth field. (We use a sphere; with h=1e-3 both should agree to
    /// within ~h-scale error.)
    #[test]
    fn forward_difference_agrees_first_order() {
        let sdf = |p: Vector3<f32>| p.norm() - 1.0;
        let p = Vector3::new(0.6, 0.0, 0.0);
        let g_cd = central_difference_gradient(p, 1.0e-3, sdf);
        let g_fd = forward_difference_gradient(p, 1.0e-3, sdf);
        close_to(g_cd, g_fd, 5.0e-3);
    }

    /// `central_difference_normal` returns the unit gradient when the field
    /// is well-conditioned.
    #[test]
    fn central_difference_normal_returns_unit_gradient() {
        let sdf = |p: Vector3<f32>| p.norm() - 0.5;
        let p = Vector3::new(0.0, 0.0, 0.6);
        let n = central_difference_normal(p, 1.0e-3, Vector3::y(), 1.0e-6, sdf);
        close_to(n, Vector3::new(0.0, 0.0, 1.0), 1.0e-3);
        // Unit length within rounding.
        assert!((n.norm() - 1.0).abs() < 1.0e-3);
    }

    /// `central_difference_normal` returns the fallback when the gradient is
    /// vanishingly small (constant field).
    #[test]
    fn central_difference_normal_falls_back_when_degenerate() {
        let sdf = |_p: Vector3<f32>| 0.42_f32; // constant; gradient is zero
        let n = central_difference_normal(
            Vector3::zeros(),
            1.0e-3,
            Vector3::y(),
            1.0e-6,
            sdf,
        );
        assert_eq!(n, Vector3::y());
    }

    /// Hard-feature box face: the gradient on either side of a CSG max
    /// boundary is direction-correct on the "winning" half-space. Probing
    /// a point firmly inside the +x face's region of dominance should
    /// recover +x.
    ///
    /// SDF = max(|x|−0.4, |y|−0.4, |z|−0.4) — axis-aligned box of half-extent
    /// 0.4. At p = (0.5, 0.0, 0.0) the dominant active half-space is
    /// (|x|−0.4), so the gradient should be (1, 0, 0).
    #[test]
    fn box_face_gradient_picks_active_half_space() {
        let sdf = |p: Vector3<f32>| {
            (p.x.abs() - 0.4)
                .max((p.y.abs() - 0.4).max(p.z.abs() - 0.4))
        };
        let p = Vector3::new(0.5, 0.0, 0.0);
        // Use a small step so we don't straddle into a different active
        // half-space across the +x face boundary.
        let g = central_difference_gradient(p, 1.0e-3, sdf);
        close_to(g.normalize(), Vector3::new(1.0, 0.0, 0.0), 1.0e-3);
    }
}
