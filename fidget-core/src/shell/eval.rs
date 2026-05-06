//! Pure shell evaluation kernels.

use std::{
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use super::{
    OpenTopPolicy, ShellParamsView, ShellProfileSegmentTopology,
    ShellProfileTopology, ShellSegmentTopology, ShellTopology,
    topology::{
        SHELL_MAX_CANDIDATES, SHELL_MAX_NODES_PER_CURVE, ShellOpKind,
        ShellProfileNodeContinuity, ShellProfileSpanInterpolation,
        ShellSegmentInterpolation, ShellStationMapping,
    },
};

/// Reusable evaluator scratch.
#[derive(Clone, Debug)]
pub struct ShellEvalScratch {
    closest_segment: Option<usize>,
    candidate_segments: [MaybeUninit<usize>; SHELL_MAX_CANDIDATES],
    candidate_count: usize,
}

impl Default for ShellEvalScratch {
    fn default() -> Self {
        Self {
            closest_segment: None,
            candidate_segments: [const { MaybeUninit::uninit() };
                SHELL_MAX_CANDIDATES],
            candidate_count: 0,
        }
    }
}

impl ShellEvalScratch {
    /// Returns the last closest segment index.
    pub fn closest_segment(&self) -> Option<usize> {
        self.closest_segment
    }

    /// Returns the number of segment candidates used by the last evaluation.
    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Returns fixed segment candidate capacity.
    pub fn candidate_capacity(&self) -> usize {
        self.candidate_segments.len()
    }

    fn candidate_slice(&self) -> &[usize] {
        // Only the first `candidate_count` entries are initialized before this
        // slice is returned.
        unsafe {
            std::slice::from_raw_parts(
                self.candidate_segments.as_ptr().cast::<usize>(),
                self.candidate_count,
            )
        }
    }

    fn collect_single_candidate(&mut self, index: usize) -> &[usize] {
        self.candidate_segments[0].write(index);
        self.candidate_count = 1;
        self.candidate_slice()
    }

    fn collect_candidates(&mut self, topology: &ShellTopology) -> &[usize] {
        assert!(
            topology.segments.len() <= self.candidate_segments.len(),
            "shell topology exceeds fixed candidate capacity"
        );
        self.candidate_count = topology.segments.len();
        for index in 0..topology.segments.len() {
            self.candidate_segments[index].write(index);
        }
        self.candidate_slice()
    }

    fn collect_point_candidates(
        &mut self,
        topology: &ShellTopology,
        params: ShellParamsView<'_>,
        x: f32,
    ) -> &[usize] {
        assert!(
            topology.segments.len() <= self.candidate_segments.len(),
            "shell topology exceeds fixed candidate capacity"
        );
        if topology.segments.len() <= 3 {
            return self.collect_candidates(topology);
        }

        if let Some(index) = monotonic_station_segment(topology, params, x) {
            return self.collect_single_candidate(index);
        }

        let mut best_index = 0usize;
        let mut best_gap = f32::INFINITY;
        for (index, segment) in topology.segments.iter().copied().enumerate() {
            let left = topology.sections[segment.left_section].station(params);
            let right =
                topology.sections[segment.right_section].station(params);
            let min_x = left.min(right);
            let max_x = left.max(right);
            let gap = if x < min_x {
                min_x - x
            } else if x > max_x {
                x - max_x
            } else {
                0.0
            };
            if gap < best_gap {
                best_gap = gap;
                best_index = index;
            }
        }

        self.collect_single_candidate(best_index)
    }
}

fn monotonic_station_segment(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    x: f32,
) -> Option<usize> {
    if !params.is_empty() {
        return None;
    }

    let first = topology.sections.first()?.station;
    let last = topology.sections.last()?.station;
    match topology.station_mapping {
        ShellStationMapping::Increasing => {
            if x <= first {
                return Some(0);
            }
            if x >= last {
                return Some(topology.segments.len() - 1);
            }
            let mut lo = 0usize;
            let mut hi = topology.sections.len() - 1;
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if topology.sections[mid].station <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Some(lo)
        }
        ShellStationMapping::Decreasing => {
            if x >= first {
                return Some(0);
            }
            if x <= last {
                return Some(topology.segments.len() - 1);
            }
            let mut lo = 0usize;
            let mut hi = topology.sections.len() - 1;
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if topology.sections[mid].station >= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Some(lo)
        }
        ShellStationMapping::Unordered => None,
    }
}

/// Result of a shell sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellSample {
    /// Signed distance.
    pub distance: f32,
    /// Closest segment index.
    pub segment: usize,
}

static PROFILE2D_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_SEGMENT_TESTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_BEZIER_TESTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static INTERVAL_CALLS: AtomicU64 = AtomicU64::new(0);
static INTERVAL_HOT_LOOP_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static SHELL_EVAL_STATS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Aggregated native shell evaluator counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShellEvalStats {
    /// Calls into the authored 2D station-profile evaluator.
    pub profile2d_calls: u64,
    /// Profile boundary segment tests performed by the 2D evaluator.
    pub profile2d_segment_tests: u64,
    /// Quadratic profile edge closest-point tests.
    pub profile2d_bezier_tests: u64,
    /// Degenerate quadratic edges that fell back to linear segment distance.
    pub profile2d_fallbacks: u64,
    /// Calls into the native shell interval evaluator.
    pub interval_calls: u64,
    /// Dynamic allocations in native shell interval hot loops.
    pub interval_hot_loop_allocations: u64,
    /// Dynamic allocations in all native shell hot loops.
    pub hot_loop_allocations: u64,
}

/// Enables or disables shell evaluator counters.
pub fn set_shell_eval_stats_enabled(enabled: bool) {
    SHELL_EVAL_STATS_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Resets global native shell evaluator counters before a measured render.
pub fn reset_shell_eval_stats() {
    PROFILE2D_CALLS.store(0, Ordering::Relaxed);
    PROFILE2D_SEGMENT_TESTS.store(0, Ordering::Relaxed);
    PROFILE2D_BEZIER_TESTS.store(0, Ordering::Relaxed);
    PROFILE2D_FALLBACKS.store(0, Ordering::Relaxed);
    INTERVAL_CALLS.store(0, Ordering::Relaxed);
    INTERVAL_HOT_LOOP_ALLOCATIONS.store(0, Ordering::Relaxed);
}

/// Reads global native shell evaluator counters.
pub fn shell_eval_stats() -> ShellEvalStats {
    let interval_hot_loop_allocations =
        INTERVAL_HOT_LOOP_ALLOCATIONS.load(Ordering::Relaxed);
    ShellEvalStats {
        profile2d_calls: PROFILE2D_CALLS.load(Ordering::Relaxed),
        profile2d_segment_tests: PROFILE2D_SEGMENT_TESTS
            .load(Ordering::Relaxed),
        profile2d_bezier_tests: PROFILE2D_BEZIER_TESTS.load(Ordering::Relaxed),
        profile2d_fallbacks: PROFILE2D_FALLBACKS.load(Ordering::Relaxed),
        interval_calls: INTERVAL_CALLS.load(Ordering::Relaxed),
        interval_hot_loop_allocations,
        hot_loop_allocations: interval_hot_loop_allocations,
    }
}

pub(crate) fn record_shell_interval_call() {
    if shell_eval_stats_enabled() {
        INTERVAL_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Evaluates a native shell distance at one point.
#[inline(always)]
pub fn eval_shell_distance(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> ShellSample {
    let sample = match topology.kind {
        ShellOpKind::LineLoft | ShellOpKind::CurveLoft => {
            eval_solid_loft(topology, params, scratch, x, y, z)
        }
        ShellOpKind::ShellHull => {
            eval_shell_hull(topology, params, scratch, x, y, z)
        }
        ShellOpKind::PerimeterExtrude
        | ShellOpKind::Revolve
        | ShellOpKind::Extrude => {
            eval_solid_loft(topology, params, scratch, x, y, z)
        }
    };
    let mut distance = sample.distance;

    if let OpenTopPolicy::BoxCut {
        cut_z,
        half_length,
        half_width,
        offset_x,
    } = topology.open_top
    {
        let x_window = (x - offset_x).abs() - half_length;
        let y_window = y.abs() - half_width;
        let z_cut = cut_z - z;
        let opening = z_cut.max(x_window).max(y_window);
        distance = distance.max(-opening);
    }

    scratch.closest_segment = Some(sample.segment);
    ShellSample {
        distance,
        segment: sample.segment,
    }
}

fn eval_solid_loft(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> ShellSample {
    eval_solid_loft_with_radius_offset(topology, params, scratch, 0.0, x, y, z)
}

fn eval_solid_loft_with_radius_offset(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    scratch: &mut ShellEvalScratch,
    radius_offset: f32,
    x: f32,
    y: f32,
    z: f32,
) -> ShellSample {
    let mut best = ShellSample {
        distance: f32::INFINITY,
        segment: 0,
    };

    for &index in scratch.collect_point_candidates(topology, params, x) {
        let segment = topology.segments[index];
        let distance =
            eval_segment(topology, params, segment, radius_offset, x, y, z);
        if distance < best.distance {
            best = ShellSample {
                distance,
                segment: index,
            };
        }
    }
    best
}

#[inline(always)]
fn eval_shell_hull(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> ShellSample {
    if params.is_empty()
        && let Some(profile) = topology.profile.as_ref()
    {
        return eval_profile_shell_hull(topology, profile, scratch, x, y, z);
    }

    let mut best = ShellSample {
        distance: f32::INFINITY,
        segment: 0,
    };

    for &index in scratch.collect_point_candidates(topology, params, x) {
        let segment = topology.segments[index];
        let half_thickness = topology.shell_thickness * 0.5;
        let mid_surface =
            eval_segment(topology, params, segment, -half_thickness, x, y, z);
        let distance = mid_surface.abs() - half_thickness;
        if distance < best.distance {
            best = ShellSample {
                distance,
                segment: index,
            };
        }
    }
    best
}

#[inline(always)]
fn eval_profile_shell_hull(
    topology: &ShellTopology,
    profile: &ShellProfileTopology,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> ShellSample {
    let mut best = ShellSample {
        distance: f32::INFINITY,
        segment: 0,
    };

    for &index in
        scratch.collect_point_candidates(topology, ShellParamsView::empty(), x)
    {
        let segment = profile.segments[index.min(profile.segments.len() - 1)];
        let outer = eval_profile_solid(profile, segment, 0.0, x, y, z);
        let thickness = topology.shell_thickness.max(0.0);
        let distance = if thickness <= 1.0e-6 || outer > 0.0 {
            outer
        } else {
            let inner =
                eval_profile_solid(profile, segment, thickness, x, y, z);
            outer.max(-inner)
        };
        if distance < best.distance {
            best = ShellSample {
                distance,
                segment: index,
            };
        }
    }

    best
}

#[inline(always)]
fn eval_profile_solid(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    inset: f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let left = profile.sections[segment.left_section];
    let right = profile.sections[segment.right_section];
    let span = right.station - left.station;
    let t = if span.abs() <= 1.0e-6 {
        0.0
    } else {
        ((x - left.station) / span).clamp(0.0, 1.0)
    };

    let section = eval_profile_section_sdf(profile, segment, t, inset, y, z);
    let first = profile.sections[0].station;
    let last = profile.sections[profile.sections.len() - 1].station;
    let left_cap = (first - profile.bow_cap_extension
        + inset * profile.cap_inset_scale)
        - x;
    let right_cap = x
        - (last + profile.stern_cap_extension
            - inset * profile.cap_inset_scale);
    section.max(left_cap.max(right_cap))
}

#[derive(Clone, Copy, Debug)]
struct ProfileNodeSample {
    half_width: f32,
    z: f32,
    continuity: ShellProfileNodeContinuity,
}

#[inline(always)]
fn eval_profile_section_sdf(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    t: f32,
    inset: f32,
    y: f32,
    z: f32,
) -> f32 {
    if shell_eval_stats_enabled() {
        PROFILE2D_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    let mut nodes = [ProfileNodeSample {
        half_width: 0.0,
        z: 0.0,
        continuity: ShellProfileNodeContinuity::Linear,
    }; SHELL_MAX_NODES_PER_CURVE];
    let node_count =
        sample_profile_nodes(profile, segment, t, inset, &mut nodes);

    let y = y.abs();
    let (distance_sq, width) =
        profile_section_distance_sq_and_width([y, z], &nodes[..node_count]);
    let bottom_z = nodes[0].z.min(nodes[node_count - 1].z);
    let top_z = nodes[0].z.max(nodes[node_count - 1].z);
    let inside = z >= bottom_z && z <= top_z && y <= width;
    let distance = distance_sq.sqrt();
    if inside { -distance } else { distance }
}

fn sample_profile_nodes(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    t: f32,
    inset: f32,
    out: &mut [ProfileNodeSample; SHELL_MAX_NODES_PER_CURVE],
) -> usize {
    let left = profile.sections[segment.left_section];
    let right = profile.sections[segment.right_section];
    let node_count = segment.node_count.max(2).min(out.len());
    let inset = inset.max(0.0);
    let mut max_half_width = 0.0_f32;

    for (node_index, out_node) in out.iter_mut().enumerate().take(node_count) {
        let left_node = left.nodes[node_index.min(left.node_count - 1)];
        let right_node = right.nodes[node_index.min(right.node_count - 1)];
        let (half_width, z) = match segment.interpolation {
            ShellProfileSpanInterpolation::Linear => (
                lerp(left_node.half_width, right_node.half_width, t),
                lerp(left_node.z, right_node.z, t),
            ),
            ShellProfileSpanInterpolation::SmoothCatmullRom => {
                let node = segment.nodes[node_index];
                (node.half_width.eval(t), node.z.eval(t))
            }
        };

        let continuity = if left_node.continuity
            == ShellProfileNodeContinuity::Linear
            || right_node.continuity == ShellProfileNodeContinuity::Linear
        {
            ShellProfileNodeContinuity::Linear
        } else {
            ShellProfileNodeContinuity::Smooth
        };

        let half_width = half_width.abs().max(0.0);
        max_half_width = max_half_width.max(half_width);
        *out_node = ProfileNodeSample {
            half_width,
            z,
            continuity,
        };
    }

    if inset > 0.0 {
        let width_floor = max_half_width.min(0.012);
        let width_scale = if max_half_width <= 1.0e-6 {
            0.0
        } else {
            (max_half_width - inset * 0.96).max(width_floor) / max_half_width
        };

        for (node_index, node) in out.iter_mut().enumerate().take(node_count) {
            node.half_width *= width_scale;
            if node_index == 0 {
                node.z += inset * 0.92;
            } else if node_index + 1 == node_count {
                node.z -= inset * 0.92;
            }
        }
    }

    node_count
}

fn profile_section_distance_sq_and_width(
    p: [f32; 2],
    nodes: &[ProfileNodeSample],
) -> (f32, f32) {
    let mut distance_sq = f32::INFINITY;
    let mut width = 0.0_f32;
    let mut slopes = [0.0_f32; SHELL_MAX_NODES_PER_CURVE];
    let z = p[1];
    let stats_enabled = shell_eval_stats_enabled();

    profile_node_slopes(nodes, &mut slopes);

    for index in 0..nodes.len() - 1 {
        if stats_enabled {
            PROFILE2D_SEGMENT_TESTS.fetch_add(1, Ordering::Relaxed);
        }
        let a = nodes[index];
        let c = nodes[index + 1];
        distance_sq = distance_sq.min(profile_edge_distance_sq(
            p,
            a,
            c,
            slopes[index],
            slopes[index + 1],
        ));
        let min_z = a.z.min(c.z);
        let max_z = a.z.max(c.z);
        if z >= min_z && z <= max_z {
            width = width.max(profile_edge_x_at_z(
                a,
                c,
                slopes[index],
                slopes[index + 1],
                z,
            ));
        }
    }

    let top = nodes[nodes.len() - 1];
    if top.half_width > 1.0e-6 {
        if stats_enabled {
            PROFILE2D_SEGMENT_TESTS.fetch_add(1, Ordering::Relaxed);
        }
        distance_sq = distance_sq.min(distance_sq_to_segment(
            p,
            [top.half_width, top.z],
            [0.0, top.z],
        ));
    }

    (distance_sq, width)
}

fn profile_node_slopes(
    nodes: &[ProfileNodeSample],
    out: &mut [f32; SHELL_MAX_NODES_PER_CURVE],
) {
    for index in 0..nodes.len() {
        out[index] =
            if nodes[index].continuity == ShellProfileNodeContinuity::Linear {
                0.0
            } else if index == 0 {
                profile_secant_slope(nodes[0], nodes[1])
            } else if index + 1 == nodes.len() {
                profile_secant_slope(nodes[index - 1], nodes[index])
            } else {
                profile_smooth_slope(
                    nodes[index - 1],
                    nodes[index],
                    nodes[index + 1],
                )
            };
    }
}

fn profile_smooth_slope(
    previous: ProfileNodeSample,
    current: ProfileNodeSample,
    next: ProfileNodeSample,
) -> f32 {
    let h0 = (current.z - previous.z).abs();
    let h1 = (next.z - current.z).abs();
    if h0 <= 1.0e-6 || h1 <= 1.0e-6 {
        return 0.0;
    }

    let d0 = profile_secant_slope(previous, current);
    let d1 = profile_secant_slope(current, next);
    if d0 == 0.0 || d1 == 0.0 {
        return 0.0;
    }
    if d0.signum() != d1.signum() {
        let bias = 0.25 * d0.abs().min(d1.abs()) * d0.signum();
        return (d0 + d1) * 0.5 + bias;
    }

    let w0 = 2.0 * h1 + h0;
    let w1 = h1 + 2.0 * h0;
    (w0 + w1) / (w0 / d0 + w1 / d1)
}

fn profile_secant_slope(a: ProfileNodeSample, c: ProfileNodeSample) -> f32 {
    let dz = c.z - a.z;
    if dz.abs() <= 1.0e-6 {
        0.0
    } else {
        (c.half_width - a.half_width) / dz
    }
}

fn profile_edge_distance_sq(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
) -> f32 {
    if a.continuity == ShellProfileNodeContinuity::Linear
        && c.continuity == ShellProfileNodeContinuity::Linear
    {
        return distance_sq_to_segment(
            p,
            [a.half_width, a.z],
            [c.half_width, c.z],
        );
    }

    if shell_eval_stats_enabled() {
        PROFILE2D_BEZIER_TESTS.fetch_add(1, Ordering::Relaxed);
    }

    let edge_slope = profile_secant_slope(a, c);
    let a_slope = if a.continuity == ShellProfileNodeContinuity::Smooth {
        a_slope
    } else {
        edge_slope
    };
    let c_slope = if c.continuity == ShellProfileNodeContinuity::Smooth {
        c_slope
    } else {
        edge_slope
    };
    distance_sq_to_profile_hermite(p, a, c, a_slope, c_slope)
}

fn profile_edge_x_at_z(
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
    z: f32,
) -> f32 {
    let span = c.z - a.z;
    if span.abs() <= 1.0e-6 {
        return a.half_width.max(c.half_width);
    }
    let t = ((z - a.z) / span).clamp(0.0, 1.0);
    if a.continuity == ShellProfileNodeContinuity::Linear
        && c.continuity == ShellProfileNodeContinuity::Linear
    {
        return lerp(a.half_width, c.half_width, t);
    }
    let edge_slope = profile_secant_slope(a, c);
    let a_slope = if a.continuity == ShellProfileNodeContinuity::Smooth {
        a_slope
    } else {
        edge_slope
    };
    let c_slope = if c.continuity == ShellProfileNodeContinuity::Smooth {
        c_slope
    } else {
        edge_slope
    };
    profile_hermite_width_at(a, c, a_slope, c_slope, t)
}

fn distance_sq_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let ab_len_sq = ab[0].mul_add(ab[0], ab[1] * ab[1]);
    if ab_len_sq <= 1.0e-12 {
        return ap[0].mul_add(ap[0], ap[1] * ap[1]);
    }
    let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq).clamp(0.0, 1.0);
    let closest = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    let dx = p[0] - closest[0];
    let dz = p[1] - closest[1];
    dx.mul_add(dx, dz * dz)
}

fn distance_sq_to_profile_hermite(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
) -> f32 {
    let dz = c.z - a.z;
    if dz.abs() <= 1.0e-8 {
        if shell_eval_stats_enabled() {
            PROFILE2D_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
        return distance_sq_to_segment(
            p,
            [a.half_width, a.z],
            [c.half_width, c.z],
        );
    }

    let mut best = f32::INFINITY;
    let z_t = ((p[1] - a.z) / dz).clamp(0.0, 1.0);
    for candidate in [0.0, 0.25, 0.5, 0.75, 1.0, z_t] {
        let t = refine_profile_hermite_closest_t(
            p, a, c, a_slope, c_slope, candidate,
        );
        best = best
            .min(profile_hermite_distance_sq_at(p, a, c, a_slope, c_slope, t));
    }

    best
}

fn shell_eval_stats_enabled() -> bool {
    SHELL_EVAL_STATS_ENABLED.load(Ordering::Relaxed)
}

fn refine_profile_hermite_closest_t(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
    mut t: f32,
) -> f32 {
    let dz = c.z - a.z;
    for _ in 0..4 {
        let y = profile_hermite_width_at(a, c, a_slope, c_slope, t);
        let dy = profile_hermite_width_derivative(a, c, a_slope, c_slope, t);
        let ddy =
            profile_hermite_width_second_derivative(a, c, a_slope, c_slope, t);
        let z = a.z + dz * t;
        let py = y - p[0];
        let pz = z - p[1];
        let first = py.mul_add(dy, pz * dz);
        let second = dy.mul_add(dy, py * ddy) + dz * dz;
        if second.abs() <= 1.0e-8 {
            break;
        }
        t = (t - first / second).clamp(0.0, 1.0);
    }
    t
}

fn profile_hermite_distance_sq_at(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
    t: f32,
) -> f32 {
    let width = profile_hermite_width_at(a, c, a_slope, c_slope, t);
    let z = lerp(a.z, c.z, t);
    let dw = p[0] - width;
    let dz = p[1] - z;
    dw.mul_add(dw, dz * dz)
}

fn profile_hermite_width_at(
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
    t: f32,
) -> f32 {
    let dz = c.z - a.z;
    let m0 = a_slope * dz;
    let m1 = c_slope * dz;
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * a.half_width
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * c.half_width
        + (t3 - t2) * m1
}

fn profile_hermite_width_derivative(
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
    t: f32,
) -> f32 {
    let dz = c.z - a.z;
    let m0 = a_slope * dz;
    let m1 = c_slope * dz;
    let t2 = t * t;
    (6.0 * t2 - 6.0 * t) * a.half_width
        + (3.0 * t2 - 4.0 * t + 1.0) * m0
        + (-6.0 * t2 + 6.0 * t) * c.half_width
        + (3.0 * t2 - 2.0 * t) * m1
}

fn profile_hermite_width_second_derivative(
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
    t: f32,
) -> f32 {
    let dz = c.z - a.z;
    let m0 = a_slope * dz;
    let m1 = c_slope * dz;
    (12.0 * t - 6.0) * a.half_width
        + (6.0 * t - 4.0) * m0
        + (-12.0 * t + 6.0) * c.half_width
        + (6.0 * t - 2.0) * m1
}

fn eval_segment(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    segment: ShellSegmentTopology,
    radius_offset: f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let left = topology.sections[segment.left_section];
    let right = topology.sections[segment.right_section];
    let left_x = left.station(params);
    let right_x = right.station(params);
    let span = right_x - left_x;
    let t = if span.abs() <= 1.0e-6 {
        0.0
    } else {
        ((x - left_x) / span).clamp(0.0, 1.0)
    };

    let (left_y, left_z) = left.center(params);
    let (right_y, right_z) = right.center(params);
    let (center_y, center_z, radius) = eval_segment_profile(
        segment,
        params,
        left_y,
        left_z,
        right_y,
        right_z,
        left.radius(params),
        right.radius(params),
        t,
    );
    let radius = radius + radius_offset;
    let radius = radius.max(1.0e-5);

    let radial = ((y - center_y)
        .mul_add(y - center_y, (z - center_z) * (z - center_z)))
    .sqrt()
        - radius;
    let x_cap = (left_x - x).max(x - right_x);
    radial.max(x_cap)
}

#[allow(clippy::too_many_arguments)]
fn eval_segment_profile(
    segment: ShellSegmentTopology,
    params: ShellParamsView<'_>,
    left_y: f32,
    left_z: f32,
    right_y: f32,
    right_z: f32,
    left_radius: f32,
    right_radius: f32,
    t: f32,
) -> (f32, f32, f32) {
    if params.is_empty()
        && let ShellSegmentInterpolation::Cubic {
            center_y,
            center_z,
            radius,
        } = segment.interpolation
    {
        return (
            center_y.eval(t),
            center_z.eval(t),
            radius.eval(t).max(1.0e-5),
        );
    }

    (
        lerp(left_y, right_y, t),
        lerp(left_z, right_z, t),
        lerp(left_radius, right_radius, t),
    )
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t
}
