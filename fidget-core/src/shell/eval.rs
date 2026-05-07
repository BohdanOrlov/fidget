//! Pure shell evaluation kernels.

use std::{
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::types::Grad;

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

#[derive(Clone, Copy, Debug, PartialEq)]
struct ShellGradientSample {
    distance: f32,
    segment: usize,
    gradient: [f32; 3],
}

static PROFILE2D_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_DISTANCE_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_GRADIENT_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_SEGMENT_TESTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_BEZIER_TESTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_EDGES_CONSIDERED: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_EDGES_AABB_PRUNED: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_LINEAR_EDGES: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_SMOOTH_EDGES: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_ENDPOINT_BEST_KEPT: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_EDGES_REFINED: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_SEED_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_NEWTON_ITERATIONS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_ITERATION_1_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_ITERATION_2_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_ITERATION_3_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_ITERATION_4_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_CLAMPED_ENDPOINT_ATTEMPTS: AtomicU64 =
    AtomicU64::new(0);
static PROFILE2D_HERMITE_DUPLICATE_T_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_DISTANCE_EVALUATIONS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_WINS_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_ENDPOINT_WINS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_QUARTER_WINS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_QUARTER_25_WINS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_QUARTER_50_WINS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_QUARTER_75_WINS: AtomicU64 = AtomicU64::new(0);
static PROFILE2D_HERMITE_HEIGHT_WINS: AtomicU64 = AtomicU64::new(0);
static INTERVAL_CALLS: AtomicU64 = AtomicU64::new(0);
static INTERVAL_HOT_LOOP_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static SHELL_EVAL_STATS_ENABLED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) static SHELL_EVAL_STATS_TEST_LOCK: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

/// Aggregated native shell evaluator counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShellEvalStats {
    /// Calls into the authored 2D station-profile evaluator.
    pub profile2d_calls: u64,
    /// 2D station-profile evaluator calls from regular distance evaluation.
    pub profile2d_distance_calls: u64,
    /// 2D station-profile evaluator calls from gradient evaluation.
    pub profile2d_gradient_calls: u64,
    /// Profile boundary segment tests performed by the 2D evaluator.
    pub profile2d_segment_tests: u64,
    /// Quadratic profile edge closest-point tests.
    pub profile2d_bezier_tests: u64,
    /// Degenerate quadratic edges that fell back to linear segment distance.
    pub profile2d_fallbacks: u64,
    /// Candidate profile contour edges considered after z-span filtering.
    pub profile2d_edges_considered: u64,
    /// Candidate profile contour edges pruned by AABB lower-bound tests.
    pub profile2d_edges_aabb_pruned: u64,
    /// Candidate profile contour edges with linear endpoint continuity.
    pub profile2d_linear_edges: u64,
    /// Candidate profile contour edges with at least one smooth endpoint.
    pub profile2d_smooth_edges: u64,
    /// Profile calls whose final closest feature remained a station endpoint.
    pub profile2d_endpoint_best_kept: u64,
    /// Smooth profile edges that reached Hermite closest-point refinement.
    pub profile2d_hermite_edges_refined: u64,
    /// Hermite closest-point seed refinements attempted.
    pub profile2d_hermite_seed_attempts: u64,
    /// Newton refinement iterations across Hermite seed attempts.
    pub profile2d_hermite_newton_iterations: u64,
    /// Hermite seed attempts that converged or stopped after 1 Newton iteration.
    pub profile2d_hermite_iteration_1_attempts: u64,
    /// Hermite seed attempts that converged or stopped after 2 Newton iterations.
    pub profile2d_hermite_iteration_2_attempts: u64,
    /// Hermite seed attempts that converged or stopped after 3 Newton iterations.
    pub profile2d_hermite_iteration_3_attempts: u64,
    /// Hermite seed attempts that used all 4 Newton iterations.
    pub profile2d_hermite_iteration_4_attempts: u64,
    /// Hermite seed attempts whose refined point clamped to an endpoint.
    pub profile2d_hermite_clamped_endpoint_attempts: u64,
    /// Hermite seed attempts whose refined t duplicated an earlier seed result.
    pub profile2d_hermite_duplicate_t_attempts: u64,
    /// Hermite distance evaluations after duplicate refined roots are removed.
    pub profile2d_hermite_distance_evaluations: u64,
    /// Hermite closest-point evaluations with an observed winning seed.
    pub profile2d_hermite_wins_total: u64,
    /// Hermite refinements where an endpoint seed produced the closest point.
    pub profile2d_hermite_endpoint_wins: u64,
    /// Hermite refinements where a quarter-span seed produced the closest point.
    pub profile2d_hermite_quarter_wins: u64,
    /// Hermite refinements where the 0.25 seed produced the closest point.
    pub profile2d_hermite_quarter_25_wins: u64,
    /// Hermite refinements where the 0.50 seed produced the closest point.
    pub profile2d_hermite_quarter_50_wins: u64,
    /// Hermite refinements where the 0.75 seed produced the closest point.
    pub profile2d_hermite_quarter_75_wins: u64,
    /// Hermite refinements where the same-height seed produced the closest point.
    pub profile2d_hermite_height_wins: u64,
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
    PROFILE2D_DISTANCE_CALLS.store(0, Ordering::Relaxed);
    PROFILE2D_GRADIENT_CALLS.store(0, Ordering::Relaxed);
    PROFILE2D_SEGMENT_TESTS.store(0, Ordering::Relaxed);
    PROFILE2D_BEZIER_TESTS.store(0, Ordering::Relaxed);
    PROFILE2D_FALLBACKS.store(0, Ordering::Relaxed);
    PROFILE2D_EDGES_CONSIDERED.store(0, Ordering::Relaxed);
    PROFILE2D_EDGES_AABB_PRUNED.store(0, Ordering::Relaxed);
    PROFILE2D_LINEAR_EDGES.store(0, Ordering::Relaxed);
    PROFILE2D_SMOOTH_EDGES.store(0, Ordering::Relaxed);
    PROFILE2D_ENDPOINT_BEST_KEPT.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_EDGES_REFINED.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_SEED_ATTEMPTS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_NEWTON_ITERATIONS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_ITERATION_1_ATTEMPTS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_ITERATION_2_ATTEMPTS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_ITERATION_3_ATTEMPTS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_ITERATION_4_ATTEMPTS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_CLAMPED_ENDPOINT_ATTEMPTS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_DUPLICATE_T_ATTEMPTS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_DISTANCE_EVALUATIONS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_WINS_TOTAL.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_ENDPOINT_WINS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_QUARTER_WINS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_QUARTER_25_WINS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_QUARTER_50_WINS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_QUARTER_75_WINS.store(0, Ordering::Relaxed);
    PROFILE2D_HERMITE_HEIGHT_WINS.store(0, Ordering::Relaxed);
    INTERVAL_CALLS.store(0, Ordering::Relaxed);
    INTERVAL_HOT_LOOP_ALLOCATIONS.store(0, Ordering::Relaxed);
}

/// Reads global native shell evaluator counters.
pub fn shell_eval_stats() -> ShellEvalStats {
    let interval_hot_loop_allocations =
        INTERVAL_HOT_LOOP_ALLOCATIONS.load(Ordering::Relaxed);
    ShellEvalStats {
        profile2d_calls: PROFILE2D_CALLS.load(Ordering::Relaxed),
        profile2d_distance_calls: PROFILE2D_DISTANCE_CALLS
            .load(Ordering::Relaxed),
        profile2d_gradient_calls: PROFILE2D_GRADIENT_CALLS
            .load(Ordering::Relaxed),
        profile2d_segment_tests: PROFILE2D_SEGMENT_TESTS
            .load(Ordering::Relaxed),
        profile2d_bezier_tests: PROFILE2D_BEZIER_TESTS.load(Ordering::Relaxed),
        profile2d_fallbacks: PROFILE2D_FALLBACKS.load(Ordering::Relaxed),
        profile2d_edges_considered: PROFILE2D_EDGES_CONSIDERED
            .load(Ordering::Relaxed),
        profile2d_edges_aabb_pruned: PROFILE2D_EDGES_AABB_PRUNED
            .load(Ordering::Relaxed),
        profile2d_linear_edges: PROFILE2D_LINEAR_EDGES
            .load(Ordering::Relaxed),
        profile2d_smooth_edges: PROFILE2D_SMOOTH_EDGES
            .load(Ordering::Relaxed),
        profile2d_endpoint_best_kept: PROFILE2D_ENDPOINT_BEST_KEPT
            .load(Ordering::Relaxed),
        profile2d_hermite_edges_refined: PROFILE2D_HERMITE_EDGES_REFINED
            .load(Ordering::Relaxed),
        profile2d_hermite_seed_attempts: PROFILE2D_HERMITE_SEED_ATTEMPTS
            .load(Ordering::Relaxed),
        profile2d_hermite_newton_iterations:
            PROFILE2D_HERMITE_NEWTON_ITERATIONS.load(Ordering::Relaxed),
        profile2d_hermite_iteration_1_attempts:
            PROFILE2D_HERMITE_ITERATION_1_ATTEMPTS.load(Ordering::Relaxed),
        profile2d_hermite_iteration_2_attempts:
            PROFILE2D_HERMITE_ITERATION_2_ATTEMPTS.load(Ordering::Relaxed),
        profile2d_hermite_iteration_3_attempts:
            PROFILE2D_HERMITE_ITERATION_3_ATTEMPTS.load(Ordering::Relaxed),
        profile2d_hermite_iteration_4_attempts:
            PROFILE2D_HERMITE_ITERATION_4_ATTEMPTS.load(Ordering::Relaxed),
        profile2d_hermite_clamped_endpoint_attempts:
            PROFILE2D_HERMITE_CLAMPED_ENDPOINT_ATTEMPTS
                .load(Ordering::Relaxed),
        profile2d_hermite_duplicate_t_attempts:
            PROFILE2D_HERMITE_DUPLICATE_T_ATTEMPTS.load(Ordering::Relaxed),
        profile2d_hermite_distance_evaluations:
            PROFILE2D_HERMITE_DISTANCE_EVALUATIONS.load(Ordering::Relaxed),
        profile2d_hermite_wins_total: PROFILE2D_HERMITE_WINS_TOTAL
            .load(Ordering::Relaxed),
        profile2d_hermite_endpoint_wins: PROFILE2D_HERMITE_ENDPOINT_WINS
            .load(Ordering::Relaxed),
        profile2d_hermite_quarter_wins: PROFILE2D_HERMITE_QUARTER_WINS
            .load(Ordering::Relaxed),
        profile2d_hermite_quarter_25_wins:
            PROFILE2D_HERMITE_QUARTER_25_WINS.load(Ordering::Relaxed),
        profile2d_hermite_quarter_50_wins:
            PROFILE2D_HERMITE_QUARTER_50_WINS.load(Ordering::Relaxed),
        profile2d_hermite_quarter_75_wins:
            PROFILE2D_HERMITE_QUARTER_75_WINS.load(Ordering::Relaxed),
        profile2d_hermite_height_wins: PROFILE2D_HERMITE_HEIGHT_WINS
            .load(Ordering::Relaxed),
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

#[derive(Clone, Copy)]
enum Profile2dCallSource {
    Distance,
    Gradient,
}

#[inline(always)]
fn record_profile2d_call(source: Profile2dCallSource) {
    if shell_eval_stats_enabled() {
        PROFILE2D_CALLS.fetch_add(1, Ordering::Relaxed);
        match source {
            Profile2dCallSource::Distance => {
                PROFILE2D_DISTANCE_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            Profile2dCallSource::Gradient => {
                PROFILE2D_GRADIENT_CALLS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Evaluates a native shell gradient at one point.
#[inline(always)]
pub fn eval_shell_grad(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    scratch: &mut ShellEvalScratch,
    x: Grad,
    y: Grad,
    z: Grad,
) -> Grad {
    let sample =
        eval_shell_gradient_sample(topology, params, scratch, x.v, y.v, z.v);
    Grad::new(
        sample.distance,
        sample.gradient[0].mul_add(
            x.dx,
            sample.gradient[1].mul_add(y.dx, sample.gradient[2] * z.dx),
        ),
        sample.gradient[0].mul_add(
            x.dy,
            sample.gradient[1].mul_add(y.dy, sample.gradient[2] * z.dy),
        ),
        sample.gradient[0].mul_add(
            x.dz,
            sample.gradient[1].mul_add(y.dz, sample.gradient[2] * z.dz),
        ),
    )
}

fn eval_shell_gradient_sample(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> ShellGradientSample {
    let mut sample = if topology.kind == ShellOpKind::ShellHull
        && params.is_empty()
        && let Some(profile) = topology.profile.as_ref()
    {
        eval_profile_shell_hull_gradient(topology, profile, scratch, x, y, z)
    } else {
        eval_shell_gradient_finite_difference(
            topology, params, scratch, x, y, z,
        )
    };

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
        let (opening, opening_gradient) =
            if z_cut >= x_window && z_cut >= y_window {
                (z_cut, [0.0, 0.0, -1.0])
            } else if x_window >= y_window {
                (x_window, [(x - offset_x).signum(), 0.0, 0.0])
            } else {
                (y_window, [0.0, y.signum(), 0.0])
            };
        let cut_distance = -opening;
        if cut_distance > sample.distance {
            sample.distance = cut_distance;
            sample.gradient = [
                -opening_gradient[0],
                -opening_gradient[1],
                -opening_gradient[2],
            ];
        }
    }

    scratch.closest_segment = Some(sample.segment);
    sample
}

fn eval_shell_gradient_finite_difference(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> ShellGradientSample {
    let eps = 1.0e-3;
    let value = eval_shell_distance(topology, params, scratch, x, y, z);
    let dx = (eval_shell_distance(topology, params, scratch, x + eps, y, z)
        .distance
        - eval_shell_distance(topology, params, scratch, x - eps, y, z)
            .distance)
        / (2.0 * eps);
    let dy = (eval_shell_distance(topology, params, scratch, x, y + eps, z)
        .distance
        - eval_shell_distance(topology, params, scratch, x, y - eps, z)
            .distance)
        / (2.0 * eps);
    let dz = (eval_shell_distance(topology, params, scratch, x, y, z + eps)
        .distance
        - eval_shell_distance(topology, params, scratch, x, y, z - eps)
            .distance)
        / (2.0 * eps);
    ShellGradientSample {
        distance: value.distance,
        segment: value.segment,
        gradient: [dx, dy, dz],
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
        let outer = eval_profile_solid(
            profile,
            segment,
            0.0,
            x,
            y,
            z,
            Profile2dCallSource::Distance,
        );
        let thickness = topology.shell_thickness.max(0.0);
        let distance = if thickness <= 1.0e-6 || outer > 0.0 {
            outer
        } else {
            let inner = eval_profile_solid(
                profile,
                segment,
                thickness,
                x,
                y,
                z,
                Profile2dCallSource::Distance,
            );
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
fn eval_profile_shell_hull_gradient(
    topology: &ShellTopology,
    profile: &ShellProfileTopology,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> ShellGradientSample {
    let mut best = ShellGradientSample {
        distance: f32::INFINITY,
        segment: 0,
        gradient: [0.0, 0.0, 1.0],
    };

    for &index in
        scratch.collect_point_candidates(topology, ShellParamsView::empty(), x)
    {
        let segment = profile.segments[index.min(profile.segments.len() - 1)];
        let outer = eval_profile_solid_gradient(profile, segment, 0.0, x, y, z);
        let thickness = topology.shell_thickness.max(0.0);
        let sample = if thickness <= 1.0e-6 || outer.distance > 0.0 {
            outer
        } else {
            let inner = eval_profile_solid_gradient(
                profile, segment, thickness, x, y, z,
            );
            if outer.distance >= -inner.distance {
                outer
            } else {
                ShellGradientSample {
                    distance: -inner.distance,
                    segment: index,
                    gradient: [
                        -inner.gradient[0],
                        -inner.gradient[1],
                        -inner.gradient[2],
                    ],
                }
            }
        };
        if sample.distance < best.distance {
            best = ShellGradientSample {
                distance: sample.distance,
                segment: index,
                gradient: sample.gradient,
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
    source: Profile2dCallSource,
) -> f32 {
    let left = profile.sections[segment.left_section];
    let right = profile.sections[segment.right_section];
    let span = right.station - left.station;
    let t = if span.abs() <= 1.0e-6 {
        0.0
    } else {
        ((x - left.station) / span).clamp(0.0, 1.0)
    };

    let section =
        eval_profile_section_sdf(profile, segment, t, inset, y, z, source);
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

#[inline(always)]
fn eval_profile_solid_gradient(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    inset: f32,
    x: f32,
    y: f32,
    z: f32,
) -> ShellGradientSample {
    let left = profile.sections[segment.left_section];
    let right = profile.sections[segment.right_section];
    let span = right.station - left.station;
    let t = if span.abs() <= 1.0e-6 {
        0.0
    } else {
        ((x - left.station) / span).clamp(0.0, 1.0)
    };

    let mut section =
        eval_profile_section_sdf_gradient(profile, segment, t, inset, x, y, z);
    let first = profile.sections[0].station;
    let last = profile.sections[profile.sections.len() - 1].station;
    let left_cap = (first - profile.bow_cap_extension
        + inset * profile.cap_inset_scale)
        - x;
    let right_cap = x
        - (last + profile.stern_cap_extension
            - inset * profile.cap_inset_scale);

    if left_cap >= section.distance && left_cap >= right_cap {
        section.distance = left_cap;
        section.gradient = [-1.0, 0.0, 0.0];
    } else if right_cap >= section.distance {
        section.distance = right_cap;
        section.gradient = [1.0, 0.0, 0.0];
    }
    section
}

#[derive(Clone, Copy, Debug)]
struct ProfileNodeSample {
    half_width: f32,
    z: f32,
    continuity: ShellProfileNodeContinuity,
}

#[derive(Clone, Copy, Debug)]
struct ProfileNodeSampleSet {
    node_count: usize,
    monotonic_in_z: bool,
}

#[derive(Clone, Copy, Debug)]
struct ProfileSectionDistance {
    distance_sq: f32,
    inside: bool,
    closest: [f32; 2],
}

#[derive(Clone, Copy, Debug)]
struct ProfileEdgeDistance {
    distance_sq: f32,
    closest: [f32; 2],
}

#[inline(always)]
fn eval_profile_section_sdf(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    t: f32,
    inset: f32,
    y: f32,
    z: f32,
    source: Profile2dCallSource,
) -> f32 {
    record_profile2d_call(source);
    let mut nodes = [ProfileNodeSample {
        half_width: 0.0,
        z: 0.0,
        continuity: ShellProfileNodeContinuity::Linear,
    }; SHELL_MAX_NODES_PER_CURVE];
    let sampled = sample_profile_nodes(profile, segment, t, inset, &mut nodes);

    let y = y.abs();
    let section = profile_section_distance(
        [y, z],
        &nodes[..sampled.node_count],
        sampled.monotonic_in_z,
    );
    let distance = section.distance_sq.sqrt();
    if section.inside { -distance } else { distance }
}

#[inline(always)]
fn eval_profile_section_sdf_gradient(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    t: f32,
    inset: f32,
    x: f32,
    y: f32,
    z: f32,
) -> ShellGradientSample {
    record_profile2d_call(Profile2dCallSource::Gradient);
    let mut nodes = [ProfileNodeSample {
        half_width: 0.0,
        z: 0.0,
        continuity: ShellProfileNodeContinuity::Linear,
    }; SHELL_MAX_NODES_PER_CURVE];
    let sampled = sample_profile_nodes(profile, segment, t, inset, &mut nodes);
    let y_abs = y.abs();
    let section = profile_section_distance(
        [y_abs, z],
        &nodes[..sampled.node_count],
        sampled.monotonic_in_z,
    );
    let unsigned = section.distance_sq.sqrt();
    let sign = if section.inside { -1.0 } else { 1.0 };
    let inv_len = if unsigned <= 1.0e-8 {
        0.0
    } else {
        1.0 / unsigned
    };
    let normal_y = (y_abs - section.closest[0]) * inv_len * y.signum();
    let normal_z = (z - section.closest[1]) * inv_len;

    // Keep the station derivative conservative for this first analytic pass:
    // y/z are computed from the chosen contour feature, while x remains a
    // two-sided sample of the exact solid profile including caps.
    let eps = 1.0e-3;
    let dx = (eval_profile_solid(
        profile,
        segment,
        inset,
        x + eps,
        y,
        z,
        Profile2dCallSource::Gradient,
    ) - eval_profile_solid(
        profile,
        segment,
        inset,
        x - eps,
        y,
        z,
        Profile2dCallSource::Gradient,
    ))
        / (2.0 * eps);

    ShellGradientSample {
        distance: sign * unsigned,
        segment: segment.left_section,
        gradient: [dx, sign * normal_y, sign * normal_z],
    }
}

#[inline(always)]
fn sample_profile_nodes(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    t: f32,
    inset: f32,
    out: &mut [ProfileNodeSample; SHELL_MAX_NODES_PER_CURVE],
) -> ProfileNodeSampleSet {
    let left = profile.sections[segment.left_section];
    let right = profile.sections[segment.right_section];
    let node_count = segment.node_count.max(2).min(out.len());
    let inset = inset.max(0.0);
    let mut max_half_width = 0.0_f32;
    let mut nondecreasing = true;
    let mut nonincreasing = true;
    let mut previous_z = 0.0_f32;

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
        if node_index > 0 {
            if z < previous_z {
                nondecreasing = false;
            }
            if z > previous_z {
                nonincreasing = false;
            }
        }
        previous_z = z;
    }
    let mut monotonic_in_z = nondecreasing || nonincreasing;

    if inset > 0.0 {
        let width_floor = max_half_width.min(0.012);
        let width_scale = if max_half_width <= 1.0e-6 {
            0.0
        } else {
            (max_half_width - inset * 0.96).max(width_floor) / max_half_width
        };

        nondecreasing = true;
        nonincreasing = true;
        previous_z = 0.0;
        for (node_index, node) in out.iter_mut().enumerate().take(node_count) {
            node.half_width *= width_scale;
            if node_index == 0 {
                node.z += inset * 0.92;
            } else if node_index + 1 == node_count {
                node.z -= inset * 0.92;
            }
            if node_index > 0 {
                if node.z < previous_z {
                    nondecreasing = false;
                }
                if node.z > previous_z {
                    nonincreasing = false;
                }
            }
            previous_z = node.z;
        }
        monotonic_in_z = nondecreasing || nonincreasing;
    }

    ProfileNodeSampleSet {
        node_count,
        monotonic_in_z,
    }
}

#[inline(always)]
fn profile_section_distance(
    p: [f32; 2],
    nodes: &[ProfileNodeSample],
    monotonic_in_z: bool,
) -> ProfileSectionDistance {
    let mut best = profile_endpoint_distance(p, nodes);
    let mut slopes = [0.0_f32; SHELL_MAX_NODES_PER_CURVE];
    let stats_enabled = shell_eval_stats_enabled();

    profile_node_slopes(nodes, &mut slopes);
    if monotonic_in_z {
        best = profile_section_distance_monotonic(
            p,
            nodes,
            &slopes,
            best,
            stats_enabled,
        );
    } else {
        best = profile_section_distance_full_scan(
            p,
            nodes,
            &slopes,
            best,
            stats_enabled,
        );
    }

    let top = nodes[nodes.len() - 1];
    best = profile_top_edge_distance(p, top, best, stats_enabled);
    if stats_enabled && profile_best_is_endpoint(best, nodes) {
        PROFILE2D_ENDPOINT_BEST_KEPT.fetch_add(1, Ordering::Relaxed);
    }
    let inside = if monotonic_in_z {
        profile_contour_contains_point_monotonic(p, nodes, &slopes)
    } else {
        profile_contour_contains_point(p, nodes, &slopes)
    };

    ProfileSectionDistance {
        distance_sq: best.distance_sq,
        inside,
        closest: best.closest,
    }
}

#[inline(always)]
fn profile_best_is_endpoint(
    best: ProfileEdgeDistance,
    nodes: &[ProfileNodeSample],
) -> bool {
    nodes.iter().any(|node| {
        (best.closest[0] - node.half_width).abs() <= 1.0e-6
            && (best.closest[1] - node.z).abs() <= 1.0e-6
    })
}

#[inline(always)]
fn profile_section_distance_full_scan(
    p: [f32; 2],
    nodes: &[ProfileNodeSample],
    slopes: &[f32; SHELL_MAX_NODES_PER_CURVE],
    mut best: ProfileEdgeDistance,
    stats_enabled: bool,
) -> ProfileEdgeDistance {
    for index in 0..nodes.len() - 1 {
        best = profile_edge_distance_if_relevant(
            p,
            nodes[index],
            nodes[index + 1],
            slopes[index],
            slopes[index + 1],
            best,
            stats_enabled,
        );
    }
    best
}

#[inline(always)]
fn profile_section_distance_monotonic(
    p: [f32; 2],
    nodes: &[ProfileNodeSample],
    slopes: &[f32; SHELL_MAX_NODES_PER_CURVE],
    mut best: ProfileEdgeDistance,
    stats_enabled: bool,
) -> ProfileEdgeDistance {
    let z = p[1];
    let pivot = closest_profile_edge_by_z(z, nodes);
    best = profile_edge_distance_if_relevant(
        p,
        nodes[pivot],
        nodes[pivot + 1],
        slopes[pivot],
        slopes[pivot + 1],
        best,
        stats_enabled,
    );

    let mut lower = pivot;
    let mut upper = pivot + 1;
    while lower > 0 || upper < nodes.len() - 1 {
        let mut progressed = false;

        if lower > 0 {
            let index = lower - 1;
            if profile_edge_z_gap_sq(z, nodes[index], nodes[index + 1])
                <= best.distance_sq
            {
                best = profile_edge_distance_if_relevant(
                    p,
                    nodes[index],
                    nodes[index + 1],
                    slopes[index],
                    slopes[index + 1],
                    best,
                    stats_enabled,
                );
                lower = index;
                progressed = true;
            } else {
                lower = 0;
            }
        }

        if upper < nodes.len() - 1 {
            let index = upper;
            if profile_edge_z_gap_sq(z, nodes[index], nodes[index + 1])
                <= best.distance_sq
            {
                best = profile_edge_distance_if_relevant(
                    p,
                    nodes[index],
                    nodes[index + 1],
                    slopes[index],
                    slopes[index + 1],
                    best,
                    stats_enabled,
                );
                upper = index + 1;
                progressed = true;
            } else {
                upper = nodes.len() - 1;
            }
        }

        if !progressed {
            break;
        }
    }

    best
}

#[inline(always)]
fn profile_edge_distance_if_relevant(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
    best: ProfileEdgeDistance,
    stats_enabled: bool,
) -> ProfileEdgeDistance {
    let (a_slope, c_slope) =
        profile_effective_edge_slopes(a, c, a_slope, c_slope);
    let linear_edge = a.continuity == ShellProfileNodeContinuity::Linear
        && c.continuity == ShellProfileNodeContinuity::Linear;
    if stats_enabled {
        PROFILE2D_EDGES_CONSIDERED.fetch_add(1, Ordering::Relaxed);
        if linear_edge {
            PROFILE2D_LINEAR_EDGES.fetch_add(1, Ordering::Relaxed);
        } else {
            PROFILE2D_SMOOTH_EDGES.fetch_add(1, Ordering::Relaxed);
        }
    }
    let edge = if linear_edge {
        if profile_edge_aabb_lower_bound_sq(p, a, c, a_slope, c_slope)
            >= best.distance_sq
        {
            if stats_enabled {
                PROFILE2D_EDGES_AABB_PRUNED.fetch_add(1, Ordering::Relaxed);
            }
            return best;
        }
        if stats_enabled {
            PROFILE2D_SEGMENT_TESTS.fetch_add(1, Ordering::Relaxed);
        }
        distance_to_segment(
            p,
            [a.half_width, a.z],
            [c.half_width, c.z],
        )
    } else {
        let hermite = ProfileHermiteEdge::new(a, c, a_slope, c_slope);
        if profile_edge_aabb_lower_bound_sq_for_hermite(p, a, c, hermite)
            >= best.distance_sq
        {
            if stats_enabled {
                PROFILE2D_EDGES_AABB_PRUNED.fetch_add(1, Ordering::Relaxed);
            }
            return best;
        }
        if profile_edge_bezier_hull_cannot_beat(p, hermite, best.distance_sq)
        {
            return best;
        }
        if stats_enabled {
            PROFILE2D_SEGMENT_TESTS.fetch_add(1, Ordering::Relaxed);
            PROFILE2D_BEZIER_TESTS.fetch_add(1, Ordering::Relaxed);
            PROFILE2D_HERMITE_EDGES_REFINED.fetch_add(1, Ordering::Relaxed);
        }
        distance_to_profile_hermite_edge(p, a, c, hermite, stats_enabled)
    };
    if edge.distance_sq < best.distance_sq {
        edge
    } else {
        best
    }
}

#[inline(always)]
fn profile_top_edge_distance(
    p: [f32; 2],
    top: ProfileNodeSample,
    best: ProfileEdgeDistance,
    stats_enabled: bool,
) -> ProfileEdgeDistance {
    if top.half_width <= 1.0e-6 {
        return best;
    }
    if distance_sq_to_aabb(p, 0.0, top.half_width, top.z, top.z)
        > best.distance_sq
    {
        return best;
    }
    if stats_enabled {
        PROFILE2D_SEGMENT_TESTS.fetch_add(1, Ordering::Relaxed);
    }
    let edge = distance_to_segment(p, [top.half_width, top.z], [0.0, top.z]);
    if edge.distance_sq < best.distance_sq {
        edge
    } else {
        best
    }
}

#[inline(always)]
fn profile_contour_contains_point(
    p: [f32; 2],
    nodes: &[ProfileNodeSample],
    slopes: &[f32; SHELL_MAX_NODES_PER_CURVE],
) -> bool {
    let x = p[0].max(0.0);
    let z = p[1];
    let bottom_z = nodes[0].z.min(nodes[nodes.len() - 1].z);
    let top = nodes[nodes.len() - 1];
    let top_z = nodes[0].z.max(top.z);
    if z < bottom_z || z > top_z {
        return false;
    }
    if x <= 1.0e-6 {
        return true;
    }
    if (z - top.z).abs() <= 1.0e-6 && x <= top.half_width {
        return true;
    }

    let mut inside = false;
    for index in 0..nodes.len() - 1 {
        let a = nodes[index];
        let c = nodes[index + 1];
        if (a.z > z) != (c.z > z) {
            let edge_x =
                profile_edge_x_at_z(a, c, slopes[index], slopes[index + 1], z);
            if edge_x > x {
                inside = !inside;
            }
        }
    }
    inside
}

#[inline(always)]
fn profile_contour_contains_point_monotonic(
    p: [f32; 2],
    nodes: &[ProfileNodeSample],
    slopes: &[f32; SHELL_MAX_NODES_PER_CURVE],
) -> bool {
    let x = p[0].max(0.0);
    let z = p[1];
    let bottom_z = nodes[0].z.min(nodes[nodes.len() - 1].z);
    let top = nodes[nodes.len() - 1];
    let top_z = nodes[0].z.max(top.z);
    if z < bottom_z || z > top_z {
        return false;
    }
    if x <= 1.0e-6 {
        return true;
    }
    if (z - top.z).abs() <= 1.0e-6 && x <= top.half_width {
        return true;
    }

    for index in 0..nodes.len() - 1 {
        let a = nodes[index];
        let c = nodes[index + 1];
        if (a.z > z) != (c.z > z) {
            let edge_x =
                profile_edge_x_at_z(a, c, slopes[index], slopes[index + 1], z);
            return edge_x > x;
        }
    }
    false
}

#[inline(always)]
fn closest_profile_edge_by_z(z: f32, nodes: &[ProfileNodeSample]) -> usize {
    let mut best_index = 0usize;
    let mut best_gap_sq = f32::INFINITY;
    for index in 0..nodes.len() - 1 {
        let gap_sq = profile_edge_z_gap_sq(z, nodes[index], nodes[index + 1]);
        if gap_sq < best_gap_sq {
            best_gap_sq = gap_sq;
            best_index = index;
        }
    }
    best_index
}

#[inline(always)]
fn profile_edge_z_gap_sq(
    z: f32,
    a: ProfileNodeSample,
    c: ProfileNodeSample,
) -> f32 {
    let min_z = a.z.min(c.z);
    let max_z = a.z.max(c.z);
    let dz = if z < min_z {
        min_z - z
    } else if z > max_z {
        z - max_z
    } else {
        0.0
    };
    dz * dz
}

#[inline(always)]
fn profile_endpoint_distance(
    p: [f32; 2],
    nodes: &[ProfileNodeSample],
) -> ProfileEdgeDistance {
    let mut best = ProfileEdgeDistance {
        distance_sq: f32::INFINITY,
        closest: [0.0, 0.0],
    };
    for node in nodes {
        let dx = p[0] - node.half_width;
        let dz = p[1] - node.z;
        let distance_sq = dx.mul_add(dx, dz * dz);
        if distance_sq < best.distance_sq {
            best = ProfileEdgeDistance {
                distance_sq,
                closest: [node.half_width, node.z],
            };
        }
    }

    let top = nodes[nodes.len() - 1];
    if top.half_width > 1.0e-6 {
        let dx = p[0];
        let dz = p[1] - top.z;
        let distance_sq = dx.mul_add(dx, dz * dz);
        if distance_sq < best.distance_sq {
            best = ProfileEdgeDistance {
                distance_sq,
                closest: [0.0, top.z],
            };
        }
    }
    best
}

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
fn profile_secant_slope(a: ProfileNodeSample, c: ProfileNodeSample) -> f32 {
    let dz = c.z - a.z;
    if dz.abs() <= 1.0e-6 {
        0.0
    } else {
        (c.half_width - a.half_width) / dz
    }
}

#[inline(always)]
fn profile_edge_aabb_lower_bound_sq(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
) -> f32 {
    let (min_x, max_x) = profile_edge_x_bounds(a, c, a_slope, c_slope);
    distance_sq_to_aabb(p, min_x, max_x, a.z.min(c.z), a.z.max(c.z))
}

#[inline(always)]
fn profile_edge_aabb_lower_bound_sq_for_hermite(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    edge: ProfileHermiteEdge,
) -> f32 {
    let (min_x, max_x) = profile_edge_x_bounds_for_hermite(a, c, edge);
    distance_sq_to_aabb(p, min_x, max_x, a.z.min(c.z), a.z.max(c.z))
}

#[inline(always)]
fn profile_edge_x_bounds(
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
) -> (f32, f32) {
    let mut min_x = a.half_width.min(c.half_width);
    let mut max_x = a.half_width.max(c.half_width);
    if a.continuity != ShellProfileNodeContinuity::Linear
        || c.continuity != ShellProfileNodeContinuity::Linear
    {
        let edge = ProfileHermiteEdge::new(a, c, a_slope, c_slope);
        (min_x, max_x) = profile_edge_x_bounds_for_hermite(a, c, edge);
    }
    (min_x, max_x)
}

#[inline(always)]
fn profile_edge_x_bounds_for_hermite(
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    edge: ProfileHermiteEdge,
) -> (f32, f32) {
    let mut min_x = a.half_width.min(c.half_width);
    let mut max_x = a.half_width.max(c.half_width);
    profile_hermite_width_extrema(edge, &mut min_x, &mut max_x);
    (min_x, max_x)
}

#[inline(always)]
fn profile_effective_edge_slopes(
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    a_slope: f32,
    c_slope: f32,
) -> (f32, f32) {
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
    (a_slope, c_slope)
}

#[inline(always)]
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
    let (a_slope, c_slope) =
        profile_effective_edge_slopes(a, c, a_slope, c_slope);
    profile_hermite_width_at(a, c, a_slope, c_slope, t)
}

#[inline(always)]
fn distance_to_segment(
    p: [f32; 2],
    a: [f32; 2],
    b: [f32; 2],
) -> ProfileEdgeDistance {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let ab_len_sq = ab[0].mul_add(ab[0], ab[1] * ab[1]);
    if ab_len_sq <= 1.0e-12 {
        return ProfileEdgeDistance {
            distance_sq: ap[0].mul_add(ap[0], ap[1] * ap[1]),
            closest: a,
        };
    }
    let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq).clamp(0.0, 1.0);
    let closest = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    let dx = p[0] - closest[0];
    let dz = p[1] - closest[1];
    ProfileEdgeDistance {
        distance_sq: dx.mul_add(dx, dz * dz),
        closest,
    }
}

#[inline(always)]
fn distance_sq_to_aabb(
    p: [f32; 2],
    min_x: f32,
    max_x: f32,
    min_z: f32,
    max_z: f32,
) -> f32 {
    let dx = if p[0] < min_x {
        min_x - p[0]
    } else if p[0] > max_x {
        p[0] - max_x
    } else {
        0.0
    };
    let dz = if p[1] < min_z {
        min_z - p[1]
    } else if p[1] > max_z {
        p[1] - max_z
    } else {
        0.0
    };
    dx.mul_add(dx, dz * dz)
}

#[inline(always)]
fn profile_edge_bezier_hull_cannot_beat(
    p: [f32; 2],
    edge: ProfileHermiteEdge,
    best_distance_sq: f32,
) -> bool {
    let controls = profile_edge_bezier_controls(edge);
    distance_sq_to_control_frame_bounds(p, controls) >= best_distance_sq
}

#[inline(always)]
fn profile_edge_bezier_controls(edge: ProfileHermiteEdge) -> [[f32; 2]; 4] {
    let width_1 = edge.width_at(1.0);
    let derivative_1 = edge.width_derivative(1.0);
    [
        [edge.c0, edge.z0],
        [edge.c0 + edge.c1 / 3.0, edge.z0 + edge.dz / 3.0],
        [width_1 - derivative_1 / 3.0, edge.z0 + edge.dz * (2.0 / 3.0)],
        [width_1, edge.z0 + edge.dz],
    ]
}

#[inline(always)]
fn distance_sq_to_control_frame_bounds(p: [f32; 2], controls: [[f32; 2]; 4]) -> f32 {
    let axis = [
        controls[3][0] - controls[0][0],
        controls[3][1] - controls[0][1],
    ];
    let axis_len_sq = axis[0].mul_add(axis[0], axis[1] * axis[1]);
    if axis_len_sq <= 1.0e-12 {
        return distance_sq_to_aabb(
            p,
            controls
                .iter()
                .fold(f32::INFINITY, |acc, point| acc.min(point[0])),
            controls
                .iter()
                .fold(f32::NEG_INFINITY, |acc, point| acc.max(point[0])),
            controls
                .iter()
                .fold(f32::INFINITY, |acc, point| acc.min(point[1])),
            controls
                .iter()
                .fold(f32::NEG_INFINITY, |acc, point| acc.max(point[1])),
        );
    }

    let tangent = axis;
    let normal = [-axis[1], axis[0]];
    let mut tangent_min = f32::INFINITY;
    let mut tangent_max = f32::NEG_INFINITY;
    let mut normal_min = f32::INFINITY;
    let mut normal_max = f32::NEG_INFINITY;
    for point in controls {
        let rel = [point[0] - controls[0][0], point[1] - controls[0][1]];
        let tangent_projection = rel[0].mul_add(tangent[0], rel[1] * tangent[1]);
        let normal_projection = rel[0].mul_add(normal[0], rel[1] * normal[1]);
        tangent_min = tangent_min.min(tangent_projection);
        tangent_max = tangent_max.max(tangent_projection);
        normal_min = normal_min.min(normal_projection);
        normal_max = normal_max.max(normal_projection);
    }

    let rel = [p[0] - controls[0][0], p[1] - controls[0][1]];
    let tangent_projection = rel[0].mul_add(tangent[0], rel[1] * tangent[1]);
    let normal_projection = rel[0].mul_add(normal[0], rel[1] * normal[1]);
    let dt = axis_interval_gap(tangent_projection, tangent_min, tangent_max);
    let dn = axis_interval_gap(normal_projection, normal_min, normal_max);
    dt.mul_add(dt, dn * dn) / axis_len_sq
}

#[inline(always)]
fn axis_interval_gap(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min - value
    } else if value > max {
        value - max
    } else {
        0.0
    }
}

#[inline(always)]
fn distance_to_profile_hermite_edge(
    p: [f32; 2],
    a: ProfileNodeSample,
    c: ProfileNodeSample,
    edge: ProfileHermiteEdge,
    stats_enabled: bool,
) -> ProfileEdgeDistance {
    if edge.dz.abs() <= 1.0e-8 {
        if stats_enabled {
            PROFILE2D_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
        return distance_to_segment(
            p,
            [a.half_width, a.z],
            [c.half_width, c.z],
        );
    }

    let mut best = ProfileEdgeDistance {
        distance_sq: f32::INFINITY,
        closest: [a.half_width, a.z],
    };
    let mut winning_seed = HermiteSeedKind::Endpoint;
    let mut refined_ts = [0.0_f32; 5];
    let mut refined_t_count = 0usize;
    for (candidate, seed) in [
        (0.0, HermiteSeedKind::Endpoint),
        (0.25, HermiteSeedKind::Quarter25),
        (0.5, HermiteSeedKind::Quarter50),
        (0.75, HermiteSeedKind::Quarter75),
        (1.0, HermiteSeedKind::Endpoint),
    ] {
        let (t, iterations) = refine_profile_hermite_closest_t(p, edge, candidate);
        if stats_enabled {
            PROFILE2D_HERMITE_SEED_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
            PROFILE2D_HERMITE_NEWTON_ITERATIONS
                .fetch_add(iterations as u64, Ordering::Relaxed);
            record_hermite_iteration_bucket(iterations);
            if t <= 1.0e-6 || t >= 1.0 - 1.0e-6 {
                PROFILE2D_HERMITE_CLAMPED_ENDPOINT_ATTEMPTS
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        if stats_enabled {
            let duplicate_t = refined_ts[..refined_t_count]
                .iter()
                .any(|previous| t.to_bits() == previous.to_bits());
            if duplicate_t {
                PROFILE2D_HERMITE_DUPLICATE_T_ATTEMPTS
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                PROFILE2D_HERMITE_DISTANCE_EVALUATIONS
                    .fetch_add(1, Ordering::Relaxed);
            }
            refined_ts[refined_t_count] = t;
            refined_t_count += 1;
        }
        let distance = edge.distance_at(p, t);
        if distance.distance_sq < best.distance_sq {
            best = distance;
            winning_seed = seed;
        }
    }
    if stats_enabled {
        PROFILE2D_HERMITE_WINS_TOTAL.fetch_add(1, Ordering::Relaxed);
        match winning_seed {
            HermiteSeedKind::Endpoint => {
                PROFILE2D_HERMITE_ENDPOINT_WINS.fetch_add(1, Ordering::Relaxed);
            }
            HermiteSeedKind::Quarter25 => {
                PROFILE2D_HERMITE_QUARTER_WINS.fetch_add(1, Ordering::Relaxed);
                PROFILE2D_HERMITE_QUARTER_25_WINS
                    .fetch_add(1, Ordering::Relaxed);
            }
            HermiteSeedKind::Quarter50 => {
                PROFILE2D_HERMITE_QUARTER_WINS.fetch_add(1, Ordering::Relaxed);
                PROFILE2D_HERMITE_QUARTER_50_WINS
                    .fetch_add(1, Ordering::Relaxed);
            }
            HermiteSeedKind::Quarter75 => {
                PROFILE2D_HERMITE_QUARTER_WINS.fetch_add(1, Ordering::Relaxed);
                PROFILE2D_HERMITE_QUARTER_75_WINS
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    best
}

fn shell_eval_stats_enabled() -> bool {
    SHELL_EVAL_STATS_ENABLED.load(Ordering::Relaxed)
}

#[inline(always)]
fn record_hermite_iteration_bucket(iterations: u32) {
    match iterations {
        0 | 1 => {
            PROFILE2D_HERMITE_ITERATION_1_ATTEMPTS
                .fetch_add(1, Ordering::Relaxed);
        }
        2 => {
            PROFILE2D_HERMITE_ITERATION_2_ATTEMPTS
                .fetch_add(1, Ordering::Relaxed);
        }
        3 => {
            PROFILE2D_HERMITE_ITERATION_3_ATTEMPTS
                .fetch_add(1, Ordering::Relaxed);
        }
        _ => {
            PROFILE2D_HERMITE_ITERATION_4_ATTEMPTS
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[inline(always)]
fn refine_profile_hermite_closest_t(
    p: [f32; 2],
    edge: ProfileHermiteEdge,
    mut t: f32,
) -> (f32, u32) {
    let mut iterations = 0;
    for _ in 0..4 {
        iterations += 1;
        let y = edge.width_at(t);
        let dy = edge.width_derivative(t);
        let ddy = edge.width_second_derivative(t);
        let z = edge.z_at(t);
        let py = y - p[0];
        let pz = z - p[1];
        let first = py.mul_add(dy, pz * edge.dz);
        let second = dy.mul_add(dy, py * ddy) + edge.dz * edge.dz;
        if second.abs() <= 1.0e-8 {
            break;
        }
        let next = (t - first / second).clamp(0.0, 1.0);
        if (next - t).abs() <= 1.0e-6 {
            t = next;
            break;
        }
        t = next;
    }
    (t, iterations)
}

#[cfg(test)]
fn profile_hermite_distance_derivative(
    p: [f32; 2],
    edge: ProfileHermiteEdge,
    t: f32,
) -> f32 {
    let y = edge.width_at(t);
    let dy = edge.width_derivative(t);
    let z = edge.z_at(t);
    let py = y - p[0];
    let pz = z - p[1];
    py.mul_add(dy, pz * edge.dz)
}

#[cfg(test)]
fn profile_hermite_distance_second_derivative(
    p: [f32; 2],
    edge: ProfileHermiteEdge,
    t: f32,
) -> f32 {
    let y = edge.width_at(t);
    let dy = edge.width_derivative(t);
    let ddy = edge.width_second_derivative(t);
    let py = y - p[0];
    dy.mul_add(dy, py * ddy) + edge.dz * edge.dz
}

#[cfg(test)]
fn refine_profile_hermite_bracketed_root(
    p: [f32; 2],
    edge: ProfileHermiteEdge,
    mut lo: f32,
    mut hi: f32,
    mut lo_derivative: f32,
    mut hi_derivative: f32,
) -> (f32, u32) {
    let mut t = 0.5 * (lo + hi);
    let mut iterations = 0;
    for _ in 0..6 {
        iterations += 1;
        let derivative = profile_hermite_distance_derivative(p, edge, t);
        if derivative.abs() <= 1.0e-7 || (hi - lo).abs() <= 1.0e-6 {
            break;
        }

        if derivative.signum() == lo_derivative.signum() {
            lo = t;
            lo_derivative = derivative;
        } else {
            hi = t;
            hi_derivative = derivative;
        }

        let second =
            profile_hermite_distance_second_derivative(p, edge, t);
        let newton = if second.abs() > 1.0e-8 {
            t - derivative / second
        } else {
            f32::NAN
        };
        t = if newton > lo && newton < hi && newton.is_finite() {
            newton
        } else {
            0.5 * (lo + hi)
        };
    }

    let _ = hi_derivative;
    (t.clamp(0.0, 1.0), iterations)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HermiteSeedKind {
    Endpoint,
    Quarter25,
    Quarter50,
    Quarter75,
}

#[derive(Clone, Copy, Debug)]
struct ProfileHermiteEdge {
    z0: f32,
    dz: f32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
}

impl ProfileHermiteEdge {
    #[inline(always)]
    fn new(
        a: ProfileNodeSample,
        c: ProfileNodeSample,
        a_slope: f32,
        c_slope: f32,
    ) -> Self {
        let dz = c.z - a.z;
        let m0 = a_slope * dz;
        let m1 = c_slope * dz;
        Self {
            z0: a.z,
            dz,
            c0: a.half_width,
            c1: m0,
            c2: -3.0 * a.half_width - 2.0 * m0 + 3.0 * c.half_width - m1,
            c3: 2.0 * a.half_width + m0 - 2.0 * c.half_width + m1,
        }
    }

    #[inline(always)]
    fn width_at(self, t: f32) -> f32 {
        ((self.c3 * t + self.c2) * t + self.c1) * t + self.c0
    }

    #[inline(always)]
    fn width_derivative(self, t: f32) -> f32 {
        (3.0 * self.c3 * t + 2.0 * self.c2) * t + self.c1
    }

    #[inline(always)]
    fn width_second_derivative(self, t: f32) -> f32 {
        6.0 * self.c3 * t + 2.0 * self.c2
    }

    #[inline(always)]
    fn z_at(self, t: f32) -> f32 {
        self.z0 + self.dz * t
    }

    #[inline(always)]
    fn distance_at(self, p: [f32; 2], t: f32) -> ProfileEdgeDistance {
        let width = self.width_at(t);
        let z = self.z_at(t);
        let dw = p[0] - width;
        let dz = p[1] - z;
        ProfileEdgeDistance {
            distance_sq: dw.mul_add(dw, dz * dz),
            closest: [width, z],
        }
    }
}

#[inline(always)]
fn profile_hermite_width_extrema(
    edge: ProfileHermiteEdge,
    min_x: &mut f32,
    max_x: &mut f32,
) {
    let a = 3.0 * edge.c3;
    let b = 2.0 * edge.c2;
    let c = edge.c1;

    if a.abs() <= 1.0e-8 {
        if b.abs() > 1.0e-8 {
            profile_hermite_include_width_extremum(-c / b, edge, min_x, max_x);
        }
        return;
    }

    let discriminant = b.mul_add(b, -4.0 * a * c);
    if discriminant < 0.0 {
        return;
    }

    let root = discriminant.sqrt();
    let scale = 0.5 / a;
    profile_hermite_include_width_extremum(
        (-b - root) * scale,
        edge,
        min_x,
        max_x,
    );
    profile_hermite_include_width_extremum(
        (-b + root) * scale,
        edge,
        min_x,
        max_x,
    );
}

#[inline(always)]
fn profile_hermite_include_width_extremum(
    t: f32,
    edge: ProfileHermiteEdge,
    min_x: &mut f32,
    max_x: &mut f32,
) {
    if t > 0.0 && t < 1.0 {
        let width = edge.width_at(t);
        *min_x = min_x.min(width);
        *max_x = max_x.max(width);
    }
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

#[cfg(test)]
mod tests {
    use crate::{
        shell::{ShellProfileNodeTopology, ShellProfileSectionTopology},
        types::Grad,
    };

    use super::*;

    #[test]
    fn profile2d_skips_distant_smooth_edges_before_hermite_refinement() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let sample = eval_shell_distance(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            0.50,
            0.95,
            0.95,
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(sample.distance.is_finite());
        assert_eq!(stats.profile2d_calls, 1);
        assert!(
            stats.profile2d_bezier_tests <= 1,
            "distant smooth spans should be rejected by conservative profile AABBs before Hermite refinement; stats={stats:?}",
        );
    }

    #[test]
    fn profile2d_tests_query_height_edge_before_other_smooth_edges() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let sample = eval_shell_distance(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            0.50,
            0.50,
            0.19,
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(sample.distance.is_finite());
        assert_eq!(stats.profile2d_calls, 1);
        assert!(
            stats.profile2d_bezier_tests <= 1,
            "query-height profile edge should establish the best distance before other smooth spans run refinement; stats={stats:?}",
        );
    }

    #[test]
    fn profile2d_stops_scanning_monotonic_edges_after_z_bounds_cannot_win() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let sample = eval_shell_distance(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            0.50,
            0.50,
            0.19,
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(sample.distance.is_finite());
        assert_eq!(stats.profile2d_calls, 1);
        assert!(
            stats.profile2d_segment_tests <= 3,
            "monotonic profile z-bounds should stop scanning distant spans once they cannot beat the current closest edge; stats={stats:?}",
        );
    }

    #[test]
    fn profile2d_skips_hermite_when_edge_aabb_can_only_tie_endpoint_best() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let sample = eval_shell_distance(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            0.50,
            0.0,
            -0.70,
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(sample.distance.is_finite());
        assert_eq!(stats.profile2d_calls, 1);
        assert_eq!(
            stats.profile2d_bezier_tests, 0,
            "an edge whose AABB only ties the precomputed endpoint distance should not run Hermite refinement; stats={stats:?}",
        );
    }

    #[test]
    fn profile2d_skips_hermite_when_bezier_hull_cannot_beat_best() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let a = ProfileNodeSample {
            half_width: 0.0,
            z: 0.0,
            continuity: ShellProfileNodeContinuity::Smooth,
        };
        let c = ProfileNodeSample {
            half_width: 1.0,
            z: 1.0,
            continuity: ShellProfileNodeContinuity::Smooth,
        };
        let best = ProfileEdgeDistance {
            distance_sq: 0.20,
            closest: [0.4, 0.4],
        };

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let edge = profile_edge_distance_if_relevant(
            [0.0, 1.0],
            a,
            c,
            1.0,
            1.0,
            best,
            true,
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert_eq!(edge.distance_sq, best.distance_sq);
        assert_eq!(
            stats.profile2d_hermite_seed_attempts, 0,
            "Bezier hull lower bound should skip Hermite refinement even when the edge AABB overlaps the query; stats={stats:?}",
        );
    }

    #[test]
    fn smooth_profile_edge_bounds_use_actual_hermite_extrema() {
        let a = ProfileNodeSample {
            half_width: 0.0,
            z: 0.0,
            continuity: ShellProfileNodeContinuity::Smooth,
        };
        let c = ProfileNodeSample {
            half_width: 0.0,
            z: 1.0,
            continuity: ShellProfileNodeContinuity::Smooth,
        };

        let (min_x, max_x) = profile_edge_x_bounds(a, c, 10.0, -10.0);

        assert_eq!(min_x, 0.0);
        assert!(
            (max_x - 2.5).abs() <= 1.0e-5,
            "Hermite bounds should use interior width extrema, not loose control handles; got {max_x}",
        );
    }

    #[test]
    fn profile2d_records_hermite_seed_attempts_and_winners() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let sample = eval_shell_distance(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            0.50,
            0.50,
            0.19,
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(sample.distance.is_finite());
        assert_eq!(
            stats.profile2d_hermite_seed_attempts,
            stats.profile2d_hermite_wins_total * 5,
            "Hermite refinement should avoid the redundant same-height seed for the ship profile hot path; stats={stats:?}",
        );
        assert_eq!(
            stats.profile2d_hermite_wins_total,
            stats.profile2d_hermite_endpoint_wins
                + stats.profile2d_hermite_quarter_wins
                + stats.profile2d_hermite_height_wins,
            "Hermite winner buckets should account for every selected smooth-edge minimum; stats={stats:?}",
        );
        assert_eq!(
            stats.profile2d_hermite_quarter_wins,
            stats.profile2d_hermite_quarter_25_wins
                + stats.profile2d_hermite_quarter_50_wins
                + stats.profile2d_hermite_quarter_75_wins,
            "Quarter-seed winner buckets should account for every quarter-seed minimum; stats={stats:?}",
        );
        assert_eq!(
            stats.profile2d_hermite_seed_attempts,
            stats.profile2d_hermite_iteration_1_attempts
                + stats.profile2d_hermite_iteration_2_attempts
                + stats.profile2d_hermite_iteration_3_attempts
                + stats.profile2d_hermite_iteration_4_attempts,
            "Hermite convergence buckets should account for every seed attempt; stats={stats:?}",
        );
        assert!(
            stats.profile2d_hermite_clamped_endpoint_attempts
                <= stats.profile2d_hermite_seed_attempts,
            "clamped Hermite attempts cannot exceed total attempts; stats={stats:?}",
        );
        assert!(
            stats.profile2d_hermite_duplicate_t_attempts
                <= stats.profile2d_hermite_seed_attempts,
            "duplicate Hermite roots cannot exceed total attempts; stats={stats:?}",
        );
        assert_eq!(
            stats.profile2d_hermite_seed_attempts,
            stats.profile2d_hermite_distance_evaluations
                + stats.profile2d_hermite_duplicate_t_attempts,
            "Hermite distance evaluations plus duplicate refined roots should account for every seed attempt; stats={stats:?}",
        );
    }

    #[test]
    fn profile2d_records_edge_shape_and_pruning_counters() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let sample = eval_shell_distance(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            0.50,
            0.50,
            0.19,
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(sample.distance.is_finite());
        assert!(stats.profile2d_edges_considered > 0, "stats={stats:?}");
        assert_eq!(
            stats.profile2d_edges_considered,
            stats.profile2d_linear_edges + stats.profile2d_smooth_edges,
            "edge type counters should account for every considered contour edge; stats={stats:?}",
        );
        assert!(
            stats.profile2d_edges_aabb_pruned <= stats.profile2d_edges_considered,
            "AABB-pruned edges cannot exceed considered edges; stats={stats:?}",
        );
        assert!(
            stats.profile2d_hermite_edges_refined <= stats.profile2d_smooth_edges,
            "Hermite refinements should only come from smooth edges; stats={stats:?}",
        );
        assert!(
            stats.profile2d_endpoint_best_kept <= stats.profile2d_calls,
            "endpoint-best calls cannot exceed profile calls; stats={stats:?}",
        );
    }

    #[test]
    fn ship_profile_hermite_distance_matches_without_height_seed() {
        let nodes = test_profile_samples();
        let mut slopes = [0.0_f32; SHELL_MAX_NODES_PER_CURVE];
        profile_node_slopes(&nodes, &mut slopes);

        for edge_index in 0..nodes.len() - 1 {
            let a = nodes[edge_index];
            let c = nodes[edge_index + 1];
            if a.continuity == ShellProfileNodeContinuity::Linear
                && c.continuity == ShellProfileNodeContinuity::Linear
            {
                continue;
            }
            let (a_slope, c_slope) =
                profile_effective_edge_slopes(a, c, slopes[edge_index], slopes[edge_index + 1]);
            let edge = ProfileHermiteEdge::new(a, c, a_slope, c_slope);

            for y_step in 0..=14 {
                for z_step in 0..=24 {
                    let p = [
                        y_step as f32 * 0.05,
                        -0.50 + z_step as f32 * 0.055,
                    ];
                    let full =
                        distance_to_profile_hermite_edge(p, a, c, edge, false);
                    let reduced =
                        distance_to_profile_hermite_edge_without_height_seed_for_test(
                            p, a, c, edge,
                        );
                    assert!(
                        (full.distance_sq - reduced.distance_sq).abs() <= 1.0e-6,
                        "reduced Hermite seeds changed distance on edge {edge_index} at {p:?}: full={full:?} reduced={reduced:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn ship_profile_quarter_hermite_seeds_are_semantically_observed() {
        let nodes = test_profile_samples();
        let mut slopes = [0.0_f32; SHELL_MAX_NODES_PER_CURVE];
        profile_node_slopes(&nodes, &mut slopes);
        let mut changed_without_025 = false;
        let mut changed_without_050 = false;
        let mut changed_without_075 = false;

        for edge_index in 0..nodes.len() - 1 {
            let a = nodes[edge_index];
            let c = nodes[edge_index + 1];
            if a.continuity == ShellProfileNodeContinuity::Linear
                && c.continuity == ShellProfileNodeContinuity::Linear
            {
                continue;
            }
            let (a_slope, c_slope) =
                profile_effective_edge_slopes(a, c, slopes[edge_index], slopes[edge_index + 1]);
            let edge = ProfileHermiteEdge::new(a, c, a_slope, c_slope);

            for y_step in 0..=14 {
                for z_step in 0..=24 {
                    let p = [
                        y_step as f32 * 0.05,
                        -0.50 + z_step as f32 * 0.055,
                    ];
                    let full =
                        distance_to_profile_hermite_edge(p, a, c, edge, false);
                    let without_025 =
                        distance_to_profile_hermite_edge_without_seed_for_test(
                            p, a, c, edge, 0.25,
                        );
                    let without_050 =
                        distance_to_profile_hermite_edge_without_seed_for_test(
                            p, a, c, edge, 0.5,
                        );
                    let without_075 =
                        distance_to_profile_hermite_edge_without_seed_for_test(
                            p, a, c, edge, 0.75,
                        );
                    changed_without_025 |=
                        (full.distance_sq - without_025.distance_sq).abs() > 1.0e-6;
                    changed_without_050 |=
                        (full.distance_sq - without_050.distance_sq).abs() > 1.0e-6;
                    changed_without_075 |=
                        (full.distance_sq - without_075.distance_sq).abs() > 1.0e-6;
                }
            }
        }

        assert!(
            changed_without_025,
            "dropping the 0.25 Hermite seed should change at least one sampled ship-profile distance"
        );
        assert!(
            changed_without_050,
            "dropping the 0.50 Hermite seed should change at least one sampled ship-profile distance"
        );
        assert!(
            changed_without_075,
            "dropping the 0.75 Hermite seed should change at least one sampled ship-profile distance"
        );
    }

    #[test]
    fn bracketed_hermite_solver_is_not_semantics_preserving_for_ship_profile() {
        let nodes = test_profile_samples();
        let mut slopes = [0.0_f32; SHELL_MAX_NODES_PER_CURVE];
        profile_node_slopes(&nodes, &mut slopes);
        let mut changed_any_distance = false;

        for edge_index in 0..nodes.len() - 1 {
            let a = nodes[edge_index];
            let c = nodes[edge_index + 1];
            if a.continuity == ShellProfileNodeContinuity::Linear
                && c.continuity == ShellProfileNodeContinuity::Linear
            {
                continue;
            }
            let (a_slope, c_slope) =
                profile_effective_edge_slopes(a, c, slopes[edge_index], slopes[edge_index + 1]);
            let edge = ProfileHermiteEdge::new(a, c, a_slope, c_slope);

            for y_step in 0..=18 {
                for z_step in 0..=32 {
                    let p = [
                        y_step as f32 * 0.045,
                        -0.65 + z_step as f32 * 0.05,
                    ];
                    let current =
                        distance_to_profile_hermite_edge(p, a, c, edge, false);
                    let bracketed =
                        distance_to_profile_hermite_edge_bracketed_for_test(p, a, c, edge);
                    let changed =
                        (current.distance_sq - bracketed.distance_sq).abs() > 1.0e-6;
                    if changed {
                        changed_any_distance = true;
                    }
                }
            }
        }

        assert!(
            changed_any_distance,
            "bracketed stationary-root solving should be treated as a semantic change unless the profile distance contract is intentionally updated"
        );
    }

    #[test]
    fn profile_contour_sign_uses_edge_crossings() {
        let nodes = test_profile_samples();
        let mut slopes = [0.0_f32; SHELL_MAX_NODES_PER_CURVE];
        profile_node_slopes(&nodes, &mut slopes);

        assert!(
            profile_contour_contains_point([0.24, 0.02], &nodes, &slopes),
            "point left of the crossed profile edge should be inside the half-section contour"
        );
        assert!(
            !profile_contour_contains_point([0.70, 0.02], &nodes, &slopes),
            "point right of the crossed profile edge should be outside the half-section contour"
        );
        assert!(
            !profile_contour_contains_point([0.24, 0.90], &nodes, &slopes),
            "point above the top contour should be outside"
        );
        for point in [
            [0.0, -0.40],
            [0.24, 0.02],
            [0.70, 0.02],
            [0.24, 0.90],
            [0.61, 0.72],
        ] {
            assert_eq!(
                profile_contour_contains_point(point, &nodes, &slopes),
                profile_contour_contains_point_monotonic(
                    point, &nodes, &slopes
                ),
                "monotonic contour fast path should match winding sign for {point:?}",
            );
        }
    }

    #[test]
    fn profile_shell_grad_uses_profile_normal_instead_of_full_central_difference()
     {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let grad = eval_shell_grad(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            Grad::new(0.50, 1.0, 0.0, 0.0),
            Grad::new(0.50, 0.0, 1.0, 0.0),
            Grad::new(0.19, 0.0, 0.0, 1.0),
        );
        let stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(grad.v.is_finite());
        assert!(grad.dy.is_finite() && grad.dz.is_finite());
        assert!(
            stats.profile2d_calls <= 3,
            "profile shell gradient should avoid full central differencing over y/z; stats={stats:?}"
        );
    }

    #[test]
    fn profile2d_splits_distance_and_gradient_call_sources() {
        let _lock = SHELL_EVAL_STATS_TEST_LOCK
            .lock()
            .expect("stats lock should not be poisoned");
        let topology = ShellTopology::ship_profile_shell_hull(
            [test_profile_section(0.0), test_profile_section(1.0)],
            0.0,
            OpenTopPolicy::Closed,
        );
        let mut scratch = ShellEvalScratch::default();

        set_shell_eval_stats_enabled(true);
        reset_shell_eval_stats();
        let distance = eval_shell_distance(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            0.50,
            0.50,
            0.19,
        );
        let distance_stats = shell_eval_stats();
        reset_shell_eval_stats();
        let grad = eval_shell_grad(
            &topology,
            ShellParamsView::empty(),
            &mut scratch,
            Grad::new(0.50, 1.0, 0.0, 0.0),
            Grad::new(0.50, 0.0, 1.0, 0.0),
            Grad::new(0.19, 0.0, 0.0, 1.0),
        );
        let grad_stats = shell_eval_stats();
        set_shell_eval_stats_enabled(false);

        assert!(distance.distance.is_finite());
        assert!(grad.v.is_finite());
        assert_eq!(
            distance_stats.profile2d_calls,
            distance_stats.profile2d_distance_calls
                + distance_stats.profile2d_gradient_calls,
            "distance call split should account for all profile calls; stats={distance_stats:?}",
        );
        assert_eq!(distance_stats.profile2d_distance_calls, 1);
        assert_eq!(distance_stats.profile2d_gradient_calls, 0);
        assert_eq!(
            grad_stats.profile2d_calls,
            grad_stats.profile2d_distance_calls
                + grad_stats.profile2d_gradient_calls,
            "gradient call split should account for all profile calls; stats={grad_stats:?}",
        );
        assert_eq!(grad_stats.profile2d_distance_calls, 0);
        assert!(
            grad_stats.profile2d_gradient_calls >= 1,
            "gradient eval should attribute profile-section calls to the gradient path; stats={grad_stats:?}",
        );
    }

    fn test_profile_section(station: f32) -> ShellProfileSectionTopology {
        let nodes = [
            ShellProfileNodeTopology::new(
                0.00,
                -0.40,
                ShellProfileNodeContinuity::Linear,
            ),
            ShellProfileNodeTopology::new(
                0.08,
                -0.30,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                0.24,
                -0.12,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                0.42,
                0.08,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                0.58,
                0.30,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                0.68,
                0.56,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                0.62,
                0.72,
                ShellProfileNodeContinuity::Linear,
            ),
        ];
        ShellProfileSectionTopology::station_curve(station, &nodes)
    }

    fn test_profile_samples() -> [ProfileNodeSample; 7] {
        [
            ProfileNodeSample {
                half_width: 0.00,
                z: -0.40,
                continuity: ShellProfileNodeContinuity::Linear,
            },
            ProfileNodeSample {
                half_width: 0.08,
                z: -0.30,
                continuity: ShellProfileNodeContinuity::Smooth,
            },
            ProfileNodeSample {
                half_width: 0.24,
                z: -0.12,
                continuity: ShellProfileNodeContinuity::Smooth,
            },
            ProfileNodeSample {
                half_width: 0.42,
                z: 0.08,
                continuity: ShellProfileNodeContinuity::Smooth,
            },
            ProfileNodeSample {
                half_width: 0.58,
                z: 0.30,
                continuity: ShellProfileNodeContinuity::Smooth,
            },
            ProfileNodeSample {
                half_width: 0.68,
                z: 0.56,
                continuity: ShellProfileNodeContinuity::Smooth,
            },
            ProfileNodeSample {
                half_width: 0.62,
                z: 0.72,
                continuity: ShellProfileNodeContinuity::Linear,
            },
        ]
    }

    fn distance_to_profile_hermite_edge_without_height_seed_for_test(
        p: [f32; 2],
        a: ProfileNodeSample,
        c: ProfileNodeSample,
        edge: ProfileHermiteEdge,
    ) -> ProfileEdgeDistance {
        if edge.dz.abs() <= 1.0e-8 {
            return distance_to_segment(
                p,
                [a.half_width, a.z],
                [c.half_width, c.z],
            );
        }

        let mut best = ProfileEdgeDistance {
            distance_sq: f32::INFINITY,
            closest: [a.half_width, a.z],
        };
        for candidate in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let (t, _iterations) =
                refine_profile_hermite_closest_t(p, edge, candidate);
            let distance = edge.distance_at(p, t);
            if distance.distance_sq < best.distance_sq {
                best = distance;
            }
        }
        best
    }

    fn distance_to_profile_hermite_edge_without_seed_for_test(
        p: [f32; 2],
        a: ProfileNodeSample,
        c: ProfileNodeSample,
        edge: ProfileHermiteEdge,
        skipped_seed: f32,
    ) -> ProfileEdgeDistance {
        if edge.dz.abs() <= 1.0e-8 {
            return distance_to_segment(
                p,
                [a.half_width, a.z],
                [c.half_width, c.z],
            );
        }

        let mut best = ProfileEdgeDistance {
            distance_sq: f32::INFINITY,
            closest: [a.half_width, a.z],
        };
        for candidate in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            if (candidate - skipped_seed).abs() <= f32::EPSILON {
                continue;
            }
            let (t, _iterations) =
                refine_profile_hermite_closest_t(p, edge, candidate);
            let distance = edge.distance_at(p, t);
            if distance.distance_sq < best.distance_sq {
                best = distance;
            }
        }
        best
    }

    fn distance_to_profile_hermite_edge_bracketed_for_test(
        p: [f32; 2],
        a: ProfileNodeSample,
        c: ProfileNodeSample,
        edge: ProfileHermiteEdge,
    ) -> ProfileEdgeDistance {
        if edge.dz.abs() <= 1.0e-8 {
            return distance_to_segment(
                p,
                [a.half_width, a.z],
                [c.half_width, c.z],
            );
        }

        let mut best = edge.distance_at(p, 0.0);
        let endpoint = edge.distance_at(p, 1.0);
        if endpoint.distance_sq < best.distance_sq {
            best = endpoint;
        }

        let samples = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let mut derivatives = [0.0_f32; 5];
        for (index, sample) in samples.iter().copied().enumerate() {
            derivatives[index] =
                profile_hermite_distance_derivative(p, edge, sample);
            if derivatives[index].abs() <= 1.0e-7 {
                let distance = edge.distance_at(p, sample);
                if distance.distance_sq < best.distance_sq {
                    best = distance;
                }
            }
        }

        for index in 0..samples.len() - 1 {
            let lo_derivative = derivatives[index];
            let hi_derivative = derivatives[index + 1];
            if lo_derivative == 0.0 || hi_derivative == 0.0 {
                continue;
            }
            if lo_derivative.signum() == hi_derivative.signum() {
                continue;
            }
            let (t, _iterations) = refine_profile_hermite_bracketed_root(
                p,
                edge,
                samples[index],
                samples[index + 1],
                lo_derivative,
                hi_derivative,
            );
            let distance = edge.distance_at(p, t);
            if distance.distance_sq < best.distance_sq {
                best = distance;
            }
        }

        best
    }
}
