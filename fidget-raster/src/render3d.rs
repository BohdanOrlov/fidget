//! 3D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    GeometryBuffer, GeometryPixel, LeafDebugBuffer, LeafDebugPixel,
    RenderConfig, RenderWorker, TileSizesRef, VoxelSize,
    config::{Tile, VoxelRenderConfig},
};
use fidget_core::{
    eval::Function,
    shape::{Shape, ShapeBulkEval, ShapeTracingEval, ShapeVars},
    shell::{
        ShellBounds, profile2d_outer_distance_batch_calls,
        reset_profile2d_outer_distance_batch_calls, reset_shell_eval_stats,
        set_shell_eval_stats_enabled, shell_eval_stats,
    },
    types::{Grad, Interval},
};

use nalgebra::{Matrix4, Point3, Vector2, Vector3};
use std::time::{Duration, Instant};

////////////////////////////////////////////////////////////////////////////////

struct Scratch {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,

    xg: Vec<Grad>,
    yg: Vec<Grad>,
    zg: Vec<Grad>,

    /// Depth of each column
    columns: Vec<usize>,

    /// Output pixel offsets that need gradient shading
    grad_pixels: Vec<usize>,
}

impl Scratch {
    fn new(tile_size: usize) -> Self {
        let size2 = tile_size.pow(2);
        let size3 = tile_size.pow(3);

        Self {
            x: vec![0.0; size3],
            y: vec![0.0; size3],
            z: vec![0.0; size3],

            xg: vec![Grad::from(0.0); size2],
            yg: vec![Grad::from(0.0); size2],
            zg: vec![Grad::from(0.0); size2],

            columns: vec![0; size2],
            grad_pixels: Vec::with_capacity(size2),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Worker<'a, F: Function> {
    tile_sizes: TileSizesRef<'a>,
    image_size: VoxelSize,
    model_from_image: Matrix4<f32>,

    /// Reusable workspace for evaluation, to minimize allocation
    scratch: Scratch,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,

    /// Output images for this specific tile
    out: GeometryBuffer,
    stats: VoxelRenderStats,
    shell_stats_enabled: bool,
}

/// Aggregated counters and timings from one 3D voxel render.
#[derive(Clone, Copy, Debug, Default)]
pub struct VoxelRenderStats {
    /// Aggregated worker tile time.
    pub total_tile_time: Duration,
    /// Time spent in interval evaluation.
    pub interval_eval_time: Duration,
    /// Time spent simplifying traced shapes.
    pub simplify_time: Duration,
    /// Time spent in scalar float-slice evaluation.
    pub float_eval_time: Duration,
    /// Time spent in gradient-slice evaluation.
    pub grad_eval_time: Duration,
    /// Number of interval evaluations.
    pub interval_eval_calls: u64,
    /// Number of shape simplification calls.
    pub simplify_calls: u64,
    /// Number of float-slice evaluation calls.
    pub float_eval_calls: u64,
    /// Number of gradient-slice evaluation calls.
    pub grad_eval_calls: u64,
    /// Number of scalar distance samples evaluated.
    pub float_eval_samples: u64,
    /// Number of gradient samples evaluated.
    pub grad_eval_samples: u64,
    /// Calls into native shell 2D profile evaluator.
    pub shell_hull_profile2d_calls: u64,
    /// Native shell 2D profile evaluator calls from distance sampling.
    pub shell_hull_profile2d_distance_calls: u64,
    /// Native shell 2D profile evaluator calls from gradient sampling.
    pub shell_hull_profile2d_gradient_calls: u64,
    /// Native shell 2D outer-profile calls from distance sampling.
    pub shell_hull_profile2d_outer_distance_calls: u64,
    /// Native shell 2D inner-profile calls from distance sampling.
    pub shell_hull_profile2d_inner_distance_calls: u64,
    /// Native shell 2D outer-profile calls from gradient sampling.
    pub shell_hull_profile2d_outer_gradient_calls: u64,
    /// Native shell 2D inner-profile calls from gradient sampling.
    pub shell_hull_profile2d_inner_gradient_calls: u64,
    /// Native shell station-segment lookup calls for profile-shell samples.
    pub shell_hull_profile2d_station_lookup_calls: u64,
    /// Four-lane profile packet station lookup attempts.
    pub shell_hull_profile2d_station_lookup_packet4_attempts: u64,
    /// Four-lane profile packet station lookup hits.
    pub shell_hull_profile2d_station_lookup_packet4_hits: u64,
    /// Four-lane profile packet station lookup misses.
    pub shell_hull_profile2d_station_lookup_packet4_misses: u64,
    /// Calls into the JIT float4 native shell distance helper.
    pub jit_shell_float4_helper_calls: u64,
    /// Lanes passed through the JIT float4 native shell distance helper.
    pub jit_shell_float4_helper_lanes: u64,
    /// JIT float4 helper calls that used the same-segment packet fast path.
    pub jit_shell_float4_packet_fast_path_hits: u64,
    /// JIT float4 helper calls that fell back to scalar lane evaluation.
    pub jit_shell_float4_scalar_fallbacks: u64,
    /// JIT float4 helper lanes evaluated by the scalar fallback.
    pub jit_shell_float4_scalar_fallback_lanes: u64,
    /// Proxy bytes moved by visible JIT helper spill/restore code.
    pub jit_shell_float4_spill_restore_bytes: u64,
    /// Float-eval batches that performed outer profile distance work.
    pub shell_hull_profile2d_outer_distance_batches: u64,
    /// Samples in batches that performed outer profile distance work.
    pub shell_hull_profile2d_outer_distance_batch_samples: u64,
    /// Largest outer profile distance call count observed in one float batch.
    pub shell_hull_profile2d_outer_distance_max_batch_calls: u64,
    /// Float-eval batches with outer distance calls and native-AABB-rejectable samples.
    pub shell_hull_profile2d_outer_distance_aabb_reject_batches: u64,
    /// Float-eval batches where every sample is outside native shell AABBs.
    pub shell_hull_profile2d_outer_distance_aabb_reject_full_batches: u64,
    /// Samples in outer-distance batches that are outside native shell AABBs.
    pub shell_hull_profile2d_outer_distance_aabb_reject_samples: u64,
    /// Native shell 2D profile boundary segment tests.
    pub shell_hull_profile2d_segment_tests: u64,
    /// Native shell 2D quadratic-Bezier edge tests.
    pub shell_hull_profile2d_bezier_tests: u64,
    /// Native shell 2D profile fallback count.
    pub shell_hull_profile2d_fallbacks: u64,
    /// Native shell 2D candidate contour edges considered.
    pub shell_hull_profile2d_edges_considered: u64,
    /// Native shell 2D candidate contour edges pruned by AABB tests.
    pub shell_hull_profile2d_edges_aabb_pruned: u64,
    /// Native shell 2D smooth contour edges pruned by Bezier control hulls.
    pub shell_hull_profile2d_edges_bezier_hull_pruned: u64,
    /// Native shell 2D concrete edge distance evaluations after pruning.
    pub shell_hull_profile2d_edge_distance_evaluations: u64,
    /// Native shell 2D candidate linear contour edges.
    pub shell_hull_profile2d_linear_edges: u64,
    /// Native shell 2D candidate smooth contour edges.
    pub shell_hull_profile2d_smooth_edges: u64,
    /// Native shell 2D profile calls where an endpoint remained closest.
    pub shell_hull_profile2d_endpoint_best_kept: u64,
    /// Native shell 2D smooth edges that reached Hermite refinement.
    pub shell_hull_profile2d_hermite_edges_refined: u64,
    /// Native shell 2D Hermite seed refinement attempts.
    pub shell_hull_profile2d_hermite_seed_attempts: u64,
    /// Native shell 2D Hermite Newton iterations.
    pub shell_hull_profile2d_hermite_newton_iterations: u64,
    /// Native shell 2D Hermite attempts that stopped after 1 iteration.
    pub shell_hull_profile2d_hermite_iteration_1_attempts: u64,
    /// Native shell 2D Hermite attempts that stopped after 2 iterations.
    pub shell_hull_profile2d_hermite_iteration_2_attempts: u64,
    /// Native shell 2D Hermite attempts that stopped after 3 iterations.
    pub shell_hull_profile2d_hermite_iteration_3_attempts: u64,
    /// Native shell 2D Hermite attempts that used all 4 iterations.
    pub shell_hull_profile2d_hermite_iteration_4_attempts: u64,
    /// Native shell 2D Hermite attempts clamped to an endpoint.
    pub shell_hull_profile2d_hermite_clamped_endpoint_attempts: u64,
    /// Native shell 2D Hermite attempts duplicating an earlier refined t.
    pub shell_hull_profile2d_hermite_duplicate_t_attempts: u64,
    /// Native shell 2D Hermite distance evaluations after deduping roots.
    pub shell_hull_profile2d_hermite_distance_evaluations: u64,
    /// Native shell 2D Hermite distance evaluations performed for refined seeds.
    pub shell_hull_profile2d_hermite_final_distance_evaluations: u64,
    /// Native shell 2D Hermite closest-point winner count.
    pub shell_hull_profile2d_hermite_wins_total: u64,
    /// Native shell 2D Hermite endpoint-seed wins.
    pub shell_hull_profile2d_hermite_endpoint_wins: u64,
    /// Native shell 2D Hermite quarter-seed wins.
    pub shell_hull_profile2d_hermite_quarter_wins: u64,
    /// Native shell 2D Hermite 0.25-seed wins.
    pub shell_hull_profile2d_hermite_quarter_25_wins: u64,
    /// Native shell 2D Hermite 0.50-seed wins.
    pub shell_hull_profile2d_hermite_quarter_50_wins: u64,
    /// Native shell 2D Hermite 0.75-seed wins.
    pub shell_hull_profile2d_hermite_quarter_75_wins: u64,
    /// Native shell 2D Hermite height-seed wins.
    pub shell_hull_profile2d_hermite_height_wins: u64,
    /// Profile-shell interval tiles tested before exact profile sampling.
    pub shell_profile_interval_tiles: u64,
    /// Profile-shell interval tiles rejected before exact profile sampling.
    pub shell_profile_interval_rejected_tiles: u64,
    /// Profile-shell interval tiles that still straddle the shell.
    pub shell_profile_interval_active_tiles: u64,
    /// Active profile-shell interval tiles narrowed to one station segment.
    pub shell_profile_interval_single_segment_tiles: u64,
    /// Active profile-shell interval tiles spanning multiple station segments.
    pub shell_profile_interval_multi_segment_tiles: u64,
    /// Native shell interval calls.
    pub shell_interval_calls: u64,
    /// Interval tiles rejected as completely outside.
    pub shell_interval_rejects: u64,
    /// Sum of active profile-shell segment counts across active interval tiles.
    pub shell_active_segment_sum: u64,
    /// Active profile-shell interval tiles contributing to the segment average.
    pub shell_active_segment_samples: u64,
    /// Native shell closest-point iterations.
    pub shell_closest_iterations: u64,
    /// Native shell gradient helper calls.
    pub shell_grad_helper_calls: u64,
    /// Native shell interval hot-loop allocation count.
    pub shell_interval_hot_loop_allocations: u64,
    /// Native shell float-slice hot-loop allocation count.
    pub shell_float_slice_hot_loop_allocations: u64,
    /// Native shell grad-slice hot-loop allocation count.
    pub shell_grad_slice_hot_loop_allocations: u64,
    /// Native shell hot-loop allocation count.
    pub shell_hot_loop_allocations: u64,
    /// Native shell allocation count exposed under the renderer stats contract.
    pub shell_allocations: u64,
}

impl VoxelRenderStats {
    fn merge(&mut self, other: VoxelRenderStats) {
        self.total_tile_time += other.total_tile_time;
        self.interval_eval_time += other.interval_eval_time;
        self.simplify_time += other.simplify_time;
        self.float_eval_time += other.float_eval_time;
        self.grad_eval_time += other.grad_eval_time;
        self.interval_eval_calls += other.interval_eval_calls;
        self.simplify_calls += other.simplify_calls;
        self.float_eval_calls += other.float_eval_calls;
        self.grad_eval_calls += other.grad_eval_calls;
        self.float_eval_samples += other.float_eval_samples;
        self.grad_eval_samples += other.grad_eval_samples;
        self.shell_hull_profile2d_calls += other.shell_hull_profile2d_calls;
        self.shell_hull_profile2d_distance_calls +=
            other.shell_hull_profile2d_distance_calls;
        self.shell_hull_profile2d_gradient_calls +=
            other.shell_hull_profile2d_gradient_calls;
        self.shell_hull_profile2d_outer_distance_calls +=
            other.shell_hull_profile2d_outer_distance_calls;
        self.shell_hull_profile2d_inner_distance_calls +=
            other.shell_hull_profile2d_inner_distance_calls;
        self.shell_hull_profile2d_outer_gradient_calls +=
            other.shell_hull_profile2d_outer_gradient_calls;
        self.shell_hull_profile2d_inner_gradient_calls +=
            other.shell_hull_profile2d_inner_gradient_calls;
        self.shell_hull_profile2d_station_lookup_calls +=
            other.shell_hull_profile2d_station_lookup_calls;
        self.shell_hull_profile2d_station_lookup_packet4_attempts +=
            other.shell_hull_profile2d_station_lookup_packet4_attempts;
        self.shell_hull_profile2d_station_lookup_packet4_hits +=
            other.shell_hull_profile2d_station_lookup_packet4_hits;
        self.shell_hull_profile2d_station_lookup_packet4_misses +=
            other.shell_hull_profile2d_station_lookup_packet4_misses;
        self.jit_shell_float4_helper_calls +=
            other.jit_shell_float4_helper_calls;
        self.jit_shell_float4_helper_lanes +=
            other.jit_shell_float4_helper_lanes;
        self.jit_shell_float4_packet_fast_path_hits +=
            other.jit_shell_float4_packet_fast_path_hits;
        self.jit_shell_float4_scalar_fallbacks +=
            other.jit_shell_float4_scalar_fallbacks;
        self.jit_shell_float4_scalar_fallback_lanes +=
            other.jit_shell_float4_scalar_fallback_lanes;
        self.jit_shell_float4_spill_restore_bytes +=
            other.jit_shell_float4_spill_restore_bytes;
        self.shell_hull_profile2d_outer_distance_batches +=
            other.shell_hull_profile2d_outer_distance_batches;
        self.shell_hull_profile2d_outer_distance_batch_samples +=
            other.shell_hull_profile2d_outer_distance_batch_samples;
        self.shell_hull_profile2d_outer_distance_max_batch_calls = self
            .shell_hull_profile2d_outer_distance_max_batch_calls
            .max(other.shell_hull_profile2d_outer_distance_max_batch_calls);
        self.shell_hull_profile2d_outer_distance_aabb_reject_batches +=
            other.shell_hull_profile2d_outer_distance_aabb_reject_batches;
        self.shell_hull_profile2d_outer_distance_aabb_reject_full_batches +=
            other.shell_hull_profile2d_outer_distance_aabb_reject_full_batches;
        self.shell_hull_profile2d_outer_distance_aabb_reject_samples +=
            other.shell_hull_profile2d_outer_distance_aabb_reject_samples;
        self.shell_hull_profile2d_segment_tests +=
            other.shell_hull_profile2d_segment_tests;
        self.shell_hull_profile2d_bezier_tests +=
            other.shell_hull_profile2d_bezier_tests;
        self.shell_hull_profile2d_fallbacks +=
            other.shell_hull_profile2d_fallbacks;
        self.shell_hull_profile2d_edges_considered +=
            other.shell_hull_profile2d_edges_considered;
        self.shell_hull_profile2d_edges_aabb_pruned +=
            other.shell_hull_profile2d_edges_aabb_pruned;
        self.shell_hull_profile2d_edges_bezier_hull_pruned +=
            other.shell_hull_profile2d_edges_bezier_hull_pruned;
        self.shell_hull_profile2d_edge_distance_evaluations +=
            other.shell_hull_profile2d_edge_distance_evaluations;
        self.shell_hull_profile2d_linear_edges +=
            other.shell_hull_profile2d_linear_edges;
        self.shell_hull_profile2d_smooth_edges +=
            other.shell_hull_profile2d_smooth_edges;
        self.shell_hull_profile2d_endpoint_best_kept +=
            other.shell_hull_profile2d_endpoint_best_kept;
        self.shell_hull_profile2d_hermite_edges_refined +=
            other.shell_hull_profile2d_hermite_edges_refined;
        self.shell_hull_profile2d_hermite_seed_attempts +=
            other.shell_hull_profile2d_hermite_seed_attempts;
        self.shell_hull_profile2d_hermite_newton_iterations +=
            other.shell_hull_profile2d_hermite_newton_iterations;
        self.shell_hull_profile2d_hermite_iteration_1_attempts +=
            other.shell_hull_profile2d_hermite_iteration_1_attempts;
        self.shell_hull_profile2d_hermite_iteration_2_attempts +=
            other.shell_hull_profile2d_hermite_iteration_2_attempts;
        self.shell_hull_profile2d_hermite_iteration_3_attempts +=
            other.shell_hull_profile2d_hermite_iteration_3_attempts;
        self.shell_hull_profile2d_hermite_iteration_4_attempts +=
            other.shell_hull_profile2d_hermite_iteration_4_attempts;
        self.shell_hull_profile2d_hermite_clamped_endpoint_attempts +=
            other.shell_hull_profile2d_hermite_clamped_endpoint_attempts;
        self.shell_hull_profile2d_hermite_duplicate_t_attempts +=
            other.shell_hull_profile2d_hermite_duplicate_t_attempts;
        self.shell_hull_profile2d_hermite_distance_evaluations +=
            other.shell_hull_profile2d_hermite_distance_evaluations;
        self.shell_hull_profile2d_hermite_final_distance_evaluations +=
            other.shell_hull_profile2d_hermite_final_distance_evaluations;
        self.shell_hull_profile2d_hermite_wins_total +=
            other.shell_hull_profile2d_hermite_wins_total;
        self.shell_hull_profile2d_hermite_endpoint_wins +=
            other.shell_hull_profile2d_hermite_endpoint_wins;
        self.shell_hull_profile2d_hermite_quarter_wins +=
            other.shell_hull_profile2d_hermite_quarter_wins;
        self.shell_hull_profile2d_hermite_quarter_25_wins +=
            other.shell_hull_profile2d_hermite_quarter_25_wins;
        self.shell_hull_profile2d_hermite_quarter_50_wins +=
            other.shell_hull_profile2d_hermite_quarter_50_wins;
        self.shell_hull_profile2d_hermite_quarter_75_wins +=
            other.shell_hull_profile2d_hermite_quarter_75_wins;
        self.shell_hull_profile2d_hermite_height_wins +=
            other.shell_hull_profile2d_hermite_height_wins;
        self.shell_profile_interval_tiles += other.shell_profile_interval_tiles;
        self.shell_profile_interval_rejected_tiles +=
            other.shell_profile_interval_rejected_tiles;
        self.shell_profile_interval_active_tiles +=
            other.shell_profile_interval_active_tiles;
        self.shell_profile_interval_single_segment_tiles +=
            other.shell_profile_interval_single_segment_tiles;
        self.shell_profile_interval_multi_segment_tiles +=
            other.shell_profile_interval_multi_segment_tiles;
        self.shell_interval_calls += other.shell_interval_calls;
        self.shell_interval_rejects += other.shell_interval_rejects;
        self.shell_active_segment_sum += other.shell_active_segment_sum;
        self.shell_active_segment_samples += other.shell_active_segment_samples;
        self.shell_closest_iterations += other.shell_closest_iterations;
        self.shell_grad_helper_calls += other.shell_grad_helper_calls;
        self.shell_interval_hot_loop_allocations +=
            other.shell_interval_hot_loop_allocations;
        self.shell_float_slice_hot_loop_allocations +=
            other.shell_float_slice_hot_loop_allocations;
        self.shell_grad_slice_hot_loop_allocations +=
            other.shell_grad_slice_hot_loop_allocations;
        self.shell_hot_loop_allocations += other.shell_hot_loop_allocations;
        self.shell_allocations += other.shell_allocations;
    }

    /// Average active profile-shell segments per active interval tile.
    pub fn shell_active_segment_avg(&self) -> f64 {
        if self.shell_active_segment_samples == 0 {
            0.0
        } else {
            self.shell_active_segment_sum as f64
                / self.shell_active_segment_samples as f64
        }
    }

    /// Share of JIT float4 helper calls that stayed in one profile segment.
    pub fn jit_shell_float4_same_segment_rate(&self) -> f64 {
        if self.jit_shell_float4_helper_calls == 0 {
            0.0
        } else {
            self.jit_shell_float4_packet_fast_path_hits as f64
                / self.jit_shell_float4_helper_calls as f64
        }
    }

    /// Average lane batch size for JIT float4 helper calls.
    pub fn jit_shell_float4_avg_helper_batch(&self) -> f64 {
        if self.jit_shell_float4_helper_calls == 0 {
            0.0
        } else {
            self.jit_shell_float4_helper_lanes as f64
                / self.jit_shell_float4_helper_calls as f64
        }
    }

    fn record_profile2d_outer_distance_batch(
        &mut self,
        calls: u64,
        sample_count: usize,
    ) {
        if calls == 0 {
            return;
        }
        self.shell_hull_profile2d_outer_distance_batches += 1;
        self.shell_hull_profile2d_outer_distance_batch_samples +=
            sample_count as u64;
        self.shell_hull_profile2d_outer_distance_max_batch_calls = self
            .shell_hull_profile2d_outer_distance_max_batch_calls
            .max(calls);
    }

    fn record_profile2d_outer_distance_aabb_rejection_potential(
        &mut self,
        calls: u64,
        sample_count: usize,
        rejectable_samples: usize,
    ) {
        if calls == 0 || rejectable_samples == 0 {
            return;
        }
        self.shell_hull_profile2d_outer_distance_aabb_reject_batches += 1;
        self.shell_hull_profile2d_outer_distance_aabb_reject_samples +=
            rejectable_samples as u64;
        if rejectable_samples == sample_count {
            self.shell_hull_profile2d_outer_distance_aabb_reject_full_batches +=
                1;
        }
    }
}

#[inline]
fn native_aabb_rejectable_sample_count(
    bounds: &ShellBounds,
    model_from_image: &Matrix4<f32>,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
) -> usize {
    xs.iter()
        .zip(ys)
        .zip(zs)
        .filter(|((x, y), z)| {
            let point =
                model_from_image.transform_point(&Point3::new(**x, **y, **z));
            point.x < bounds.min_x
                || point.x > bounds.max_x
                || point.y < bounds.min_y
                || point.y > bounds.max_y
                || point.z < bounds.min_z
                || point.z > bounds.max_z
        })
        .count()
}

struct TileRenderOutput {
    image: GeometryBuffer,
    stats: VoxelRenderStats,
}

#[derive(Clone, Copy)]
struct LeafSampleRequest {
    tile: Tile<3>,
    tile_size: usize,
    sampled_depth: f32,
}

struct DebugWorker<'a, F: Function> {
    tile_sizes: TileSizesRef<'a>,
    image_size: VoxelSize,

    /// Reusable workspace for evaluation, to minimize allocation
    scratch: Scratch,

    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,

    /// Output images for this specific tile
    out: LeafDebugBuffer,
    stats: VoxelRenderStats,
}

struct TileDebugRenderOutput {
    image: LeafDebugBuffer,
    stats: VoxelRenderStats,
}

impl<'a, F: Function, T> RenderWorker<'a, F, T> for Worker<'a, F> {
    type Config = VoxelRenderConfig<'a>;
    type Output = TileRenderOutput;

    fn new(cfg: &'a Self::Config) -> Self {
        let tile_sizes = cfg.tile_sizes();
        let buf_size = tile_sizes.last();
        let scratch = Scratch::new(buf_size);
        Worker {
            scratch,
            out: Default::default(),
            tile_sizes,
            image_size: cfg.image_size,
            model_from_image: cfg.mat(),

            eval_float_slice: Default::default(),
            eval_interval: Default::default(),
            eval_grad_slice: Default::default(),

            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
            stats: Default::default(),
            shell_stats_enabled: std::env::var_os("FIDGET_RENDER3D_STATS")
                .is_some(),
        }
    }

    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        tile: super::config::Tile<2>,
    ) -> Self::Output {
        self.stats = VoxelRenderStats::default();
        let tile_start = Instant::now();

        // Prepare local tile data to fill out
        let root_tile_size = self.tile_sizes[0];
        self.out = GeometryBuffer::new(VoxelSize::from(root_tile_size as u32));
        for k in (0..self.image_size[2].div_ceil(root_tile_size as u32)).rev() {
            let tile = Tile::new(Point3::new(
                tile.corner.x,
                tile.corner.y,
                k as usize * root_tile_size,
            ));
            if !self.render_tile_recurse(shape, vars, 0, tile) {
                break;
            }
        }
        self.stats.total_tile_time += tile_start.elapsed();
        TileRenderOutput {
            image: std::mem::take(&mut self.out),
            stats: self.stats,
        }
    }
}

impl<'a, F: Function, T> RenderWorker<'a, F, T> for DebugWorker<'a, F> {
    type Config = VoxelRenderConfig<'a>;
    type Output = TileDebugRenderOutput;

    fn new(cfg: &'a Self::Config) -> Self {
        let tile_sizes = cfg.tile_sizes();
        let buf_size = tile_sizes.last();
        let scratch = Scratch::new(buf_size);
        DebugWorker {
            scratch,
            out: Default::default(),
            tile_sizes,
            image_size: cfg.image_size,

            eval_interval: Default::default(),
            eval_grad_slice: Default::default(),

            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
            stats: Default::default(),
        }
    }

    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        tile: super::config::Tile<2>,
    ) -> Self::Output {
        self.stats = VoxelRenderStats::default();
        let tile_start = Instant::now();

        // Prepare local tile data to fill out
        let root_tile_size = self.tile_sizes[0];
        self.out = LeafDebugBuffer::new(VoxelSize::from(root_tile_size as u32));
        for k in (0..self.image_size[2].div_ceil(root_tile_size as u32)).rev() {
            let tile = Tile::new(Point3::new(
                tile.corner.x,
                tile.corner.y,
                k as usize * root_tile_size,
            ));
            if !self.render_tile_recurse(shape, vars, 0, tile) {
                break;
            }
        }
        self.stats.total_tile_time += tile_start.elapsed();
        TileDebugRenderOutput {
            image: std::mem::take(&mut self.out),
            stats: self.stats,
        }
    }
}

impl<F: Function> Worker<'_, F> {
    /// Returns the data offset of a row within a subtile
    pub(crate) fn tile_row_offset(&self, tile: Tile<3>, row: usize) -> usize {
        self.tile_sizes.pixel_offset(tile.add(Vector2::new(0, row)))
    }

    /// Render a single tile
    ///
    /// Returns `true` if we should keep rendering, `false` otherwise
    fn render_tile_recurse<T>(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        depth: usize,
        tile: Tile<3>,
    ) -> bool {
        // Early exit if every single pixel is filled
        let tile_size = self.tile_sizes[depth];
        let fill_z = (tile.corner[2] + tile_size + 1) as f32;
        if (0..tile_size).all(|y| {
            let i = self.tile_row_offset(tile, y);
            (0..tile_size).all(|x| self.out[i + x].depth >= fill_z)
        }) {
            return false;
        }

        let base = Point3::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(base.z, base.z + tile_size as f32);

        let start = Instant::now();
        let (i, trace) = self
            .eval_interval
            .eval_v(shape.i_tape(&mut self.tape_storage), x, y, z, vars)
            .unwrap();
        self.stats.interval_eval_time += start.elapsed();
        self.stats.interval_eval_calls += 1;

        // Return early if this tile is completely empty or full, returning
        // `data_interval` to scratch memory for reuse.
        if i.upper() < 0.0 {
            for y in 0..tile_size {
                let i = self.tile_row_offset(tile, y);
                for x in 0..tile_size {
                    self.out[i + x].depth = self.out[i + x].depth.max(fill_z);
                }
            }
            return false; // completely full, stop rendering
        } else if i.lower() > 0.0 {
            self.stats.shell_interval_rejects += 1;
            return true; // complete empty, keep going
        }

        // Calculate a simplified tape based on the trace
        let sub_tape = if let Some(trace) = trace.as_ref() {
            let start = Instant::now();
            let out = shape.simplify(
                trace,
                &mut self.workspace,
                &mut self.shape_storage,
                &mut self.tape_storage,
            );
            self.stats.simplify_time += start.elapsed();
            self.stats.simplify_calls += 1;
            out
        } else {
            shape
        };

        // Recurse!
        if let Some(next_tile_size) = self.tile_sizes.get(depth + 1) {
            let n = tile_size / next_tile_size;

            for j in 0..n {
                for i in 0..n {
                    for k in (0..n).rev() {
                        self.render_tile_recurse(
                            sub_tape,
                            vars,
                            depth + 1,
                            Tile::new(
                                tile.corner
                                    + Vector3::new(i, j, k) * next_tile_size,
                            ),
                        );
                    }
                }
            }
        } else {
            self.render_tile_pixels(sub_tape, vars, tile_size, tile);
        };
        // TODO recycle something here?
        true // keep going
    }

    fn render_tile_pixels<T>(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        tile_size: usize,
        tile: Tile<3>,
    ) {
        // Prepare for pixel-by-pixel evaluation
        assert!(self.scratch.x.len() >= tile_size.pow(3));
        assert!(self.scratch.y.len() >= tile_size.pow(3));
        assert!(self.scratch.z.len() >= tile_size.pow(3));
        self.scratch.columns.clear();
        self.scratch.grad_pixels.clear();
        for xy in 0..tile_size.pow(2) {
            let i = xy % tile_size;
            let j = xy / tile_size;

            let o = self.tile_sizes.pixel_offset(tile.add(Vector2::new(i, j)));

            // Skip pixels which are behind the image
            let zmax = (tile.corner[2] + tile_size) as f32;
            if self.out[o].depth >= zmax {
                continue;
            }

            self.scratch.columns.push(xy);
        }

        const LEAF_DEPTH_BATCH: usize = 1;

        let mut grad = 0;
        let mut upper_k = tile_size;
        while upper_k > 0 {
            let size = self.scratch.columns.len();
            if size == 0 {
                break;
            }
            let lower_k = upper_k.saturating_sub(LEAF_DEPTH_BATCH);
            let depth_count = upper_k - lower_k;

            let mut sample_count = 0;
            for column_index in 0..size {
                let xy = self.scratch.columns[column_index];
                let i = xy % tile_size;
                let j = xy / tile_size;

                for k in (lower_k..upper_k).rev() {
                    // SAFETY:
                    // Index cannot exceed tile_size**3, which is (a) the size
                    // that we allocated in `Scratch::new` and (b) checked by
                    // assertions above.
                    //
                    // Using unsafe indexing here is a roughly 2.5% speedup,
                    // since this is the hottest loop.
                    unsafe {
                        *self.scratch.x.get_unchecked_mut(sample_count) =
                            (tile.corner[0] + i) as f32;
                        *self.scratch.y.get_unchecked_mut(sample_count) =
                            (tile.corner[1] + j) as f32;
                        *self.scratch.z.get_unchecked_mut(sample_count) =
                            (tile.corner[2] + k) as f32;
                    }
                    sample_count += 1;
                }
            }

            let aabb_rejectable_samples = if self.shell_stats_enabled {
                shape
                    .native_render_metadata()
                    .and_then(|metadata| metadata.global_aabb)
                    .map(|bounds| {
                        native_aabb_rejectable_sample_count(
                            &bounds,
                            &self.model_from_image,
                            &self.scratch.x[..sample_count],
                            &self.scratch.y[..sample_count],
                            &self.scratch.z[..sample_count],
                        )
                    })
                    .unwrap_or(0)
            } else {
                0
            };

            if self.shell_stats_enabled {
                reset_profile2d_outer_distance_batch_calls();
            }
            let start = Instant::now();
            let out = self
                .eval_float_slice
                .eval_v(
                    shape.f_tape(&mut self.tape_storage),
                    &self.scratch.x[..sample_count],
                    &self.scratch.y[..sample_count],
                    &self.scratch.z[..sample_count],
                    vars,
                )
                .unwrap();
            self.stats.float_eval_time += start.elapsed();
            self.stats.float_eval_calls += 1;
            self.stats.float_eval_samples += sample_count as u64;
            if self.shell_stats_enabled {
                let outer_distance_calls =
                    profile2d_outer_distance_batch_calls();
                self.stats.record_profile2d_outer_distance_batch(
                    outer_distance_calls,
                    sample_count,
                );
                self.stats
                    .record_profile2d_outer_distance_aabb_rejection_potential(
                        outer_distance_calls,
                        sample_count,
                        aabb_rejectable_samples,
                    );
            }

            let mut write = 0;
            for column_index in 0..size {
                let xy = self.scratch.columns[column_index];
                let base = column_index * depth_count;
                let hit = out[base..base + depth_count]
                    .iter()
                    .position(|distance| *distance < 0.0);
                if let Some(offset) = hit {
                    let i = xy % tile_size;
                    let j = xy / tile_size;
                    let k = upper_k - 1 - offset;

                    // Set the depth of the pixel
                    let o = self
                        .tile_sizes
                        .pixel_offset(tile.add(Vector2::new(i, j)));
                    let z = (tile.corner[2] + k + 1) as f32;
                    assert!(self.out[o].depth < z);
                    self.out[o].depth = z;

                    // Prepare to do gradient rendering of this point.
                    // We step one voxel above the surface to reduce
                    // glitchiness on edges and corners, where rendering
                    // inside the surface could pick the wrong normal.
                    self.scratch.xg[grad] =
                        Grad::new((tile.corner[0] + i) as f32, 1.0, 0.0, 0.0);
                    self.scratch.yg[grad] =
                        Grad::new((tile.corner[1] + j) as f32, 0.0, 1.0, 0.0);
                    self.scratch.zg[grad] =
                        Grad::new((tile.corner[2] + k) as f32, 0.0, 0.0, 1.0);
                    self.scratch.grad_pixels.push(o);
                    grad += 1;
                } else {
                    self.scratch.columns[write] = xy;
                    write += 1;
                }
            }
            self.scratch.columns.truncate(write);
            upper_k = lower_k;
        }

        if grad > 0 {
            let start = Instant::now();
            let out = self
                .eval_grad_slice
                .eval_v(
                    shape.g_tape(&mut self.tape_storage),
                    &self.scratch.xg[..grad],
                    &self.scratch.yg[..grad],
                    &self.scratch.zg[..grad],
                    vars,
                )
                .unwrap();
            self.stats.grad_eval_time += start.elapsed();
            self.stats.grad_eval_calls += 1;
            self.stats.grad_eval_samples += grad as u64;

            for (index, o) in
                self.scratch.grad_pixels[0..grad].iter().enumerate()
            {
                let g = out[index];
                self.out[*o].normal = [g.dx, g.dy, g.dz];
            }
        }
    }
}

impl<F: Function> DebugWorker<'_, F> {
    /// Returns the data offset of a row within a subtile
    pub(crate) fn tile_row_offset(&self, tile: Tile<3>, row: usize) -> usize {
        self.tile_sizes.pixel_offset(tile.add(Vector2::new(0, row)))
    }

    /// Render a single tile
    ///
    /// Returns `true` if we should keep rendering, `false` otherwise
    fn render_tile_recurse<T>(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        depth: usize,
        tile: Tile<3>,
    ) -> bool {
        // Early exit if every single pixel is filled
        let tile_size = self.tile_sizes[depth];
        let fill_z = (tile.corner[2] + tile_size + 1) as f32;
        if (0..tile_size).all(|y| {
            let i = self.tile_row_offset(tile, y);
            (0..tile_size).all(|x| self.out[i + x].depth >= fill_z)
        }) {
            return false;
        }

        let base = Point3::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(base.z, base.z + tile_size as f32);

        let start = Instant::now();
        let (i, trace) = self
            .eval_interval
            .eval_v(shape.i_tape(&mut self.tape_storage), x, y, z, vars)
            .unwrap();
        self.stats.interval_eval_time += start.elapsed();
        self.stats.interval_eval_calls += 1;

        // Return early if this tile is completely empty or full.
        if i.upper() < 0.0 {
            // This tile is entirely inside, so we cull it without writing a
            // representative sample. In leaf-debug mode, this preserves
            // "no value" pixels (depth=0) for interval-pruned regions.
            return false;
        } else if i.lower() > 0.0 {
            self.stats.shell_interval_rejects += 1;
            return true; // completely empty, keep going
        }

        // Calculate a simplified tape based on the trace
        let sub_tape = if let Some(trace) = trace.as_ref() {
            let start = Instant::now();
            let out = shape.simplify(
                trace,
                &mut self.workspace,
                &mut self.shape_storage,
                &mut self.tape_storage,
            );
            self.stats.simplify_time += start.elapsed();
            self.stats.simplify_calls += 1;
            out
        } else {
            shape
        };

        // Recurse!
        if let Some(next_tile_size) = self.tile_sizes.get(depth + 1) {
            if self.tile_sizes.get(depth + 2).is_none() {
                self.render_terminal_leaf_children(
                    sub_tape,
                    vars,
                    tile_size,
                    next_tile_size,
                    tile,
                );
            } else {
                let n = tile_size / next_tile_size;

                for j in 0..n {
                    for i in 0..n {
                        for k in (0..n).rev() {
                            self.render_tile_recurse(
                                sub_tape,
                                vars,
                                depth + 1,
                                Tile::new(
                                    tile.corner
                                        + Vector3::new(i, j, k)
                                            * next_tile_size,
                                ),
                            );
                        }
                    }
                }
            }
        } else {
            let req = LeafSampleRequest {
                tile,
                tile_size,
                sampled_depth: tile.corner[2] as f32 + tile_size as f32 * 0.5,
            };
            self.sample_leaf_batch(sub_tape, vars, std::slice::from_ref(&req));
        };
        true // keep going
    }

    fn render_terminal_leaf_children<T>(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        tile_size: usize,
        child_tile_size: usize,
        tile: Tile<3>,
    ) {
        let n = tile_size / child_tile_size;
        let mut leaf_samples = Vec::with_capacity(n * n * n);

        let shape_ptr = shape as *mut RenderHandle<F, T>;
        for j in 0..n {
            for i in 0..n {
                for k in (0..n).rev() {
                    let child_tile = Tile::new(
                        tile.corner + Vector3::new(i, j, k) * child_tile_size,
                    );
                    let fill_z =
                        (child_tile.corner[2] + child_tile_size + 1) as f32;
                    if (0..child_tile_size).all(|y| {
                        let row = self.tile_row_offset(child_tile, y);
                        (0..child_tile_size)
                            .all(|x| self.out[row + x].depth >= fill_z)
                    }) {
                        continue;
                    }

                    let base = Point3::from(child_tile.corner).cast::<f32>();
                    let x =
                        Interval::new(base.x, base.x + child_tile_size as f32);
                    let y =
                        Interval::new(base.y, base.y + child_tile_size as f32);
                    let z =
                        Interval::new(base.z, base.z + child_tile_size as f32);

                    let start = Instant::now();
                    let (i, trace) = self
                        .eval_interval
                        .eval_v(
                            shape.i_tape(&mut self.tape_storage),
                            x,
                            y,
                            z,
                            vars,
                        )
                        .unwrap();
                    self.stats.interval_eval_time += start.elapsed();
                    self.stats.interval_eval_calls += 1;

                    // Preserve the same empty/full culling behavior as the
                    // recursive leaf path: don't write culled tiles.
                    if i.lower() > 0.0 {
                        self.stats.shell_interval_rejects += 1;
                        continue;
                    }
                    if i.upper() < 0.0 {
                        continue;
                    }

                    let req = LeafSampleRequest {
                        tile: child_tile,
                        tile_size: child_tile_size,
                        sampled_depth: child_tile.corner[2] as f32
                            + child_tile_size as f32 * 0.5,
                    };

                    if let Some(trace) = trace.as_ref() {
                        let start = Instant::now();
                        let leaf_shape = shape.simplify(
                            trace,
                            &mut self.workspace,
                            &mut self.shape_storage,
                            &mut self.tape_storage,
                        );
                        self.stats.simplify_time += start.elapsed();
                        self.stats.simplify_calls += 1;

                        if std::ptr::eq(leaf_shape as *mut _, shape_ptr) {
                            leaf_samples.push(req);
                        } else {
                            self.sample_leaf_batch(
                                leaf_shape,
                                vars,
                                std::slice::from_ref(&req),
                            );
                        }
                    } else {
                        leaf_samples.push(req);
                    }
                }
            }
        }
        self.sample_leaf_batch(shape, vars, &leaf_samples);
    }

    fn sample_leaf_batch<T>(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        samples: &[LeafSampleRequest],
    ) {
        if samples.is_empty() {
            return;
        }

        for (index, sample) in samples.iter().enumerate() {
            let center = Point3::new(
                sample.tile.corner[0] as f32 + sample.tile_size as f32 * 0.5,
                sample.tile.corner[1] as f32 + sample.tile_size as f32 * 0.5,
                sample.tile.corner[2] as f32 + sample.tile_size as f32 * 0.5,
            );
            self.scratch.xg[index] = Grad::new(center.x, 1.0, 0.0, 0.0);
            self.scratch.yg[index] = Grad::new(center.y, 0.0, 1.0, 0.0);
            self.scratch.zg[index] = Grad::new(center.z, 0.0, 0.0, 1.0);
        }
        let start = Instant::now();
        let out = self
            .eval_grad_slice
            .eval_v(
                shape.g_tape(&mut self.tape_storage),
                &self.scratch.xg[..samples.len()],
                &self.scratch.yg[..samples.len()],
                &self.scratch.zg[..samples.len()],
                vars,
            )
            .unwrap()
            .to_vec();
        self.stats.grad_eval_time += start.elapsed();
        self.stats.grad_eval_calls += 1;
        self.stats.grad_eval_samples += samples.len() as u64;

        for (index, sample) in samples.iter().enumerate() {
            let g = out[index];
            let distance = g.v;

            // Treat positive signed distance as "no value" for the leaf-debug
            // projection so outside samples remain transparent.
            if !distance.is_finite() || distance > 0.0 {
                continue;
            }

            let normal = [g.dx, g.dy, g.dz];
            let tile = sample.tile;
            let tile_size = sample.tile_size;
            let sampled_depth = sample.sampled_depth;
            for j in 0..tile_size {
                let row_offset = self.tile_row_offset(tile, j);
                for i in 0..tile_size {
                    let o = row_offset + i;
                    if sampled_depth > self.out[o].depth {
                        self.out[o] = LeafDebugPixel {
                            depth: sampled_depth,
                            distance,
                            normal,
                        };
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Renders the given tape into a 3D image according to the provided
/// configuration.
///
/// The tape provides the shape; the configuration supplies resolution,
/// transforms, etc.
///
/// This function is parameterized by shape type, which determines how we
/// perform evaluation.
///
/// Returns a [`GeometryBuffer`] of pixels, or `None` if rendering was cancelled
/// (using the [`VoxelRenderConfig::cancel`] token)
pub fn render<F: Function>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &VoxelRenderConfig,
) -> Option<GeometryBuffer> {
    render_with_stats(shape, vars, config).map(|(image, _stats)| image)
}

/// Renders the given tape into a 3D image and returns render stats.
///
/// This is the same traversal as [`render`], but returns the internal counters
/// used for benchmark reporting.
pub fn render_with_stats<F: Function>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &VoxelRenderConfig,
) -> Option<(GeometryBuffer, VoxelRenderStats)> {
    let shape = shape.with_transform(config.mat());

    let shell_stats_enabled =
        std::env::var_os("FIDGET_RENDER3D_STATS").is_some();
    set_shell_eval_stats_enabled(shell_stats_enabled);
    reset_shell_eval_stats();
    let tiles = super::render_tiles::<F, Worker<F>, _>(shape, vars, config)?;
    let tile_sizes = config.tile_sizes();

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image = GeometryBuffer::new(config.image_size);
    let mut stats = VoxelRenderStats::default();
    let merge_start = Instant::now();
    for (tile, out) in tiles {
        stats.merge(out.stats);
        let out = out.image;
        let mut index = 0;
        for j in 0..tile_sizes[0] {
            let y = j + tile.corner.y;
            for i in 0..tile_sizes[0] {
                let x = i + tile.corner.x;
                if x < width && y < height {
                    let o = y * width + x;
                    if out[index].depth >= image[o].depth {
                        // Clamp voxels to the image depth
                        let d = (config.image_size.depth() - 1) as f32;
                        if out[index].depth >= d {
                            image[o] = GeometryPixel {
                                depth: d + 1.0,
                                normal: [0.0, 0.0, 1.0],
                            };
                        } else {
                            image[o] = out[index];
                        }
                    }
                }
                index += 1;
            }
        }
    }
    let merge_time = merge_start.elapsed();
    let shell_stats = shell_eval_stats();
    stats.shell_hull_profile2d_calls = shell_stats.profile2d_calls;
    stats.shell_hull_profile2d_distance_calls =
        shell_stats.profile2d_distance_calls;
    stats.shell_hull_profile2d_gradient_calls =
        shell_stats.profile2d_gradient_calls;
    stats.shell_hull_profile2d_outer_distance_calls =
        shell_stats.profile2d_outer_distance_calls;
    stats.shell_hull_profile2d_inner_distance_calls =
        shell_stats.profile2d_inner_distance_calls;
    stats.shell_hull_profile2d_outer_gradient_calls =
        shell_stats.profile2d_outer_gradient_calls;
    stats.shell_hull_profile2d_inner_gradient_calls =
        shell_stats.profile2d_inner_gradient_calls;
    stats.shell_hull_profile2d_station_lookup_calls =
        shell_stats.profile2d_station_lookup_calls;
    stats.shell_hull_profile2d_station_lookup_packet4_attempts =
        shell_stats.profile2d_station_lookup_packet4_attempts;
    stats.shell_hull_profile2d_station_lookup_packet4_hits =
        shell_stats.profile2d_station_lookup_packet4_hits;
    stats.shell_hull_profile2d_station_lookup_packet4_misses =
        shell_stats.profile2d_station_lookup_packet4_misses;
    stats.jit_shell_float4_helper_calls =
        shell_stats.jit_shell_float4_helper_calls;
    stats.jit_shell_float4_helper_lanes =
        shell_stats.jit_shell_float4_helper_lanes;
    stats.jit_shell_float4_packet_fast_path_hits =
        shell_stats.jit_shell_float4_packet_fast_path_hits;
    stats.jit_shell_float4_scalar_fallbacks =
        shell_stats.jit_shell_float4_scalar_fallbacks;
    stats.jit_shell_float4_scalar_fallback_lanes =
        shell_stats.jit_shell_float4_scalar_fallback_lanes;
    stats.jit_shell_float4_spill_restore_bytes =
        shell_stats.jit_shell_float4_spill_restore_bytes;
    stats.shell_hull_profile2d_segment_tests =
        shell_stats.profile2d_segment_tests;
    stats.shell_hull_profile2d_bezier_tests =
        shell_stats.profile2d_bezier_tests;
    stats.shell_hull_profile2d_fallbacks = shell_stats.profile2d_fallbacks;
    stats.shell_hull_profile2d_edges_considered =
        shell_stats.profile2d_edges_considered;
    stats.shell_hull_profile2d_edges_aabb_pruned =
        shell_stats.profile2d_edges_aabb_pruned;
    stats.shell_hull_profile2d_edges_bezier_hull_pruned =
        shell_stats.profile2d_edges_bezier_hull_pruned;
    stats.shell_hull_profile2d_edge_distance_evaluations =
        shell_stats.profile2d_edge_distance_evaluations;
    stats.shell_hull_profile2d_linear_edges =
        shell_stats.profile2d_linear_edges;
    stats.shell_hull_profile2d_smooth_edges =
        shell_stats.profile2d_smooth_edges;
    stats.shell_hull_profile2d_endpoint_best_kept =
        shell_stats.profile2d_endpoint_best_kept;
    stats.shell_hull_profile2d_hermite_edges_refined =
        shell_stats.profile2d_hermite_edges_refined;
    stats.shell_hull_profile2d_hermite_seed_attempts =
        shell_stats.profile2d_hermite_seed_attempts;
    stats.shell_hull_profile2d_hermite_newton_iterations =
        shell_stats.profile2d_hermite_newton_iterations;
    stats.shell_hull_profile2d_hermite_iteration_1_attempts =
        shell_stats.profile2d_hermite_iteration_1_attempts;
    stats.shell_hull_profile2d_hermite_iteration_2_attempts =
        shell_stats.profile2d_hermite_iteration_2_attempts;
    stats.shell_hull_profile2d_hermite_iteration_3_attempts =
        shell_stats.profile2d_hermite_iteration_3_attempts;
    stats.shell_hull_profile2d_hermite_iteration_4_attempts =
        shell_stats.profile2d_hermite_iteration_4_attempts;
    stats.shell_hull_profile2d_hermite_clamped_endpoint_attempts =
        shell_stats.profile2d_hermite_clamped_endpoint_attempts;
    stats.shell_hull_profile2d_hermite_duplicate_t_attempts =
        shell_stats.profile2d_hermite_duplicate_t_attempts;
    stats.shell_hull_profile2d_hermite_distance_evaluations =
        shell_stats.profile2d_hermite_distance_evaluations;
    stats.shell_hull_profile2d_hermite_final_distance_evaluations =
        shell_stats.profile2d_hermite_final_distance_evaluations;
    stats.shell_hull_profile2d_hermite_wins_total =
        shell_stats.profile2d_hermite_wins_total;
    stats.shell_hull_profile2d_hermite_endpoint_wins =
        shell_stats.profile2d_hermite_endpoint_wins;
    stats.shell_hull_profile2d_hermite_quarter_wins =
        shell_stats.profile2d_hermite_quarter_wins;
    stats.shell_hull_profile2d_hermite_quarter_25_wins =
        shell_stats.profile2d_hermite_quarter_25_wins;
    stats.shell_hull_profile2d_hermite_quarter_50_wins =
        shell_stats.profile2d_hermite_quarter_50_wins;
    stats.shell_hull_profile2d_hermite_quarter_75_wins =
        shell_stats.profile2d_hermite_quarter_75_wins;
    stats.shell_hull_profile2d_hermite_height_wins =
        shell_stats.profile2d_hermite_height_wins;
    stats.shell_profile_interval_tiles =
        shell_stats.profile_shell_interval_tiles;
    stats.shell_profile_interval_rejected_tiles =
        shell_stats.profile_shell_interval_rejected_tiles;
    stats.shell_profile_interval_active_tiles =
        shell_stats.profile_shell_interval_active_tiles;
    stats.shell_profile_interval_single_segment_tiles =
        shell_stats.profile_shell_interval_single_segment_tiles;
    stats.shell_profile_interval_multi_segment_tiles =
        shell_stats.profile_shell_interval_multi_segment_tiles;
    stats.shell_interval_calls = shell_stats.interval_calls;
    stats.shell_active_segment_sum =
        shell_stats.profile_shell_interval_active_segment_sum;
    stats.shell_active_segment_samples =
        shell_stats.profile_shell_interval_active_tiles;
    stats.shell_closest_iterations =
        shell_stats.profile2d_hermite_newton_iterations;
    stats.shell_grad_helper_calls = shell_stats.profile2d_gradient_calls;
    stats.shell_interval_hot_loop_allocations =
        shell_stats.interval_hot_loop_allocations;
    stats.shell_float_slice_hot_loop_allocations =
        shell_stats.float_slice_hot_loop_allocations;
    stats.shell_grad_slice_hot_loop_allocations =
        shell_stats.grad_slice_hot_loop_allocations;
    stats.shell_hot_loop_allocations = shell_stats.hot_loop_allocations;
    stats.shell_allocations = shell_stats.hot_loop_allocations;
    set_shell_eval_stats_enabled(false);

    if shell_stats_enabled {
        let measured_eval_time = stats.interval_eval_time
            + stats.simplify_time
            + stats.float_eval_time
            + stats.grad_eval_time;
        let tile_overhead =
            stats.total_tile_time.saturating_sub(measured_eval_time);
        eprintln!(
            "render3d breakdown: tile_total={:.3}ms interval={:.3}ms simplify={:.3}ms float_eval={:.3}ms grad_eval={:.3}ms tile_overhead={:.3}ms merge={:.3}ms",
            stats.total_tile_time.as_secs_f64() * 1000.0,
            stats.interval_eval_time.as_secs_f64() * 1000.0,
            stats.simplify_time.as_secs_f64() * 1000.0,
            stats.float_eval_time.as_secs_f64() * 1000.0,
            stats.grad_eval_time.as_secs_f64() * 1000.0,
            tile_overhead.as_secs_f64() * 1000.0,
            merge_time.as_secs_f64() * 1000.0,
        );
        eprintln!(
            "render3d counters: interval_calls={} simplify_calls={} float_calls={} grad_calls={} float_samples={} grad_samples={}",
            stats.interval_eval_calls,
            stats.simplify_calls,
            stats.float_eval_calls,
            stats.grad_eval_calls,
            stats.float_eval_samples,
            stats.grad_eval_samples,
        );
        eprintln!(
            "render3d shell counters: shell_hull_profile2d_calls={} shell_hull_profile2d_distance_calls={} shell_hull_profile2d_gradient_calls={} shell_hull_profile2d_outer_distance_calls={} shell_hull_profile2d_inner_distance_calls={} shell_hull_profile2d_outer_gradient_calls={} shell_hull_profile2d_inner_gradient_calls={} shell_hull_profile2d_outer_distance_batches={} shell_hull_profile2d_outer_distance_batch_samples={} shell_hull_profile2d_outer_distance_max_batch_calls={} shell_hull_profile2d_outer_distance_aabb_reject_batches={} shell_hull_profile2d_outer_distance_aabb_reject_full_batches={} shell_hull_profile2d_outer_distance_aabb_reject_samples={} shell_hull_profile2d_segment_tests={} shell_hull_profile2d_bezier_tests={} shell_hull_profile2d_fallbacks={} shell_hull_profile2d_edges_considered={} shell_hull_profile2d_edges_aabb_pruned={} shell_hull_profile2d_linear_edges={} shell_hull_profile2d_smooth_edges={} shell_hull_profile2d_endpoint_best_kept={} shell_hull_profile2d_hermite_edges_refined={} shell_hull_profile2d_hermite_seed_attempts={} shell_hull_profile2d_hermite_newton_iterations={} shell_hull_profile2d_hermite_iteration_1_attempts={} shell_hull_profile2d_hermite_iteration_2_attempts={} shell_hull_profile2d_hermite_iteration_3_attempts={} shell_hull_profile2d_hermite_iteration_4_attempts={} shell_hull_profile2d_hermite_clamped_endpoint_attempts={} shell_hull_profile2d_hermite_duplicate_t_attempts={} shell_hull_profile2d_hermite_distance_evaluations={} shell_hull_profile2d_hermite_wins_total={} shell_hull_profile2d_hermite_endpoint_wins={} shell_hull_profile2d_hermite_quarter_wins={} shell_hull_profile2d_hermite_quarter_25_wins={} shell_hull_profile2d_hermite_quarter_50_wins={} shell_hull_profile2d_hermite_quarter_75_wins={} shell_hull_profile2d_hermite_height_wins={} shell_profile_interval_tiles={} shell_profile_interval_rejected_tiles={} shell_profile_interval_active_tiles={} shell_profile_interval_single_segment_tiles={} shell_profile_interval_multi_segment_tiles={} shell_interval_calls={} shell_interval_rejects={} shell_active_segment_avg={:.3} shell_closest_iterations={} shell_grad_helper_calls={} shell_interval_hot_loop_allocations={} shell_float_slice_hot_loop_allocations={} shell_grad_slice_hot_loop_allocations={} shell_hot_loop_allocations={} shell_allocations={}",
            stats.shell_hull_profile2d_calls,
            stats.shell_hull_profile2d_distance_calls,
            stats.shell_hull_profile2d_gradient_calls,
            stats.shell_hull_profile2d_outer_distance_calls,
            stats.shell_hull_profile2d_inner_distance_calls,
            stats.shell_hull_profile2d_outer_gradient_calls,
            stats.shell_hull_profile2d_inner_gradient_calls,
            stats.shell_hull_profile2d_outer_distance_batches,
            stats.shell_hull_profile2d_outer_distance_batch_samples,
            stats.shell_hull_profile2d_outer_distance_max_batch_calls,
            stats.shell_hull_profile2d_outer_distance_aabb_reject_batches,
            stats.shell_hull_profile2d_outer_distance_aabb_reject_full_batches,
            stats.shell_hull_profile2d_outer_distance_aabb_reject_samples,
            stats.shell_hull_profile2d_segment_tests,
            stats.shell_hull_profile2d_bezier_tests,
            stats.shell_hull_profile2d_fallbacks,
            stats.shell_hull_profile2d_edges_considered,
            stats.shell_hull_profile2d_edges_aabb_pruned,
            stats.shell_hull_profile2d_linear_edges,
            stats.shell_hull_profile2d_smooth_edges,
            stats.shell_hull_profile2d_endpoint_best_kept,
            stats.shell_hull_profile2d_hermite_edges_refined,
            stats.shell_hull_profile2d_hermite_seed_attempts,
            stats.shell_hull_profile2d_hermite_newton_iterations,
            stats.shell_hull_profile2d_hermite_iteration_1_attempts,
            stats.shell_hull_profile2d_hermite_iteration_2_attempts,
            stats.shell_hull_profile2d_hermite_iteration_3_attempts,
            stats.shell_hull_profile2d_hermite_iteration_4_attempts,
            stats.shell_hull_profile2d_hermite_clamped_endpoint_attempts,
            stats.shell_hull_profile2d_hermite_duplicate_t_attempts,
            stats.shell_hull_profile2d_hermite_distance_evaluations,
            stats.shell_hull_profile2d_hermite_wins_total,
            stats.shell_hull_profile2d_hermite_endpoint_wins,
            stats.shell_hull_profile2d_hermite_quarter_wins,
            stats.shell_hull_profile2d_hermite_quarter_25_wins,
            stats.shell_hull_profile2d_hermite_quarter_50_wins,
            stats.shell_hull_profile2d_hermite_quarter_75_wins,
            stats.shell_hull_profile2d_hermite_height_wins,
            stats.shell_profile_interval_tiles,
            stats.shell_profile_interval_rejected_tiles,
            stats.shell_profile_interval_active_tiles,
            stats.shell_profile_interval_single_segment_tiles,
            stats.shell_profile_interval_multi_segment_tiles,
            stats.shell_interval_calls,
            stats.shell_interval_rejects,
            stats.shell_active_segment_avg(),
            stats.shell_closest_iterations,
            stats.shell_grad_helper_calls,
            stats.shell_interval_hot_loop_allocations,
            stats.shell_float_slice_hot_loop_allocations,
            stats.shell_grad_slice_hot_loop_allocations,
            stats.shell_hot_loop_allocations,
            stats.shell_allocations,
        );
        eprintln!(
            "render3d shell profile breakdown: shell_hull_profile2d_station_lookup_calls={} shell_hull_profile2d_station_lookup_packet4_attempts={} shell_hull_profile2d_station_lookup_packet4_hits={} shell_hull_profile2d_station_lookup_packet4_misses={} jit_shell_float4_helper_calls={} jit_shell_float4_helper_lanes={} jit_shell_float4_packet_fast_path_hits={} jit_shell_float4_scalar_fallbacks={} jit_shell_float4_scalar_fallback_lanes={} jit_shell_float4_same_segment_rate={:.3} jit_shell_float4_avg_helper_batch={:.3} jit_shell_float4_spill_restore_bytes={} shell_hull_profile2d_edges_bezier_hull_pruned={} shell_hull_profile2d_edge_distance_evaluations={} shell_hull_profile2d_hermite_final_distance_evaluations={}",
            stats.shell_hull_profile2d_station_lookup_calls,
            stats.shell_hull_profile2d_station_lookup_packet4_attempts,
            stats.shell_hull_profile2d_station_lookup_packet4_hits,
            stats.shell_hull_profile2d_station_lookup_packet4_misses,
            stats.jit_shell_float4_helper_calls,
            stats.jit_shell_float4_helper_lanes,
            stats.jit_shell_float4_packet_fast_path_hits,
            stats.jit_shell_float4_scalar_fallbacks,
            stats.jit_shell_float4_scalar_fallback_lanes,
            stats.jit_shell_float4_same_segment_rate(),
            stats.jit_shell_float4_avg_helper_batch(),
            stats.jit_shell_float4_spill_restore_bytes,
            stats.shell_hull_profile2d_edges_bezier_hull_pruned,
            stats.shell_hull_profile2d_edge_distance_evaluations,
            stats.shell_hull_profile2d_hermite_final_distance_evaluations,
        );
    }
    Some((image, stats))
}

/// Renders leaf-center debug samples into a 3D image according to the provided
/// configuration.
///
/// This traversal mirrors [`render`] (same interval pruning and simplification)
/// but performs exactly one float + gradient sample per terminal 3D leaf tile.
pub fn render_leaf_debug<F: Function>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &VoxelRenderConfig,
) -> Option<LeafDebugBuffer> {
    let shape = shape.with_transform(config.mat());

    let shell_stats_enabled =
        std::env::var_os("FIDGET_RENDER3D_STATS").is_some();
    set_shell_eval_stats_enabled(shell_stats_enabled);
    reset_shell_eval_stats();
    let tiles =
        super::render_tiles::<F, DebugWorker<F>, _>(shape, vars, config)?;
    let tile_sizes = config.tile_sizes();

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image = LeafDebugBuffer::new(config.image_size);
    let mut stats = VoxelRenderStats::default();
    let merge_start = Instant::now();
    for (tile, out) in tiles {
        stats.merge(out.stats);
        let out = out.image;
        let mut index = 0;
        for j in 0..tile_sizes[0] {
            let y = j + tile.corner.y;
            for i in 0..tile_sizes[0] {
                let x = i + tile.corner.x;
                if x < width && y < height {
                    let o = y * width + x;
                    if out[index].depth >= image[o].depth {
                        image[o] = out[index];
                    }
                }
                index += 1;
            }
        }
    }
    let merge_time = merge_start.elapsed();
    let shell_stats = shell_eval_stats();
    stats.shell_hull_profile2d_calls = shell_stats.profile2d_calls;
    stats.shell_hull_profile2d_segment_tests =
        shell_stats.profile2d_segment_tests;
    stats.shell_hull_profile2d_distance_calls =
        shell_stats.profile2d_distance_calls;
    stats.shell_hull_profile2d_gradient_calls =
        shell_stats.profile2d_gradient_calls;
    stats.shell_hull_profile2d_outer_distance_calls =
        shell_stats.profile2d_outer_distance_calls;
    stats.shell_hull_profile2d_inner_distance_calls =
        shell_stats.profile2d_inner_distance_calls;
    stats.shell_hull_profile2d_outer_gradient_calls =
        shell_stats.profile2d_outer_gradient_calls;
    stats.shell_hull_profile2d_inner_gradient_calls =
        shell_stats.profile2d_inner_gradient_calls;
    stats.shell_hull_profile2d_station_lookup_calls =
        shell_stats.profile2d_station_lookup_calls;
    stats.shell_hull_profile2d_station_lookup_packet4_attempts =
        shell_stats.profile2d_station_lookup_packet4_attempts;
    stats.shell_hull_profile2d_station_lookup_packet4_hits =
        shell_stats.profile2d_station_lookup_packet4_hits;
    stats.shell_hull_profile2d_station_lookup_packet4_misses =
        shell_stats.profile2d_station_lookup_packet4_misses;
    stats.jit_shell_float4_helper_calls =
        shell_stats.jit_shell_float4_helper_calls;
    stats.jit_shell_float4_helper_lanes =
        shell_stats.jit_shell_float4_helper_lanes;
    stats.jit_shell_float4_packet_fast_path_hits =
        shell_stats.jit_shell_float4_packet_fast_path_hits;
    stats.jit_shell_float4_scalar_fallbacks =
        shell_stats.jit_shell_float4_scalar_fallbacks;
    stats.jit_shell_float4_scalar_fallback_lanes =
        shell_stats.jit_shell_float4_scalar_fallback_lanes;
    stats.jit_shell_float4_spill_restore_bytes =
        shell_stats.jit_shell_float4_spill_restore_bytes;
    stats.shell_hull_profile2d_bezier_tests =
        shell_stats.profile2d_bezier_tests;
    stats.shell_hull_profile2d_fallbacks = shell_stats.profile2d_fallbacks;
    stats.shell_hull_profile2d_edges_considered =
        shell_stats.profile2d_edges_considered;
    stats.shell_hull_profile2d_edges_aabb_pruned =
        shell_stats.profile2d_edges_aabb_pruned;
    stats.shell_hull_profile2d_edges_bezier_hull_pruned =
        shell_stats.profile2d_edges_bezier_hull_pruned;
    stats.shell_hull_profile2d_edge_distance_evaluations =
        shell_stats.profile2d_edge_distance_evaluations;
    stats.shell_hull_profile2d_linear_edges =
        shell_stats.profile2d_linear_edges;
    stats.shell_hull_profile2d_smooth_edges =
        shell_stats.profile2d_smooth_edges;
    stats.shell_hull_profile2d_endpoint_best_kept =
        shell_stats.profile2d_endpoint_best_kept;
    stats.shell_hull_profile2d_hermite_edges_refined =
        shell_stats.profile2d_hermite_edges_refined;
    stats.shell_hull_profile2d_hermite_seed_attempts =
        shell_stats.profile2d_hermite_seed_attempts;
    stats.shell_hull_profile2d_hermite_newton_iterations =
        shell_stats.profile2d_hermite_newton_iterations;
    stats.shell_hull_profile2d_hermite_iteration_1_attempts =
        shell_stats.profile2d_hermite_iteration_1_attempts;
    stats.shell_hull_profile2d_hermite_iteration_2_attempts =
        shell_stats.profile2d_hermite_iteration_2_attempts;
    stats.shell_hull_profile2d_hermite_iteration_3_attempts =
        shell_stats.profile2d_hermite_iteration_3_attempts;
    stats.shell_hull_profile2d_hermite_iteration_4_attempts =
        shell_stats.profile2d_hermite_iteration_4_attempts;
    stats.shell_hull_profile2d_hermite_clamped_endpoint_attempts =
        shell_stats.profile2d_hermite_clamped_endpoint_attempts;
    stats.shell_hull_profile2d_hermite_duplicate_t_attempts =
        shell_stats.profile2d_hermite_duplicate_t_attempts;
    stats.shell_hull_profile2d_hermite_distance_evaluations =
        shell_stats.profile2d_hermite_distance_evaluations;
    stats.shell_hull_profile2d_hermite_final_distance_evaluations =
        shell_stats.profile2d_hermite_final_distance_evaluations;
    stats.shell_hull_profile2d_hermite_wins_total =
        shell_stats.profile2d_hermite_wins_total;
    stats.shell_hull_profile2d_hermite_endpoint_wins =
        shell_stats.profile2d_hermite_endpoint_wins;
    stats.shell_hull_profile2d_hermite_quarter_wins =
        shell_stats.profile2d_hermite_quarter_wins;
    stats.shell_hull_profile2d_hermite_quarter_25_wins =
        shell_stats.profile2d_hermite_quarter_25_wins;
    stats.shell_hull_profile2d_hermite_quarter_50_wins =
        shell_stats.profile2d_hermite_quarter_50_wins;
    stats.shell_hull_profile2d_hermite_quarter_75_wins =
        shell_stats.profile2d_hermite_quarter_75_wins;
    stats.shell_hull_profile2d_hermite_height_wins =
        shell_stats.profile2d_hermite_height_wins;
    stats.shell_profile_interval_tiles =
        shell_stats.profile_shell_interval_tiles;
    stats.shell_profile_interval_rejected_tiles =
        shell_stats.profile_shell_interval_rejected_tiles;
    stats.shell_profile_interval_active_tiles =
        shell_stats.profile_shell_interval_active_tiles;
    stats.shell_profile_interval_single_segment_tiles =
        shell_stats.profile_shell_interval_single_segment_tiles;
    stats.shell_profile_interval_multi_segment_tiles =
        shell_stats.profile_shell_interval_multi_segment_tiles;
    stats.shell_interval_calls = shell_stats.interval_calls;
    stats.shell_active_segment_sum =
        shell_stats.profile_shell_interval_active_segment_sum;
    stats.shell_active_segment_samples =
        shell_stats.profile_shell_interval_active_tiles;
    stats.shell_closest_iterations =
        shell_stats.profile2d_hermite_newton_iterations;
    stats.shell_grad_helper_calls = shell_stats.profile2d_gradient_calls;
    stats.shell_interval_hot_loop_allocations =
        shell_stats.interval_hot_loop_allocations;
    stats.shell_float_slice_hot_loop_allocations =
        shell_stats.float_slice_hot_loop_allocations;
    stats.shell_grad_slice_hot_loop_allocations =
        shell_stats.grad_slice_hot_loop_allocations;
    stats.shell_hot_loop_allocations = shell_stats.hot_loop_allocations;
    stats.shell_allocations = shell_stats.hot_loop_allocations;
    set_shell_eval_stats_enabled(false);

    if shell_stats_enabled {
        let measured_eval_time = stats.interval_eval_time
            + stats.simplify_time
            + stats.float_eval_time
            + stats.grad_eval_time;
        let tile_overhead =
            stats.total_tile_time.saturating_sub(measured_eval_time);
        eprintln!(
            "render3d leaf-debug breakdown: tile_total={:.3}ms interval={:.3}ms simplify={:.3}ms float_eval={:.3}ms grad_eval={:.3}ms tile_overhead={:.3}ms merge={:.3}ms",
            stats.total_tile_time.as_secs_f64() * 1000.0,
            stats.interval_eval_time.as_secs_f64() * 1000.0,
            stats.simplify_time.as_secs_f64() * 1000.0,
            stats.float_eval_time.as_secs_f64() * 1000.0,
            stats.grad_eval_time.as_secs_f64() * 1000.0,
            tile_overhead.as_secs_f64() * 1000.0,
            merge_time.as_secs_f64() * 1000.0,
        );
        eprintln!(
            "render3d leaf-debug counters: interval_calls={} simplify_calls={} float_calls={} grad_calls={} float_samples={} grad_samples={}",
            stats.interval_eval_calls,
            stats.simplify_calls,
            stats.float_eval_calls,
            stats.grad_eval_calls,
            stats.float_eval_samples,
            stats.grad_eval_samples,
        );
        eprintln!(
            "render3d leaf-debug shell counters: shell_hull_profile2d_calls={} shell_hull_profile2d_distance_calls={} shell_hull_profile2d_gradient_calls={} shell_hull_profile2d_outer_distance_calls={} shell_hull_profile2d_inner_distance_calls={} shell_hull_profile2d_outer_gradient_calls={} shell_hull_profile2d_inner_gradient_calls={} shell_hull_profile2d_segment_tests={} shell_hull_profile2d_bezier_tests={} shell_hull_profile2d_fallbacks={} shell_hull_profile2d_edges_considered={} shell_hull_profile2d_edges_aabb_pruned={} shell_hull_profile2d_linear_edges={} shell_hull_profile2d_smooth_edges={} shell_hull_profile2d_endpoint_best_kept={} shell_hull_profile2d_hermite_edges_refined={} shell_hull_profile2d_hermite_seed_attempts={} shell_hull_profile2d_hermite_newton_iterations={} shell_hull_profile2d_hermite_iteration_1_attempts={} shell_hull_profile2d_hermite_iteration_2_attempts={} shell_hull_profile2d_hermite_iteration_3_attempts={} shell_hull_profile2d_hermite_iteration_4_attempts={} shell_hull_profile2d_hermite_clamped_endpoint_attempts={} shell_hull_profile2d_hermite_duplicate_t_attempts={} shell_hull_profile2d_hermite_distance_evaluations={} shell_hull_profile2d_hermite_wins_total={} shell_hull_profile2d_hermite_endpoint_wins={} shell_hull_profile2d_hermite_quarter_wins={} shell_hull_profile2d_hermite_quarter_25_wins={} shell_hull_profile2d_hermite_quarter_50_wins={} shell_hull_profile2d_hermite_quarter_75_wins={} shell_hull_profile2d_hermite_height_wins={} shell_profile_interval_tiles={} shell_profile_interval_rejected_tiles={} shell_profile_interval_active_tiles={} shell_profile_interval_single_segment_tiles={} shell_profile_interval_multi_segment_tiles={} shell_interval_calls={} shell_interval_rejects={} shell_active_segment_avg={:.3} shell_closest_iterations={} shell_grad_helper_calls={} shell_interval_hot_loop_allocations={} shell_float_slice_hot_loop_allocations={} shell_grad_slice_hot_loop_allocations={} shell_hot_loop_allocations={} shell_allocations={}",
            stats.shell_hull_profile2d_calls,
            stats.shell_hull_profile2d_distance_calls,
            stats.shell_hull_profile2d_gradient_calls,
            stats.shell_hull_profile2d_outer_distance_calls,
            stats.shell_hull_profile2d_inner_distance_calls,
            stats.shell_hull_profile2d_outer_gradient_calls,
            stats.shell_hull_profile2d_inner_gradient_calls,
            stats.shell_hull_profile2d_segment_tests,
            stats.shell_hull_profile2d_bezier_tests,
            stats.shell_hull_profile2d_fallbacks,
            stats.shell_hull_profile2d_edges_considered,
            stats.shell_hull_profile2d_edges_aabb_pruned,
            stats.shell_hull_profile2d_linear_edges,
            stats.shell_hull_profile2d_smooth_edges,
            stats.shell_hull_profile2d_endpoint_best_kept,
            stats.shell_hull_profile2d_hermite_edges_refined,
            stats.shell_hull_profile2d_hermite_seed_attempts,
            stats.shell_hull_profile2d_hermite_newton_iterations,
            stats.shell_hull_profile2d_hermite_iteration_1_attempts,
            stats.shell_hull_profile2d_hermite_iteration_2_attempts,
            stats.shell_hull_profile2d_hermite_iteration_3_attempts,
            stats.shell_hull_profile2d_hermite_iteration_4_attempts,
            stats.shell_hull_profile2d_hermite_clamped_endpoint_attempts,
            stats.shell_hull_profile2d_hermite_duplicate_t_attempts,
            stats.shell_hull_profile2d_hermite_distance_evaluations,
            stats.shell_hull_profile2d_hermite_wins_total,
            stats.shell_hull_profile2d_hermite_endpoint_wins,
            stats.shell_hull_profile2d_hermite_quarter_wins,
            stats.shell_hull_profile2d_hermite_quarter_25_wins,
            stats.shell_hull_profile2d_hermite_quarter_50_wins,
            stats.shell_hull_profile2d_hermite_quarter_75_wins,
            stats.shell_hull_profile2d_hermite_height_wins,
            stats.shell_profile_interval_tiles,
            stats.shell_profile_interval_rejected_tiles,
            stats.shell_profile_interval_active_tiles,
            stats.shell_profile_interval_single_segment_tiles,
            stats.shell_profile_interval_multi_segment_tiles,
            stats.shell_interval_calls,
            stats.shell_interval_rejects,
            stats.shell_active_segment_avg(),
            stats.shell_closest_iterations,
            stats.shell_grad_helper_calls,
            stats.shell_interval_hot_loop_allocations,
            stats.shell_float_slice_hot_loop_allocations,
            stats.shell_grad_slice_hot_loop_allocations,
            stats.shell_hot_loop_allocations,
            stats.shell_allocations,
        );
    }

    Some(image)
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{
        Context,
        render::{TileSizes, VoxelSize},
        vm::VmShape,
    };

    /// Make sure we don't crash if there's only a single tile
    #[test]
    fn test_tile_queues() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = VoxelRenderConfig {
            image_size: VoxelSize::from(128), // very small!
            ..Default::default()
        };
        let image = cfg.run(shape).unwrap();
        assert_eq!(image.len(), 128 * 128);
    }

    #[test]
    fn test_leaf_debug_tile_queues() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = VoxelRenderConfig {
            image_size: VoxelSize::from(128), // very small!
            ..Default::default()
        };
        let image = cfg.run_leaf_debug(shape).unwrap();
        assert_eq!(image.len(), 128 * 128);
    }

    #[test]
    fn cancel_render() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = VoxelRenderConfig {
            image_size: VoxelSize::new(64, 64, 64),
            ..Default::default()
        };
        let cancel = cfg.cancel.clone();
        cancel.cancel();
        let out = cfg.run::<_>(shape);
        assert!(out.is_none());
    }

    #[test]
    fn render_stops_sampling_leaf_column_after_visible_surface() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = VoxelRenderConfig {
            image_size: VoxelSize::new(8, 8, 8),
            tile_sizes: TileSizes::new(&[8]).unwrap(),
            ..Default::default()
        };
        let (_image, stats) = cfg.run_with_stats(shape).unwrap();

        assert_eq!(stats.float_eval_samples, 288);
    }

    #[test]
    fn stats_record_outer_profile_distance_batch_locality() {
        let mut stats = VoxelRenderStats::default();

        stats.record_profile2d_outer_distance_batch(8, 3);
        stats.record_profile2d_outer_distance_batch(0, 5);
        stats.record_profile2d_outer_distance_aabb_rejection_potential(8, 3, 2);
        stats.record_profile2d_outer_distance_aabb_rejection_potential(0, 5, 5);
        stats.record_profile2d_outer_distance_aabb_rejection_potential(8, 3, 3);

        assert_eq!(stats.shell_hull_profile2d_outer_distance_batches, 1);
        assert_eq!(stats.shell_hull_profile2d_outer_distance_batch_samples, 3);
        assert_eq!(
            stats.shell_hull_profile2d_outer_distance_max_batch_calls,
            8
        );
        assert_eq!(
            stats.shell_hull_profile2d_outer_distance_aabb_reject_batches,
            2
        );
        assert_eq!(
            stats.shell_hull_profile2d_outer_distance_aabb_reject_full_batches,
            1
        );
        assert_eq!(
            stats.shell_hull_profile2d_outer_distance_aabb_reject_samples,
            5
        );
    }

    #[test]
    fn native_aabb_rejectable_sample_count_uses_model_space_bounds() {
        let bounds = ShellBounds {
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            max_x: 1.0,
            max_y: 1.0,
            max_z: 1.0,
        };
        let model_from_image = Matrix4::new_scaling(0.5);

        let rejected = native_aabb_rejectable_sample_count(
            &bounds,
            &model_from_image,
            &[0.5, 2.0, 3.0],
            &[0.5, 2.0, 0.5],
            &[0.5, 2.0, 0.5],
        );

        assert_eq!(rejected, 1);
    }

    #[test]
    fn stats_merge_preserves_shell_summary_counters() {
        let mut stats = VoxelRenderStats {
            shell_hull_profile2d_station_lookup_calls: 29,
            shell_hull_profile2d_station_lookup_packet4_attempts: 3,
            shell_hull_profile2d_station_lookup_packet4_hits: 2,
            shell_hull_profile2d_station_lookup_packet4_misses: 1,
            jit_shell_float4_helper_calls: 7,
            jit_shell_float4_helper_lanes: 28,
            jit_shell_float4_packet_fast_path_hits: 5,
            jit_shell_float4_scalar_fallbacks: 2,
            jit_shell_float4_scalar_fallback_lanes: 8,
            jit_shell_float4_spill_restore_bytes: 100,
            shell_hull_profile2d_edges_bezier_hull_pruned: 31,
            shell_hull_profile2d_edge_distance_evaluations: 37,
            shell_hull_profile2d_hermite_final_distance_evaluations: 41,
            shell_interval_rejects: 2,
            shell_active_segment_sum: 5,
            shell_active_segment_samples: 2,
            shell_closest_iterations: 7,
            shell_grad_helper_calls: 11,
            shell_allocations: 13,
            ..Default::default()
        };
        stats.merge(VoxelRenderStats {
            shell_hull_profile2d_station_lookup_calls: 43,
            shell_hull_profile2d_station_lookup_packet4_attempts: 5,
            shell_hull_profile2d_station_lookup_packet4_hits: 4,
            shell_hull_profile2d_station_lookup_packet4_misses: 1,
            jit_shell_float4_helper_calls: 11,
            jit_shell_float4_helper_lanes: 44,
            jit_shell_float4_packet_fast_path_hits: 7,
            jit_shell_float4_scalar_fallbacks: 4,
            jit_shell_float4_scalar_fallback_lanes: 16,
            jit_shell_float4_spill_restore_bytes: 200,
            shell_hull_profile2d_edges_bezier_hull_pruned: 47,
            shell_hull_profile2d_edge_distance_evaluations: 53,
            shell_hull_profile2d_hermite_final_distance_evaluations: 59,
            shell_interval_rejects: 3,
            shell_active_segment_sum: 4,
            shell_active_segment_samples: 1,
            shell_closest_iterations: 17,
            shell_grad_helper_calls: 19,
            shell_allocations: 23,
            ..Default::default()
        });

        assert_eq!(stats.shell_hull_profile2d_station_lookup_calls, 72);
        assert_eq!(
            stats.shell_hull_profile2d_station_lookup_packet4_attempts,
            8
        );
        assert_eq!(stats.shell_hull_profile2d_station_lookup_packet4_hits, 6);
        assert_eq!(stats.shell_hull_profile2d_station_lookup_packet4_misses, 2);
        assert_eq!(stats.jit_shell_float4_helper_calls, 18);
        assert_eq!(stats.jit_shell_float4_helper_lanes, 72);
        assert_eq!(stats.jit_shell_float4_packet_fast_path_hits, 12);
        assert_eq!(stats.jit_shell_float4_scalar_fallbacks, 6);
        assert_eq!(stats.jit_shell_float4_scalar_fallback_lanes, 24);
        assert_eq!(stats.jit_shell_float4_same_segment_rate(), 12.0 / 18.0);
        assert_eq!(stats.jit_shell_float4_avg_helper_batch(), 4.0);
        assert_eq!(stats.jit_shell_float4_spill_restore_bytes, 300);
        assert_eq!(stats.shell_hull_profile2d_edges_bezier_hull_pruned, 78);
        assert_eq!(stats.shell_hull_profile2d_edge_distance_evaluations, 90);
        assert_eq!(
            stats.shell_hull_profile2d_hermite_final_distance_evaluations,
            100
        );
        assert_eq!(stats.shell_interval_rejects, 5);
        assert_eq!(stats.shell_active_segment_sum, 9);
        assert_eq!(stats.shell_active_segment_samples, 3);
        assert_eq!(stats.shell_active_segment_avg(), 3.0);
        assert_eq!(stats.shell_closest_iterations, 24);
        assert_eq!(stats.shell_grad_helper_calls, 30);
        assert_eq!(stats.shell_allocations, 36);
    }
}
