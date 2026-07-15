//! 3D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    Image as GenericImage, RenderSize as _, RenderWorker, Tile, TileSizesRef,
};
use fidget_core::{
    eval::Function,
    render::{CancelToken, RenderHints, ThreadPool, TileSizes},
    shape::{
        BoundShape, IntoBoundShape, ShapeBulkEval, ShapeTracingEval, ShapeVars,
    },
    types::{Grad, Interval},
};

use nalgebra::{Matrix4, Point3, Vector2, Vector3};
use std::time::{Duration, Instant};
use zerocopy::{FromBytes, Immutable, IntoBytes};

/// Image containing depth and normal at each pixel
pub type Image = GenericImage<GeometryPixel, RenderSize>;

/// Size type for 3D rendering
pub type RenderSize = fidget_core::render::VoxelSize;

////////////////////////////////////////////////////////////////////////////////

/// Settings for 3D rendering
pub struct RenderConfig<'a> {
    /// Render size
    ///
    /// The resulting image will have the given width and height; depth sets the
    /// number of voxels to evaluate within each pixel of the image (stacked
    /// into a column going into the screen).
    pub image_size: RenderSize,

    /// World-to-model transform
    pub world_to_model: Matrix4<f32>,

    /// Tile sizes to use during evaluation.
    ///
    /// If this is `None`, then evaluation will use
    /// [`RenderHints::tile_sizes_3d`] to select based on evaluator type.
    pub tile_sizes: Option<TileSizes>,

    /// Thread pool to use for rendering
    ///
    /// If this is `None`, then rendering is done in a single thread; otherwise,
    /// the provided pool is used.
    pub threads: Option<&'a ThreadPool>,

    /// Token to cancel rendering
    pub cancel: CancelToken,
}

impl Default for RenderConfig<'_> {
    fn default() -> Self {
        Self {
            image_size: RenderSize::from(512),
            tile_sizes: None,
            world_to_model: Matrix4::identity(),
            threads: Some(&ThreadPool::Global),
            cancel: CancelToken::new(),
        }
    }
}

impl crate::RenderConfig for RenderConfig<'_> {
    fn threads(&self) -> Option<&ThreadPool> {
        self.threads
    }
    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

impl crate::RenderSize for RenderConfig<'_> {
    fn width(&self) -> u32 {
        self.image_size.width()
    }
    fn height(&self) -> u32 {
        self.image_size.height()
    }
}

impl RenderConfig<'_> {
    /// Render a shape in 3D using this configuration
    ///
    /// In the resulting image, saturated pixels (i.e. pixels in the image which
    /// are fully occupied up to the camera) are represented with `depth =
    /// self.image_size.depth()` and a normal of `[0, 0, 1]`.
    pub fn run<'b, S>(&self, shape: S) -> Option<Image>
    where
        S: IntoBoundShape<'b>,
        S::Function: RenderHints,
    {
        render(shape.into_bound_shape().ok()?, self)
    }

    /// Render a shape in 3D, returning geometry plus compatibility stats.
    pub fn run_with_stats<'b, S>(
        &self,
        shape: S,
    ) -> Option<(Image, VoxelRenderStats)>
    where
        S: IntoBoundShape<'b>,
        S::Function: RenderHints,
    {
        render_with_stats(shape.into_bound_shape().ok()?, self)
    }

    /// Returns the combined screen-to-model transform matrix
    pub fn mat(&self) -> Matrix4<f32> {
        self.world_to_model * self.image_size.screen_to_world()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Pixel type for a [`voxel::Image`](Image)
///
/// This type can be passed directly in a buffer to the GPU.
#[repr(C)]
#[derive(
    Debug, Default, Copy, Clone, IntoBytes, FromBytes, Immutable, PartialEq,
)]
pub struct GeometryPixel {
    /// Z position of this pixel, in voxel units
    ///
    /// The fractional component is always zero. Empty pixels always have a
    /// depth of 0.
    pub depth: f32,
    /// Function gradients at this pixel
    pub normal: [f32; 3],
}

impl GeometryPixel {
    /// Converts the normal into a normalized RGB value
    pub fn to_color(&self) -> [u8; 3] {
        let [dx, dy, dz] = self.normal;
        let s = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
        if s != 0.0 {
            let scale = u8::MAX as f32 / s;
            [
                (dx.abs() * scale) as u8,
                (dy.abs() * scale) as u8,
                (dz.abs() * scale) as u8,
            ]
        } else {
            [0; 3]
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

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
    /// JIT native shell helper calls crossing the Rust helper ABI.
    pub shell_jit_helper_calls: u64,
    /// Logical sample lanes processed by JIT native shell helpers.
    pub shell_jit_helper_lanes: u64,
    /// Scalar point JIT native shell helper calls.
    pub shell_jit_point_helper_calls: u64,
    /// Four-lane float-slice JIT native shell helper calls.
    pub shell_jit_float4_helper_calls: u64,
    /// Logical sample lanes processed by float4 JIT native shell helpers.
    pub shell_jit_float4_helper_lanes: u64,
    /// Interval JIT native shell helper calls.
    pub shell_jit_interval_helper_calls: u64,
    /// Gradient JIT native shell helper calls.
    pub shell_jit_grad_helper_calls: u64,
    /// Helper calls matching the fixed-topology specialization stub.
    pub shell_jit_fixed_topology_helper_candidate_calls: u64,
    /// Helper lanes matching the fixed-topology specialization stub.
    pub shell_jit_fixed_topology_helper_candidate_lanes: u64,
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
    fn merge_worker(&mut self, other: VoxelRenderStats) {
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
        self.shell_hull_profile2d_outer_distance_batches +=
            other.shell_hull_profile2d_outer_distance_batches;
        self.shell_hull_profile2d_outer_distance_batch_samples +=
            other.shell_hull_profile2d_outer_distance_batch_samples;
        self.shell_hull_profile2d_outer_distance_max_batch_calls = self
            .shell_hull_profile2d_outer_distance_max_batch_calls
            .max(other.shell_hull_profile2d_outer_distance_max_batch_calls);
    }

    fn merge_shell(&mut self, shell: fidget_core::shell::ShellEvalStats) {
        self.shell_hull_profile2d_calls = shell.profile2d_calls;
        self.shell_hull_profile2d_distance_calls =
            shell.profile2d_distance_calls;
        self.shell_hull_profile2d_gradient_calls =
            shell.profile2d_gradient_calls;
        self.shell_hull_profile2d_outer_distance_calls =
            shell.profile2d_outer_distance_calls;
        self.shell_hull_profile2d_inner_distance_calls =
            shell.profile2d_inner_distance_calls;
        self.shell_hull_profile2d_outer_gradient_calls =
            shell.profile2d_outer_gradient_calls;
        self.shell_hull_profile2d_inner_gradient_calls =
            shell.profile2d_inner_gradient_calls;
        self.shell_hull_profile2d_station_lookup_calls =
            shell.profile2d_station_lookup_calls;
        self.shell_hull_profile2d_station_lookup_packet4_attempts =
            shell.profile2d_station_lookup_packet4_attempts;
        self.shell_hull_profile2d_station_lookup_packet4_hits =
            shell.profile2d_station_lookup_packet4_hits;
        self.shell_hull_profile2d_station_lookup_packet4_misses =
            shell.profile2d_station_lookup_packet4_misses;
        self.jit_shell_float4_helper_calls =
            shell.jit_shell_float4_helper_calls;
        self.jit_shell_float4_helper_lanes =
            shell.jit_shell_float4_helper_lanes;
        self.jit_shell_float4_packet_fast_path_hits =
            shell.jit_shell_float4_packet_fast_path_hits;
        self.jit_shell_float4_scalar_fallbacks =
            shell.jit_shell_float4_scalar_fallbacks;
        self.jit_shell_float4_scalar_fallback_lanes =
            shell.jit_shell_float4_scalar_fallback_lanes;
        self.jit_shell_float4_spill_restore_bytes =
            shell.jit_shell_float4_spill_restore_bytes;
        self.shell_hull_profile2d_segment_tests = shell.profile2d_segment_tests;
        self.shell_hull_profile2d_bezier_tests = shell.profile2d_bezier_tests;
        self.shell_hull_profile2d_fallbacks = shell.profile2d_fallbacks;
        self.shell_hull_profile2d_edges_considered =
            shell.profile2d_edges_considered;
        self.shell_hull_profile2d_edges_aabb_pruned =
            shell.profile2d_edges_aabb_pruned;
        self.shell_hull_profile2d_edges_bezier_hull_pruned =
            shell.profile2d_edges_bezier_hull_pruned;
        self.shell_hull_profile2d_edge_distance_evaluations =
            shell.profile2d_edge_distance_evaluations;
        self.shell_hull_profile2d_linear_edges = shell.profile2d_linear_edges;
        self.shell_hull_profile2d_smooth_edges = shell.profile2d_smooth_edges;
        self.shell_hull_profile2d_endpoint_best_kept =
            shell.profile2d_endpoint_best_kept;
        self.shell_hull_profile2d_hermite_edges_refined =
            shell.profile2d_hermite_edges_refined;
        self.shell_hull_profile2d_hermite_seed_attempts =
            shell.profile2d_hermite_seed_attempts;
        self.shell_hull_profile2d_hermite_newton_iterations =
            shell.profile2d_hermite_newton_iterations;
        self.shell_hull_profile2d_hermite_iteration_1_attempts =
            shell.profile2d_hermite_iteration_1_attempts;
        self.shell_hull_profile2d_hermite_iteration_2_attempts =
            shell.profile2d_hermite_iteration_2_attempts;
        self.shell_hull_profile2d_hermite_iteration_3_attempts =
            shell.profile2d_hermite_iteration_3_attempts;
        self.shell_hull_profile2d_hermite_iteration_4_attempts =
            shell.profile2d_hermite_iteration_4_attempts;
        self.shell_hull_profile2d_hermite_clamped_endpoint_attempts =
            shell.profile2d_hermite_clamped_endpoint_attempts;
        self.shell_hull_profile2d_hermite_duplicate_t_attempts =
            shell.profile2d_hermite_duplicate_t_attempts;
        self.shell_hull_profile2d_hermite_distance_evaluations =
            shell.profile2d_hermite_distance_evaluations;
        self.shell_hull_profile2d_hermite_final_distance_evaluations =
            shell.profile2d_hermite_final_distance_evaluations;
        self.shell_hull_profile2d_hermite_wins_total =
            shell.profile2d_hermite_wins_total;
        self.shell_hull_profile2d_hermite_endpoint_wins =
            shell.profile2d_hermite_endpoint_wins;
        self.shell_hull_profile2d_hermite_quarter_wins =
            shell.profile2d_hermite_quarter_wins;
        self.shell_hull_profile2d_hermite_quarter_25_wins =
            shell.profile2d_hermite_quarter_25_wins;
        self.shell_hull_profile2d_hermite_quarter_50_wins =
            shell.profile2d_hermite_quarter_50_wins;
        self.shell_hull_profile2d_hermite_quarter_75_wins =
            shell.profile2d_hermite_quarter_75_wins;
        self.shell_hull_profile2d_hermite_height_wins =
            shell.profile2d_hermite_height_wins;
        self.shell_profile_interval_tiles = shell.profile_shell_interval_tiles;
        self.shell_profile_interval_rejected_tiles =
            shell.profile_shell_interval_rejected_tiles;
        self.shell_profile_interval_active_tiles =
            shell.profile_shell_interval_active_tiles;
        self.shell_profile_interval_single_segment_tiles =
            shell.profile_shell_interval_single_segment_tiles;
        self.shell_profile_interval_multi_segment_tiles =
            shell.profile_shell_interval_multi_segment_tiles;
        self.shell_interval_calls = shell.interval_calls;
        self.shell_interval_rejects =
            shell.profile_shell_interval_rejected_tiles;
        self.shell_active_segment_sum =
            shell.profile_shell_interval_active_segment_sum;
        self.shell_active_segment_samples =
            shell.profile_shell_interval_active_tiles;
        self.shell_grad_helper_calls = shell.profile2d_gradient_calls;
        self.shell_jit_helper_calls = shell.jit_shell_helper_calls;
        self.shell_jit_helper_lanes = shell.jit_shell_helper_lanes;
        self.shell_jit_point_helper_calls = shell.jit_shell_point_helper_calls;
        self.shell_jit_float4_helper_calls =
            shell.jit_shell_float4_helper_calls;
        self.shell_jit_float4_helper_lanes =
            shell.jit_shell_float4_helper_lanes;
        self.shell_jit_interval_helper_calls =
            shell.jit_shell_interval_helper_calls;
        self.shell_jit_grad_helper_calls = shell.jit_shell_grad_helper_calls;
        self.shell_jit_fixed_topology_helper_candidate_calls =
            shell.jit_shell_fixed_topology_helper_candidate_calls;
        self.shell_jit_fixed_topology_helper_candidate_lanes =
            shell.jit_shell_fixed_topology_helper_candidate_lanes;
        self.shell_interval_hot_loop_allocations =
            shell.interval_hot_loop_allocations;
        self.shell_float_slice_hot_loop_allocations =
            shell.float_slice_hot_loop_allocations;
        self.shell_grad_slice_hot_loop_allocations =
            shell.grad_slice_hot_loop_allocations;
        self.shell_hot_loop_allocations = shell.hot_loop_allocations;
        self.shell_allocations = shell.hot_loop_allocations;
    }
}

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
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Worker<'a, F: Function> {
    tile_sizes: TileSizesRef<'a>,
    vars: &'a ShapeVars<f32>,

    transform: nalgebra::Matrix4<f32>,
    image_size: RenderSize,

    /// Reusable workspace for evaluation, to minimize allocation
    scratch: Scratch,
    stats: VoxelRenderStats,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,

    /// Output images for this specific tile
    out: Image,
}

impl<'a, F: Function> RenderWorker<'a, F> for Worker<'a, F> {
    type Config = RenderConfig<'a>;
    type Output = (Image, VoxelRenderStats);

    fn new(
        cfg: &'a Self::Config,
        tile_sizes: TileSizesRef<'a>,
        vars: &'a ShapeVars<f32>,
    ) -> Self {
        let transform = cfg.mat();
        let buf_size = tile_sizes.last();
        let scratch = Scratch::new(buf_size);
        Worker {
            tile_sizes,
            vars,

            scratch,
            stats: VoxelRenderStats::default(),
            out: Default::default(),

            transform,
            image_size: cfg.image_size,

            eval_float_slice: Default::default(),
            eval_interval: Default::default(),
            eval_grad_slice: Default::default(),

            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        }
    }

    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F>,
        tile: Tile<2>,
    ) -> Self::Output {
        let started = Instant::now();
        self.stats = VoxelRenderStats::default();
        // Prepare local tile data to fill out
        let root_tile_size = self.tile_sizes[0];
        self.out = Image::new(RenderSize::from(root_tile_size as u32));
        for k in (0..self.image_size[2].div_ceil(root_tile_size as u32)).rev() {
            let tile = Tile::new(Point3::new(
                tile.corner.x,
                tile.corner.y,
                k as usize * root_tile_size,
            ));
            if !self.render_tile_recurse(shape, 0, tile) {
                break;
            }
        }
        self.stats.total_tile_time += started.elapsed();
        (std::mem::take(&mut self.out), self.stats)
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
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
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

        let interval_started = Instant::now();
        let (i, trace) = self
            .eval_interval
            .eval_with_transform_and_vars(
                shape.i_tape(&mut self.tape_storage),
                x,
                y,
                z,
                &self.transform,
                self.vars,
            )
            .unwrap();
        self.stats.interval_eval_time += interval_started.elapsed();
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
            return true; // complete empty, keep going
        }

        // Calculate a simplified tape based on the trace
        let sub_tape = if let Some(trace) = trace.as_ref() {
            let simplify_started = Instant::now();
            let simplified = shape.simplify(
                trace,
                &mut self.workspace,
                &mut self.shape_storage,
                &mut self.tape_storage,
            );
            self.stats.simplify_time += simplify_started.elapsed();
            self.stats.simplify_calls += 1;
            simplified
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
            self.render_tile_pixels(sub_tape, tile_size, tile);
        };
        // TODO recycle something here?
        true // keep going
    }

    fn render_tile_pixels(
        &mut self,
        shape: &mut RenderHandle<F>,
        tile_size: usize,
        tile: Tile<3>,
    ) {
        // Prepare for pixel-by-pixel evaluation
        let mut index = 0;
        assert!(self.scratch.x.len() >= tile_size.pow(3));
        assert!(self.scratch.y.len() >= tile_size.pow(3));
        assert!(self.scratch.z.len() >= tile_size.pow(3));
        self.scratch.columns.clear();
        for xy in 0..tile_size.pow(2) {
            let i = xy % tile_size;
            let j = xy / tile_size;

            let o = self.tile_sizes.pixel_offset(tile.add(Vector2::new(i, j)));

            // Skip pixels which are behind the image
            let zmax = (tile.corner[2] + tile_size) as f32;
            if self.out[o].depth >= zmax {
                continue;
            }

            for k in (0..tile_size).rev() {
                // SAFETY:
                // Index cannot exceed tile_size**3, which is (a) the size
                // that we allocated in `Scratch::new` and (b) checked by
                // assertions above.
                //
                // Using unsafe indexing here is a roughly 2.5% speedup,
                // since this is the hottest loop.
                unsafe {
                    *self.scratch.x.get_unchecked_mut(index) =
                        (tile.corner[0] + i) as f32;
                    *self.scratch.y.get_unchecked_mut(index) =
                        (tile.corner[1] + j) as f32;
                    *self.scratch.z.get_unchecked_mut(index) =
                        (tile.corner[2] + k) as f32;
                }
                index += 1;
            }
            self.scratch.columns.push(xy);
        }
        let size = index;
        assert!(size > 0);

        fidget_core::shell::reset_profile2d_outer_distance_batch_calls();
        let float_started = Instant::now();
        let out = self
            .eval_float_slice
            .eval_with_transform_and_vars(
                shape.f_tape(&mut self.tape_storage),
                &self.scratch.x[..index],
                &self.scratch.y[..index],
                &self.scratch.z[..index],
                &self.transform,
                self.vars,
            )
            .unwrap();
        self.stats.float_eval_time += float_started.elapsed();
        self.stats.float_eval_calls += 1;
        self.stats.float_eval_samples += size as u64;
        let outer_distance_calls =
            fidget_core::shell::profile2d_outer_distance_batch_calls();
        if outer_distance_calls != 0 {
            self.stats.shell_hull_profile2d_outer_distance_batches += 1;
            self.stats.shell_hull_profile2d_outer_distance_batch_samples +=
                size as u64;
            self.stats
                .shell_hull_profile2d_outer_distance_max_batch_calls = self
                .stats
                .shell_hull_profile2d_outer_distance_max_batch_calls
                .max(outer_distance_calls);
        }

        // We're iterating over a few things simultaneously
        // - col refers to the xy position in the tile
        // - grad refers to points that we must do gradient evaluation on
        let mut grad = 0;
        let mut depth = out.chunks(tile_size);
        for col in 0..self.scratch.columns.len() {
            // Find the first set pixel in the column
            let depth = depth.next().unwrap();
            let k = match depth.iter().enumerate().find(|(_, d)| **d < 0.0) {
                Some((i, _)) => i,
                None => continue,
            };

            // Get X and Y values from the `columns` array.  Note that we can't
            // iterate over the array directly because we're also modifying it
            // (below)
            let xy = self.scratch.columns[col];
            let i = xy % tile_size;
            let j = xy / tile_size;

            // Flip Z value, since voxels are packed front-to-back
            let k = tile_size - 1 - k;

            // Set the depth of the pixel
            let o = self.tile_sizes.pixel_offset(tile.add(Vector2::new(i, j)));
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

            // This can only be called once per iteration, so we'll
            // never overwrite parts of columns that are still used
            // by the outer loop
            self.scratch.columns[grad] = o;
            grad += 1;
        }

        if grad > 0 {
            let grad_started = Instant::now();
            let out = self
                .eval_grad_slice
                .eval_with_transform_and_vars(
                    shape.g_tape(&mut self.tape_storage),
                    &self.scratch.xg[..grad],
                    &self.scratch.yg[..grad],
                    &self.scratch.zg[..grad],
                    &self.transform,
                    self.vars,
                )
                .unwrap();
            self.stats.grad_eval_time += grad_started.elapsed();
            self.stats.grad_eval_calls += 1;
            self.stats.grad_eval_samples += grad as u64;

            for (index, o) in self.scratch.columns[0..grad].iter().enumerate() {
                let g = out[index];
                self.out[*o].normal = [g.dx, g.dy, g.dz];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Renders the given shape into a 3D image with a particular configuration
/// configuration.
///
/// The shape provides the evaluator backend (`F`) and bound variables; the
/// configuration supplies resolution, transforms, etc.
///
/// Returns [`Some(Image)`](Image) of pixel data on success, or `None` if
/// the render was cancelled.
pub fn render<F: Function + RenderHints>(
    b: BoundShape<F, f32>,
    config: &RenderConfig,
) -> Option<Image> {
    render_inner(b, config, false).map(|(image, _stats)| image)
}

/// Renders the given shape into a 3D image and returns compatibility stats.
pub fn render_with_stats<F: Function + RenderHints>(
    b: BoundShape<F, f32>,
    config: &RenderConfig,
) -> Option<(Image, VoxelRenderStats)> {
    render_inner(b, config, true)
}

fn render_inner<F: Function + RenderHints>(
    b: BoundShape<F, f32>,
    config: &RenderConfig,
    collect_shell_stats: bool,
) -> Option<(Image, VoxelRenderStats)> {
    if collect_shell_stats {
        fidget_core::shell::set_shell_eval_stats_enabled(true);
        fidget_core::shell::reset_shell_eval_stats();
    }
    let shape = b.shape().clone();
    let vars = b.vars();
    let max_size = config.width().max(config.height()) as usize;
    let default_tile_sizes;

    let tile_sizes = if let Some(ts) = &config.tile_sizes {
        TileSizesRef::new(ts, max_size)
    } else {
        default_tile_sizes = F::tile_sizes_3d();
        TileSizesRef::new(&default_tile_sizes, max_size)
    };
    let Some(tiles) =
        super::render_tiles::<F, Worker<F>>(shape, vars, config, tile_sizes)
    else {
        if collect_shell_stats {
            fidget_core::shell::set_shell_eval_stats_enabled(false);
        }
        return None;
    };

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image = Image::new(config.image_size);
    let mut stats = VoxelRenderStats::default();
    for (tile, (out, tile_stats)) in tiles {
        stats.merge_worker(tile_stats);
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
    if collect_shell_stats {
        stats.merge_shell(fidget_core::shell::shell_eval_stats());
        fidget_core::shell::set_shell_eval_stats_enabled(false);
    }
    Some((image, stats))
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{Context, var::Var, vm::VmShape};

    /// Make sure we don't crash if there's only a single tile
    #[test]
    fn test_tile_queues() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::from(128), // very small!
            ..Default::default()
        };
        let image = cfg.run(shape).expect("rendering should not be cancelled");
        assert_eq!(image.len(), 128 * 128);
    }

    #[test]
    fn cancel_render() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::new(64, 64, 64),
            ..Default::default()
        };
        let cancel = cfg.cancel.clone();
        cancel.cancel();
        assert!(cfg.run::<_>(shape).is_none());
    }

    #[test]
    fn shape_with_var() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let var = Var::new();
        let v = ctx.var(var);
        let s = ctx.sub(x, v).unwrap();
        let shape = VmShape::new(&ctx, s).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::new(64, 64, 64),
            ..Default::default()
        };

        let mut vars = ShapeVars::new();
        let i = var.index().expect("expected Var::V");
        vars.insert(i, 1.0);
        cfg.run::<_>(shape.bind(&vars).expect("all vars present"))
            .expect("not cancelled");
    }
}
