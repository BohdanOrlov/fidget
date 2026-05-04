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
    types::{Grad, Interval},
};

use nalgebra::{Point3, Vector2, Vector3};
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
    image_size: VoxelSize,

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
    stats: Render3dStats,
}

#[derive(Clone, Copy, Default)]
struct Render3dStats {
    total_tile_time: Duration,
    interval_eval_time: Duration,
    simplify_time: Duration,
    float_eval_time: Duration,
    grad_eval_time: Duration,
    interval_eval_calls: u64,
    simplify_calls: u64,
    float_eval_calls: u64,
    grad_eval_calls: u64,
    float_eval_samples: u64,
    grad_eval_samples: u64,
}

impl Render3dStats {
    fn merge(&mut self, other: Render3dStats) {
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
    }
}

struct TileRenderOutput {
    image: GeometryBuffer,
    stats: Render3dStats,
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
    stats: Render3dStats,
}

struct TileDebugRenderOutput {
    image: LeafDebugBuffer,
    stats: Render3dStats,
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

            eval_float_slice: Default::default(),
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
        self.stats = Render3dStats::default();
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
        self.stats = Render3dStats::default();
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

        let start = Instant::now();
        let out = self
            .eval_float_slice
            .eval_v(
                shape.f_tape(&mut self.tape_storage),
                &self.scratch.x[..index],
                &self.scratch.y[..index],
                &self.scratch.z[..index],
                vars,
            )
            .unwrap();
        self.stats.float_eval_time += start.elapsed();
        self.stats.float_eval_calls += 1;
        self.stats.float_eval_samples += size as u64;

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

            for (index, o) in self.scratch.columns[0..grad].iter().enumerate() {
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
                    if i.upper() < 0.0 || i.lower() > 0.0 {
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
    let shape = shape.with_transform(config.mat());

    let tiles = super::render_tiles::<F, Worker<F>, _>(shape, vars, config)?;
    let tile_sizes = config.tile_sizes();

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image = GeometryBuffer::new(config.image_size);
    let mut stats = Render3dStats::default();
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

    if std::env::var_os("FIDGET_RENDER3D_STATS").is_some() {
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
    }
    Some(image)
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

    let tiles =
        super::render_tiles::<F, DebugWorker<F>, _>(shape, vars, config)?;
    let tile_sizes = config.tile_sizes();

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image = LeafDebugBuffer::new(config.image_size);
    let mut stats = Render3dStats::default();
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

    if std::env::var_os("FIDGET_RENDER3D_STATS").is_some() {
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
    }

    Some(image)
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{Context, render::VoxelSize, vm::VmShape};

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
}
