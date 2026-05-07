//! Conservative shell interval bounds.

use crate::types::Interval;

use super::{
    ShellParamsView, ShellProfileTopology, ShellSegmentTopology, ShellTopology,
    topology::{
        SHELL_MAX_NODES_PER_CURVE, ShellCubicCoefficients, ShellOpKind,
        ShellProfileSegmentTopology, ShellProfileSpanInterpolation,
        ShellSegmentInterpolation,
    },
};

/// Conservative native shell interval trace data.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShellIntervalTrace {
    /// Bitmask of segments that may affect the tile.
    pub active_segment_mask: u64,
    /// Total number of shell segments represented by the mask.
    pub segment_count: usize,
    /// Whether this trace is currently safe/profitable to use for reduced
    /// sidecar simplification in render traversal.
    pub sidecar_reduction_eligible: bool,
}

/// Evaluates a conservative native shell interval over an axis-aligned tile.
///
/// The first pruning tier uses the topology's global bounds.  If the tile is
/// completely outside the shell envelope, the returned interval is strictly
/// positive.  Otherwise this deliberately returns a broad interval until the
/// segment-range trace path is implemented.
pub fn eval_shell_interval(
    topology: &ShellTopology,
    x: Interval,
    y: Interval,
    z: Interval,
) -> Interval {
    super::eval::record_shell_interval_call();

    if x.has_nan() || y.has_nan() || z.has_nan() {
        return f32::NAN.into();
    }

    if let Some(profile) = topology.profile.as_ref() {
        if let Some(gap) = outside_profile_segment_gap(
            profile,
            topology.shell_thickness,
            x,
            y,
            z,
        ) {
            return Interval::new(gap, f32::INFINITY);
        }
    } else if topology.kind == ShellOpKind::ShellHull
        && let Some(interval) =
            eval_shell_hull_positive_interval(topology, x, y, z)
    {
        return interval;
    }

    if let Some(gap) = outside_segment_gap(topology, x, y, z) {
        return Interval::new(gap, f32::INFINITY);
    }

    Interval::new(f32::NEG_INFINITY, f32::INFINITY)
}

/// Evaluates a native shell interval and records conservative active segments.
pub fn eval_shell_interval_with_trace(
    topology: &ShellTopology,
    x: Interval,
    y: Interval,
    z: Interval,
) -> (Interval, ShellIntervalTrace) {
    super::eval::record_shell_interval_call();
    let mut trace = ShellIntervalTrace {
        active_segment_mask: all_segments_mask(topology.segments.len()),
        segment_count: topology.segments.len(),
        sidecar_reduction_eligible: topology.profile.is_none(),
    };

    if x.has_nan() || y.has_nan() || z.has_nan() {
        return (f32::NAN.into(), trace);
    }

    if let Some(profile) = topology.profile.as_ref() {
        let profile_trace = profile_active_segment_trace(
            profile,
            topology.shell_thickness,
            x,
            y,
            z,
        );
        trace = ShellIntervalTrace {
            active_segment_mask: profile_trace.active_segment_mask,
            segment_count: topology.segments.len(),
            sidecar_reduction_eligible: false,
        };
        if profile_trace.active_segment_mask == 0 {
            return (
                Interval::new(profile_trace.best_gap, f32::INFINITY),
                trace,
            );
        }
    } else if topology.kind == ShellOpKind::ShellHull
        && let Some(interval) =
            eval_shell_hull_positive_interval(topology, x, y, z)
    {
        trace.active_segment_mask = 0;
        return (interval, trace);
    }

    let segment_trace = active_segment_trace(topology, x, y, z);
    if topology.profile.is_none() {
        trace = ShellIntervalTrace {
            active_segment_mask: segment_trace.active_segment_mask,
            segment_count: topology.segments.len(),
            sidecar_reduction_eligible: true,
        };
    }
    if segment_trace.active_segment_mask == 0 {
        return (Interval::new(segment_trace.best_gap, f32::INFINITY), trace);
    }

    (Interval::new(f32::NEG_INFINITY, f32::INFINITY), trace)
}

#[derive(Clone, Copy)]
struct ActiveSegmentTrace {
    active_segment_mask: u64,
    best_gap: f32,
}

fn all_segments_mask(segment_count: usize) -> u64 {
    if segment_count >= u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << segment_count) - 1
    }
}

fn segment_mask(index: usize) -> u64 {
    if index >= u64::BITS as usize {
        0
    } else {
        1_u64 << index
    }
}

fn profile_active_segment_trace(
    profile: &ShellProfileTopology,
    _shell_thickness: f32,
    x: Interval,
    y: Interval,
    z: Interval,
) -> ActiveSegmentTrace {
    let mut active_segment_mask = 0_u64;
    let mut best_gap = f32::INFINITY;
    let padding = 1.0e-4;
    for (index, segment) in profile.segments.iter().copied().enumerate() {
        let left = profile.sections[segment.left_section];
        let right = profile.sections[segment.right_section];
        let bow_extension = if segment.left_section == 0 {
            profile.bow_cap_extension
        } else {
            0.0
        };
        let stern_extension =
            if segment.right_section + 1 == profile.sections.len() {
                profile.stern_cap_extension
            } else {
                0.0
            };
        let min_x = left.station.min(right.station) - bow_extension - padding;
        let max_x = left.station.max(right.station) + stern_extension + padding;
        let (keel_min, keel_max) = coeff_range(segment.keel_z, 0.0, 1.0);
        let (sheer_min, sheer_max) = coeff_range(segment.sheer_z, 0.0, 1.0);
        let (beam_min, beam_max) = coeff_range(segment.beam, 0.0, 1.0);
        let half_width = beam_min
            .abs()
            .max(beam_max.abs())
            .max(left.beam.abs())
            .max(right.beam.abs())
            * 1.02
            + padding;
        let min_z = keel_min.min(sheer_min) - padding;
        let max_z = keel_max.max(sheer_max) + padding;
        let half_width =
            if segment.ship_fast_path && axis_gap(z, min_z, max_z) == 0.0 {
                half_width.min(
                    profile_segment_half_width_for_z_interval(
                        profile, segment, x, z, half_width, padding,
                    )
                    .unwrap_or(half_width),
                )
            } else {
                half_width
            };

        let dx = axis_gap(x, min_x, max_x);
        let dy = axis_gap(y, -half_width, half_width);
        let dz = axis_gap(z, min_z, max_z);
        let gap = dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt();
        if gap == 0.0 {
            active_segment_mask |= segment_mask(index);
        }
        best_gap = best_gap.min(gap);
    }

    ActiveSegmentTrace {
        active_segment_mask,
        best_gap,
    }
}

fn active_segment_trace(
    topology: &ShellTopology,
    x: Interval,
    y: Interval,
    z: Interval,
) -> ActiveSegmentTrace {
    let params = ShellParamsView::empty();
    let mut active_segment_mask = 0_u64;
    let mut best_gap = f32::INFINITY;
    for (index, segment) in topology.segments.iter().copied().enumerate() {
        let left = topology.sections[segment.left_section];
        let right = topology.sections[segment.right_section];
        let left_x = left.station(params);
        let right_x = right.station(params);
        let (left_y, left_z) = left.center(params);
        let (right_y, right_z) = right.center(params);
        let radius = left.radius(params).max(right.radius(params))
            + topology.shell_thickness.max(0.0);

        let min_x = left_x.min(right_x);
        let max_x = left_x.max(right_x);
        let min_y = left_y.min(right_y) - radius;
        let max_y = left_y.max(right_y) + radius;
        let min_z = left_z.min(right_z) - radius;
        let max_z = left_z.max(right_z) + radius;

        let dx = axis_gap(x, min_x, max_x);
        let dy = axis_gap(y, min_y, max_y);
        let dz = axis_gap(z, min_z, max_z);
        let gap = dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt();
        if gap == 0.0 {
            active_segment_mask |= segment_mask(index);
        }
        best_gap = best_gap.min(gap);
    }

    ActiveSegmentTrace {
        active_segment_mask,
        best_gap,
    }
}

fn outside_profile_segment_gap(
    profile: &ShellProfileTopology,
    _shell_thickness: f32,
    x: Interval,
    y: Interval,
    z: Interval,
) -> Option<f32> {
    let mut best_gap = f32::INFINITY;
    let padding = 1.0e-4;
    for segment in profile.segments.iter().copied() {
        let left = profile.sections[segment.left_section];
        let right = profile.sections[segment.right_section];
        let bow_extension = if segment.left_section == 0 {
            profile.bow_cap_extension
        } else {
            0.0
        };
        let stern_extension =
            if segment.right_section + 1 == profile.sections.len() {
                profile.stern_cap_extension
            } else {
                0.0
            };
        let min_x = left.station.min(right.station) - bow_extension - padding;
        let max_x = left.station.max(right.station) + stern_extension + padding;
        let (keel_min, keel_max) = coeff_range(segment.keel_z, 0.0, 1.0);
        let (sheer_min, sheer_max) = coeff_range(segment.sheer_z, 0.0, 1.0);
        let (beam_min, beam_max) = coeff_range(segment.beam, 0.0, 1.0);
        let half_width = beam_min
            .abs()
            .max(beam_max.abs())
            .max(left.beam.abs())
            .max(right.beam.abs())
            * 1.02
            + padding;
        let min_z = keel_min.min(sheer_min) - padding;
        let max_z = keel_max.max(sheer_max) + padding;
        let half_width =
            if segment.ship_fast_path && axis_gap(z, min_z, max_z) == 0.0 {
                half_width.min(
                    profile_segment_half_width_for_z_interval(
                        profile, segment, x, z, half_width, padding,
                    )
                    .unwrap_or(half_width),
                )
            } else {
                half_width
            };

        let dx = axis_gap(x, min_x, max_x);
        let dy = axis_gap(y, -half_width, half_width);
        let dz = axis_gap(z, min_z, max_z);
        let gap = dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt();
        if gap == 0.0 {
            return None;
        }
        best_gap = best_gap.min(gap);
    }

    best_gap.is_finite().then_some(best_gap)
}

#[derive(Clone, Copy)]
struct ProfileNodeRange {
    max_half_width: f32,
    min_z: f32,
    max_z: f32,
}

fn profile_segment_half_width_for_z_interval(
    profile: &ShellProfileTopology,
    segment: ShellProfileSegmentTopology,
    x: Interval,
    z: Interval,
    fallback_half_width: f32,
    padding: f32,
) -> Option<f32> {
    let left = profile.sections[segment.left_section];
    let right = profile.sections[segment.right_section];
    let (t0, t1) = segment_t_interval(left.station, right.station, x);
    let node_count = segment.node_count.clamp(2, SHELL_MAX_NODES_PER_CURVE);
    let mut nodes = [ProfileNodeRange {
        max_half_width: 0.0,
        min_z: 0.0,
        max_z: 0.0,
    }; SHELL_MAX_NODES_PER_CURVE];

    for (node_index, node) in nodes.iter_mut().enumerate().take(node_count) {
        let left_node = left.nodes[node_index.min(left.node_count - 1)];
        let right_node = right.nodes[node_index.min(right.node_count - 1)];
        let ((width_min, width_max), (z_min, z_max)) =
            match segment.interpolation {
                ShellProfileSpanInterpolation::Linear => {
                    let left_width = left_node.half_width.abs();
                    let right_width = right_node.half_width.abs();
                    let width_a = lerp(left_width, right_width, t0);
                    let width_b = lerp(left_width, right_width, t1);
                    let z_a = lerp(left_node.z, right_node.z, t0);
                    let z_b = lerp(left_node.z, right_node.z, t1);
                    (
                        (width_a.min(width_b), width_a.max(width_b)),
                        (z_a.min(z_b), z_a.max(z_b)),
                    )
                }
                ShellProfileSpanInterpolation::SmoothCatmullRom => {
                    let segment_node = segment.nodes[node_index];
                    (
                        coeff_range(segment_node.half_width, t0, t1),
                        coeff_range(segment_node.z, t0, t1),
                    )
                }
            };
        *node = ProfileNodeRange {
            max_half_width: width_min.abs().max(width_max.abs()),
            min_z: z_min,
            max_z: z_max,
        };
    }

    let active_nodes = &nodes[..node_count];
    let mut max_half_width = 0.0_f32;
    let mut found_overlap = false;
    for edge_index in 0..active_nodes.len() - 1 {
        let a = active_nodes[edge_index];
        let c = active_nodes[edge_index + 1];
        let min_z = a.min_z.min(c.min_z) - padding;
        let max_z = a.max_z.max(c.max_z) + padding;
        if axis_gap(z, min_z, max_z) > 0.0 {
            continue;
        }

        found_overlap = true;
        if edge_index == 0 {
            max_half_width =
                max_half_width.max(a.max_half_width).max(c.max_half_width);

            continue;
        }

        let local_start = edge_index.saturating_sub(1);
        let local_end = (edge_index + 2).min(active_nodes.len() - 1);
        for node in &active_nodes[local_start..=local_end] {
            max_half_width = max_half_width.max(node.max_half_width);
        }
    }

    if !found_overlap {
        return None;
    }

    Some((max_half_width * 1.02 + padding).min(fallback_half_width))
}

fn eval_shell_hull_positive_interval(
    topology: &ShellTopology,
    x: Interval,
    y: Interval,
    z: Interval,
) -> Option<Interval> {
    let params = ShellParamsView::empty();
    let half_thickness = topology.shell_thickness * 0.5;
    let mut lower = f32::INFINITY;

    for segment in topology.segments.iter().copied() {
        let solid = segment_solid_interval(topology, params, segment, x, y, z);
        let segment_lower = if solid.lower() > half_thickness {
            solid.lower() - half_thickness
        } else if solid.upper() < -half_thickness {
            -solid.upper() - half_thickness
        } else {
            return None;
        };
        lower = lower.min(segment_lower);
    }

    lower
        .is_finite()
        .then_some(Interval::new(lower, f32::INFINITY))
}

fn segment_solid_interval(
    topology: &ShellTopology,
    params: ShellParamsView<'_>,
    segment: ShellSegmentTopology,
    x: Interval,
    y: Interval,
    z: Interval,
) -> Interval {
    let left = topology.sections[segment.left_section];
    let right = topology.sections[segment.right_section];
    let left_x = left.station(params);
    let right_x = right.station(params);
    let (t0, t1) = segment_t_interval(left_x, right_x, x);
    let (left_y, left_z) = left.center(params);
    let (right_y, right_z) = right.center(params);
    let (center_y_min, center_y_max) = profile_value_range(
        segment,
        left_y,
        right_y,
        t0,
        t1,
        ProfileValue::CenterY,
    );
    let (center_z_min, center_z_max) = profile_value_range(
        segment,
        left_z,
        right_z,
        t0,
        t1,
        ProfileValue::CenterZ,
    );
    let (radius_min, radius_max) = profile_value_range(
        segment,
        left.radius(params),
        right.radius(params),
        t0,
        t1,
        ProfileValue::Radius,
    );
    let radius_min = radius_min.max(1.0e-5);
    let radius_max = radius_max.max(radius_min);

    let dy_min = axis_gap(y, center_y_min, center_y_max);
    let dz_min = axis_gap(z, center_z_min, center_z_max);
    let radial_lower =
        dy_min.mul_add(dy_min, dz_min * dz_min).sqrt() - radius_max;

    let dy_max = axis_max_gap(y, center_y_min, center_y_max);
    let dz_max = axis_max_gap(z, center_z_min, center_z_max);
    let radial_upper =
        dy_max.mul_add(dy_max, dz_max * dz_max).sqrt() - radius_min;

    let left_cap = Interval::new(left_x - x.upper(), left_x - x.lower());
    let right_cap = Interval::new(x.lower() - right_x, x.upper() - right_x);
    let x_cap = Interval::new(
        left_cap.lower().max(right_cap.lower()),
        left_cap.upper().max(right_cap.upper()),
    );

    Interval::new(
        radial_lower.max(x_cap.lower()),
        radial_upper.max(x_cap.upper()),
    )
}

fn segment_t_interval(left_x: f32, right_x: f32, x: Interval) -> (f32, f32) {
    let span = right_x - left_x;
    if span.abs() <= 1.0e-6 {
        return (0.0, 0.0);
    }
    let a = ((x.lower() - left_x) / span).clamp(0.0, 1.0);
    let b = ((x.upper() - left_x) / span).clamp(0.0, 1.0);
    (a.min(b), a.max(b))
}

#[derive(Clone, Copy)]
enum ProfileValue {
    CenterY,
    CenterZ,
    Radius,
}

fn profile_value_range(
    segment: ShellSegmentTopology,
    left: f32,
    right: f32,
    t0: f32,
    t1: f32,
    value: ProfileValue,
) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut include = |v: f32| {
        min = min.min(v);
        max = max.max(v);
    };

    match segment.interpolation {
        ShellSegmentInterpolation::Linear => {
            include(lerp(left, right, t0));
            include(lerp(left, right, t1));
        }
        ShellSegmentInterpolation::Cubic {
            center_y,
            center_z,
            radius,
        } => {
            let coeffs = match value {
                ProfileValue::CenterY => center_y,
                ProfileValue::CenterZ => center_z,
                ProfileValue::Radius => radius,
            };
            include(coeffs.eval(t0));
            include(coeffs.eval(t1));
            for t in cubic_extrema(coeffs.c1, coeffs.c2, coeffs.c3) {
                if (t0..=t1).contains(&t) {
                    include(coeffs.eval(t));
                }
            }
        }
    }

    (min, max)
}

fn coeff_range(coeffs: ShellCubicCoefficients, t0: f32, t1: f32) -> (f32, f32) {
    let mut min = coeffs.eval(t0).min(coeffs.eval(t1));
    let mut max = coeffs.eval(t0).max(coeffs.eval(t1));
    for t in cubic_extrema(coeffs.c1, coeffs.c2, coeffs.c3) {
        if t.is_finite() && (t0..=t1).contains(&t) {
            let value = coeffs.eval(t);
            min = min.min(value);
            max = max.max(value);
        }
    }
    (min, max)
}

fn cubic_extrema(c1: f32, c2: f32, c3: f32) -> [f32; 2] {
    let a = 3.0 * c3;
    let b = 2.0 * c2;
    let c = c1;
    if a.abs() <= 1.0e-6 {
        if b.abs() <= 1.0e-6 {
            return [f32::NAN, f32::NAN];
        }
        return [(-c / b).clamp(0.0, 1.0), f32::NAN];
    }

    let discriminant = b.mul_add(b, -4.0 * a * c);
    if discriminant < 0.0 {
        return [f32::NAN, f32::NAN];
    }
    let sqrt = discriminant.sqrt();
    [
        ((-b - sqrt) / (2.0 * a)).clamp(0.0, 1.0),
        ((-b + sqrt) / (2.0 * a)).clamp(0.0, 1.0),
    ]
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t
}

fn outside_segment_gap(
    topology: &ShellTopology,
    x: Interval,
    y: Interval,
    z: Interval,
) -> Option<f32> {
    let params = ShellParamsView::empty();
    let mut best_gap = f32::INFINITY;
    for segment in topology.segments.iter().copied() {
        let left = topology.sections[segment.left_section];
        let right = topology.sections[segment.right_section];
        let left_x = left.station(params);
        let right_x = right.station(params);
        let (left_y, left_z) = left.center(params);
        let (right_y, right_z) = right.center(params);
        let radius = left.radius(params).max(right.radius(params))
            + topology.shell_thickness.max(0.0);

        let min_x = left_x.min(right_x);
        let max_x = left_x.max(right_x);
        let min_y = left_y.min(right_y) - radius;
        let max_y = left_y.max(right_y) + radius;
        let min_z = left_z.min(right_z) - radius;
        let max_z = left_z.max(right_z) + radius;

        let dx = axis_gap(x, min_x, max_x);
        let dy = axis_gap(y, min_y, max_y);
        let dz = axis_gap(z, min_z, max_z);
        let gap = dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt();
        if gap == 0.0 {
            return None;
        }
        best_gap = best_gap.min(gap);
    }

    best_gap.is_finite().then_some(best_gap)
}

fn axis_gap(axis: Interval, min: f32, max: f32) -> f32 {
    if axis.upper() < min {
        min - axis.upper()
    } else if axis.lower() > max {
        axis.lower() - max
    } else {
        0.0
    }
}

fn axis_max_gap(axis: Interval, min: f32, max: f32) -> f32 {
    (axis.lower() - max).abs().max((axis.upper() - min).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell::{
        OpenTopPolicy, ShellProfileSectionTopology, ShellTopology,
    };

    #[test]
    fn profile_shell_interval_prunes_width_outside_low_keel_profile() {
        let topology = ShellTopology::ship_profile_shell_hull(
            [
                ShellProfileSectionTopology::ship(0.0, -0.5, 0.2, 0.5),
                ShellProfileSectionTopology::ship(2.0, -0.5, 0.2, 0.5),
            ],
            0.10,
            OpenTopPolicy::Closed,
        );

        let interval = eval_shell_interval(
            &topology,
            Interval::new(0.9, 1.1),
            Interval::new(0.40, 0.45),
            Interval::new(-0.50, -0.45),
        );

        assert!(
            interval.lower() > 0.0,
            "low keel tile is inside the coarse beam AABB but outside the actual narrow station profile; got {interval:?}",
        );
    }

    #[test]
    fn profile_shell_interval_uses_tight_monotone_edge_width() {
        let topology = ShellTopology::ship_profile_shell_hull(
            [
                ShellProfileSectionTopology::ship(0.0, -0.5, 0.2, 0.5),
                ShellProfileSectionTopology::ship(2.0, -0.5, 0.2, 0.5),
            ],
            0.10,
            OpenTopPolicy::Closed,
        );

        let interval = eval_shell_interval(
            &topology,
            Interval::new(0.9, 1.1),
            Interval::new(0.075, 0.085),
            Interval::new(-0.50, -0.47),
        );

        assert!(
            interval.lower() > 0.0,
            "monotone lower-profile edge should reject a tile outside the edge width without being widened by the next station node; got {interval:?}",
        );
    }

}
