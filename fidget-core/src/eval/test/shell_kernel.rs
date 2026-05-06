use crate::shell::{
    OpenTopPolicy, SHELL_MAX_CANDIDATES, ShellEvalScratch, ShellParamsView,
    ShellProfileNodeContinuity, ShellProfileNodeTopology,
    ShellProfileSectionTopology, ShellProfileSpanInterpolation,
    ShellSectionTopology, ShellTopology, eval_shell_distance,
    reset_shell_eval_stats, set_shell_eval_stats_enabled, shell_eval_stats,
};

fn equal_circle_loft() -> ShellTopology {
    ShellTopology::line_loft_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
    )
}

fn sample(topology: &ShellTopology, x: f32, y: f32, z: f32) -> f32 {
    let mut scratch = ShellEvalScratch::default();
    eval_shell_distance(
        topology,
        ShellParamsView::empty(),
        &mut scratch,
        x,
        y,
        z,
    )
    .distance
}

#[test]
fn line_loft_between_equal_circles_matches_cylinder_distance() {
    let topology = equal_circle_loft();

    assert_approx_eq(sample(&topology, 1.0, 0.0, 0.0), -1.0);
    assert_approx_eq(sample(&topology, 1.0, 1.0, 0.0), 0.0);
    assert_approx_eq(sample(&topology, 1.0, 1.25, 0.0), 0.25);
    assert_approx_eq(sample(&topology, -0.25, 0.0, 0.0), 0.25);
    assert_approx_eq(sample(&topology, 2.25, 0.0, 0.0), 0.25);
}

#[test]
fn line_loft_between_scaled_circles_interpolates_radius() {
    let topology = ShellTopology::line_loft_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 0.5),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.5),
        ]
        .into_boxed_slice(),
    );

    assert_approx_eq(sample(&topology, 1.0, 1.0, 0.0), 0.0);
    assert_approx_eq(sample(&topology, 1.0, 0.25, 0.0), -0.75);
    assert_approx_eq(sample(&topology, 1.0, 1.25, 0.0), 0.25);
}

#[test]
fn curve_loft_uses_precomputed_cubic_coefficients() {
    let topology = ShellTopology::curve_loft_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 1.0, 0.0, 1.0),
            ShellSectionTopology::circle(4.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
    );

    // With clamped Catmull-Rom endpoints, segment 0 center_y at t=0.5 is
    // 0.5625 rather than the linear value 0.5.
    assert_approx_eq(sample(&topology, 1.0, 0.5625, 0.0), -1.0);
    assert_approx_eq(sample(&topology, 1.0, 1.5625, 0.0), 0.0);
}

#[test]
fn closed_shell_distance_changes_with_thickness() {
    let thin = ShellTopology::shell_hull_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
        0.10,
        OpenTopPolicy::Closed,
    );
    let thick = ShellTopology::shell_hull_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
        0.25,
        OpenTopPolicy::Closed,
    );

    assert_approx_eq(sample(&thin, 1.0, 0.95, 0.0), -0.05);
    assert_approx_eq(sample(&thin, 1.0, 0.80, 0.0), 0.10);
    assert_approx_eq(sample(&thick, 1.0, 0.80, 0.0), -0.05);
}

#[test]
fn open_top_samples_above_deck_opening_are_outside() {
    let topology = ShellTopology::shell_hull_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
        0.25,
        OpenTopPolicy::BoxCut {
            cut_z: 0.2,
            half_length: 0.75,
            half_width: 0.75,
            offset_x: 1.0,
        },
    );

    assert!(sample(&topology, 1.0, 0.0, 0.95) > 0.0);
    assert!(sample(&topology, 1.0, 0.0, -0.95) < 0.0);
}

#[test]
fn ship_profile_shell_uses_authored_v_section_instead_of_circle() {
    let topology = ShellTopology::ship_profile_shell_hull(
        vec![
            ShellProfileSectionTopology::ship(0.0, -0.5, 0.2, 0.5),
            ShellProfileSectionTopology::ship(2.0, -0.5, 0.2, 0.5),
        ]
        .into_boxed_slice(),
        0.10,
        OpenTopPolicy::Closed,
    );

    assert!(sample(&topology, 1.0, 0.0, -0.50).abs() < 1.0e-4);
    assert!(sample(&topology, 1.0, 0.0, -0.30) > 0.0);
    assert!(sample(&topology, 1.0, 0.46, 0.20).abs() < 0.05);
    assert!(sample(&topology, 1.0, 0.65, 0.00) > 0.0);
}

#[test]
fn station_curve_profile_accepts_two_node_minimum_section() {
    let section = ShellProfileSectionTopology::station_curve(
        0.0,
        &[
            ShellProfileNodeTopology::new(
                0.0,
                -0.5,
                ShellProfileNodeContinuity::Linear,
            ),
            ShellProfileNodeTopology::new(
                0.4,
                0.2,
                ShellProfileNodeContinuity::Linear,
            ),
        ],
    );
    let topology = ShellTopology::ship_profile_shell_hull(
        vec![
            section,
            ShellProfileSectionTopology {
                station: 2.0,
                ..section
            },
        ]
        .into_boxed_slice(),
        1.0,
        OpenTopPolicy::Closed,
    );

    assert_eq!(topology.profile.as_ref().unwrap().sections[0].node_count, 2);
    assert!(sample(&topology, 1.0, 0.18, -0.15) < 0.0);
}

#[test]
fn smooth_profile_nodes_fair_the_section_beyond_linear_edges() {
    let linear =
        profile_continuity_test_topology(ShellProfileNodeContinuity::Linear);
    let smooth =
        profile_continuity_test_topology(ShellProfileNodeContinuity::Smooth);

    let linear_distance = sample(&linear, 1.0, 0.34, 0.50);
    let smooth_distance = sample(&smooth, 1.0, 0.34, 0.50);

    assert!(
        linear_distance > 0.0,
        "linear edge should keep this point outside; got {linear_distance}"
    );
    assert!(
        smooth_distance < 0.0,
        "smooth edge should fair outward and include this point; got {smooth_distance}"
    );
}

#[test]
fn station_profile_distance_evaluates_whole_section_edges() {
    const SAMPLE_COUNT: usize = 100;
    let topology =
        profile_normal_rib_test_topology(ShellProfileNodeContinuity::Smooth);
    let _stats_guard = super::SHELL_EVAL_STATS_TEST_LOCK.lock().unwrap();

    set_shell_eval_stats_enabled(true);
    reset_shell_eval_stats();
    let mut distance = 0.0;
    for _ in 0..SAMPLE_COUNT {
        distance = sample(&topology, 1.0, 0.50, 0.05);
    }
    let stats = shell_eval_stats();
    set_shell_eval_stats_enabled(false);

    assert!(distance.is_finite());
    assert!(
        stats.profile2d_segment_tests >= (SAMPLE_COUNT as u64) * 4,
        "four-node station profiles should test all three spans plus deck edge per sample, got {} segment tests",
        stats.profile2d_segment_tests
    );
}

#[test]
fn smooth_profile_nodes_keep_normals_fairer_than_linear_chines() {
    let smooth =
        profile_normal_rib_test_topology(ShellProfileNodeContinuity::Smooth);
    let linear =
        profile_normal_rib_test_topology(ShellProfileNodeContinuity::Linear);

    let smooth_lower = numerical_normal(&smooth, 1.0, 0.60, -0.12);
    let smooth_upper = numerical_normal(&smooth, 1.0, 0.606, -0.08);
    let linear_lower = numerical_normal(&linear, 1.0, 0.60, -0.12);
    let linear_upper = numerical_normal(&linear, 1.0, 0.606, -0.08);
    let smooth_dot = dot3(smooth_lower, smooth_upper);
    let linear_dot = dot3(linear_lower, linear_upper);

    assert!(
        smooth_dot > linear_dot + 0.35,
        "smooth profile nodes should be visibly fairer than explicit linear chines, smooth={smooth_dot}, linear={linear_dot}"
    );
    assert!(
        linear_dot < 0.75,
        "explicit linear chines should keep a visible normal break, got {linear_dot}"
    );
}

#[test]
fn smooth_profile_nodes_share_tangents_without_normal_rib() {
    let smooth =
        profile_normal_rib_test_topology(ShellProfileNodeContinuity::Smooth);
    let linear =
        profile_normal_rib_test_topology(ShellProfileNodeContinuity::Linear);

    let smooth_lower = numerical_normal(&smooth, 1.0, 0.635, -0.112);
    let smooth_upper = numerical_normal(&smooth, 1.0, 0.635, -0.088);
    let linear_lower = numerical_normal(&linear, 1.0, 0.635, -0.112);
    let linear_upper = numerical_normal(&linear, 1.0, 0.635, -0.088);
    let smooth_dot = dot3(smooth_lower, smooth_upper);
    let linear_dot = dot3(linear_lower, linear_upper);

    assert!(
        smooth_dot > 0.98,
        "smooth profile nodes should share a tangent through width extrema, got {smooth_dot}"
    );
    assert!(
        linear_dot < 0.75,
        "explicit linear profile nodes should preserve the hard normal break, got {linear_dot}"
    );
}

#[test]
fn built_in_ship_profile_uses_whole_station_curve_evaluator() {
    let fast_sections = vec![
        ShellProfileSectionTopology::ship(0.0, -0.5, 0.2, 0.5),
        ShellProfileSectionTopology::ship(2.0, -0.5, 0.2, 0.5),
    ];
    let mut generic_sections = fast_sections.clone();
    for section in &mut generic_sections {
        section.ship_fast_path = false;
    }

    let built_in = ShellTopology::ship_profile_shell_hull(
        fast_sections.into_boxed_slice(),
        0.0,
        OpenTopPolicy::Closed,
    );
    let generic = ShellTopology::ship_profile_shell_hull(
        generic_sections.into_boxed_slice(),
        0.0,
        OpenTopPolicy::Closed,
    );

    for (x, y, z) in [(1.0, 0.30, -0.05), (1.0, 0.48, 0.10), (1.0, 0.05, -0.40)]
    {
        assert_approx_eq(sample(&built_in, x, y, z), sample(&generic, x, y, z));
    }
}
#[test]
fn station_span_interpolation_mode_is_preserved_in_profile_topology() {
    let sections = [0.0, 0.4, 0.8, 1.2]
        .into_iter()
        .enumerate()
        .map(|(index, station)| {
            let beam = if index % 2 == 0 { 0.25 } else { 0.55 };
            let span = if index == 1 {
                ShellProfileSpanInterpolation::Linear
            } else {
                ShellProfileSpanInterpolation::SmoothCatmullRom
            };
            ShellProfileSectionTopology::station_curve_with_span(
                station,
                &[
                    ShellProfileNodeTopology::new(
                        0.0,
                        -0.3,
                        ShellProfileNodeContinuity::Linear,
                    ),
                    ShellProfileNodeTopology::new(
                        beam,
                        0.0,
                        ShellProfileNodeContinuity::Smooth,
                    ),
                    ShellProfileNodeTopology::new(
                        beam * 0.8,
                        0.3,
                        ShellProfileNodeContinuity::Linear,
                    ),
                ],
                span,
            )
        })
        .collect::<Vec<_>>();
    let topology = ShellTopology::ship_profile_shell_hull(
        sections.into_boxed_slice(),
        0.2,
        OpenTopPolicy::Closed,
    );
    let profile = topology.profile.as_ref().unwrap();

    assert_eq!(
        profile.segments[0].interpolation,
        ShellProfileSpanInterpolation::SmoothCatmullRom
    );
    assert_eq!(
        profile.segments[1].interpolation,
        ShellProfileSpanInterpolation::Linear
    );
}

#[test]
fn smooth_station_span_is_distance_and_normal_continuous_across_station() {
    let topology =
        seam_test_topology(ShellProfileSpanInterpolation::SmoothCatmullRom);
    let left_x = 1.0 - 0.002;
    let right_x = 1.0 + 0.002;
    let point_y = 0.40;
    let point_z = 0.0;

    let left_distance = sample(&topology, left_x, point_y, point_z);
    let right_distance = sample(&topology, right_x, point_y, point_z);
    let left_normal = numerical_normal(&topology, left_x, point_y, point_z);
    let right_normal = numerical_normal(&topology, right_x, point_y, point_z);

    assert!(
        (left_distance - right_distance).abs() < 0.006,
        "smooth station seam should not create a distance jump, left={left_distance}, right={right_distance}"
    );
    assert!(
        dot3(left_normal, right_normal) > 0.96,
        "smooth station seam normals should remain fair, left={left_normal:?}, right={right_normal:?}"
    );
}

#[test]
fn linear_station_span_boundary_remains_finite_and_explicitly_linear() {
    let topology = seam_test_topology(ShellProfileSpanInterpolation::Linear);
    let profile = topology
        .profile
        .as_ref()
        .expect("profile topology should exist");
    assert_eq!(
        profile.segments[1].interpolation,
        ShellProfileSpanInterpolation::Linear
    );

    for x in [0.998, 1.0, 1.002] {
        let distance = sample(&topology, x, 0.40, 0.0);
        let normal = numerical_normal(&topology, x, 0.40, 0.0);
        assert!(distance.is_finite(), "linear seam distance must be finite");
        assert!(
            normal.iter().all(|component| component.is_finite()),
            "linear seam normal must be finite, got {normal:?}"
        );
    }
}

fn profile_continuity_test_topology(
    continuity: ShellProfileNodeContinuity,
) -> ShellTopology {
    let nodes = [
        ShellProfileNodeTopology::new(
            0.0,
            -1.0,
            ShellProfileNodeContinuity::Linear,
        ),
        ShellProfileNodeTopology::new(0.4, 0.0, continuity),
        ShellProfileNodeTopology::new(0.2, 1.0, continuity),
    ];
    let section = ShellProfileSectionTopology::station_curve(0.0, &nodes);
    ShellTopology::ship_profile_shell_hull(
        vec![
            section,
            ShellProfileSectionTopology {
                station: 2.0,
                ..section
            },
        ]
        .into_boxed_slice(),
        1.0,
        OpenTopPolicy::Closed,
    )
}

fn profile_normal_rib_test_topology(
    continuity: ShellProfileNodeContinuity,
) -> ShellTopology {
    let nodes = [
        ShellProfileNodeTopology::new(0.08, -0.60, continuity),
        ShellProfileNodeTopology::new(0.62, -0.10, continuity),
        ShellProfileNodeTopology::new(0.28, 0.35, continuity),
        ShellProfileNodeTopology::new(0.20, 0.70, continuity),
    ];
    let section = ShellProfileSectionTopology::station_curve(0.0, &nodes);
    ShellTopology::ship_profile_shell_hull(
        vec![
            section,
            ShellProfileSectionTopology {
                station: 2.0,
                ..section
            },
        ]
        .into_boxed_slice(),
        0.0,
        OpenTopPolicy::Closed,
    )
}

fn seam_test_topology(span: ShellProfileSpanInterpolation) -> ShellTopology {
    let mut sections = Vec::new();
    for (station, beam) in [(0.0, 0.24), (1.0, 0.52), (2.0, 0.28), (3.0, 0.36)]
    {
        sections.push(ShellProfileSectionTopology::station_curve_with_span(
            station,
            &[
                ShellProfileNodeTopology::new(
                    0.0,
                    -0.55,
                    ShellProfileNodeContinuity::Linear,
                ),
                ShellProfileNodeTopology::new(
                    beam,
                    -0.08,
                    ShellProfileNodeContinuity::Smooth,
                ),
                ShellProfileNodeTopology::new(
                    beam * 0.82,
                    0.24,
                    ShellProfileNodeContinuity::Smooth,
                ),
                ShellProfileNodeTopology::new(
                    beam * 0.55,
                    0.34,
                    ShellProfileNodeContinuity::Linear,
                ),
            ],
            span,
        ));
    }

    ShellTopology::ship_profile_shell_hull(
        sections.into_boxed_slice(),
        0.28,
        OpenTopPolicy::Closed,
    )
}

fn numerical_normal(
    topology: &ShellTopology,
    x: f32,
    y: f32,
    z: f32,
) -> [f32; 3] {
    let h = 1.0e-3;
    normalize3([
        sample(topology, x + h, y, z) - sample(topology, x - h, y, z),
        sample(topology, x, y + h, z) - sample(topology, x, y - h, z),
        sample(topology, x, y, z + h) - sample(topology, x, y, z - h),
    ])
}

fn normalize3(vector: [f32; 3]) -> [f32; 3] {
    let len =
        (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
            .sqrt();
    if !len.is_finite() || len <= 1.0e-6 {
        [0.0, 0.0, 1.0]
    } else {
        [vector[0] / len, vector[1] / len, vector[2] / len]
    }
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[test]
fn evaluator_uses_fixed_candidate_scratch() {
    let topology = equal_circle_loft();
    let mut scratch = ShellEvalScratch::default();
    let sample = eval_shell_distance(
        &topology,
        ShellParamsView::empty(),
        &mut scratch,
        1.0,
        0.25,
        0.0,
    );

    assert_approx_eq(sample.distance, -0.75);
    assert_eq!(scratch.closest_segment(), Some(0));
    assert_eq!(scratch.candidate_count(), topology.segments.len());
    assert_eq!(scratch.candidate_capacity(), SHELL_MAX_CANDIDATES);
}

#[test]
fn kernel_outputs_are_finite_for_representative_grid() {
    let topology = equal_circle_loft();

    for ix in -4..=4 {
        for iy in -4..=4 {
            for iz in -4..=4 {
                let distance = sample(
                    &topology,
                    ix as f32 * 0.4,
                    iy as f32 * 0.4,
                    iz as f32 * 0.4,
                );
                assert!(distance.is_finite());
            }
        }
    }
}

fn assert_approx_eq(left: f32, right: f32) {
    let diff = (left - right).abs();
    assert!(
        diff <= 1.0e-5,
        "expected {left} to be within tolerance of {right}; diff={diff}"
    );
}
