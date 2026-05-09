//! Immutable shell topology sidecars.

use super::{ShellBounds, ShellParamLayout, ShellParamsView};

/// Maximum authored curve count supported by fixed shell scratch buffers.
pub const SHELL_MAX_CURVES: usize = 12;
/// Maximum node count per authored curve supported by fixed shell scratch buffers.
pub const SHELL_MAX_NODES_PER_CURVE: usize = 16;
/// Maximum segment candidates supported by fixed shell scratch buffers.
pub const SHELL_MAX_CANDIDATES: usize =
    SHELL_MAX_CURVES * (SHELL_MAX_NODES_PER_CURVE - 1);

/// Continuity control for a station profile node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShellProfileNodeContinuity {
    /// Connect through this node with a smooth fair curve when both endpoints
    /// of the local profile edge are smooth.
    Smooth,
    /// Preserve a linear edge or crease at this node.
    Linear,
}

/// Interpolation mode between two authored station curves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShellProfileSpanInterpolation {
    /// Smooth Catmull-Rom interpolation of matching station node coordinates.
    SmoothCatmullRom,
    /// Linear station-to-station interpolation for hard/transom-like spans.
    Linear,
}

/// One vertically ordered node in a station half-profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellProfileNodeTopology {
    /// Half-width from centerline at this node.
    pub half_width: f32,
    /// Vertical coordinate.
    pub z: f32,
    /// Profile continuity at this node.
    pub continuity: ShellProfileNodeContinuity,
}

impl ShellProfileNodeTopology {
    /// Builds a profile node in `(half_width, z)` coordinates.
    pub fn new(
        half_width: f32,
        z: f32,
        continuity: ShellProfileNodeContinuity,
    ) -> Self {
        Self {
            half_width: half_width.abs().max(0.0),
            z,
            continuity,
        }
    }
}

impl Default for ShellProfileNodeTopology {
    fn default() -> Self {
        Self {
            half_width: 0.0,
            z: 0.0,
            continuity: ShellProfileNodeContinuity::Linear,
        }
    }
}

/// Precomputed interpolation coefficients for one matching station node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellProfileSegmentNodeTopology {
    /// Cubic half-width coefficients.
    pub half_width: ShellCubicCoefficients,
    /// Cubic z coefficients.
    pub z: ShellCubicCoefficients,
    /// Effective profile continuity for this node across the span.
    pub continuity: ShellProfileNodeContinuity,
}

impl Default for ShellProfileSegmentNodeTopology {
    fn default() -> Self {
        Self {
            half_width: ShellCubicCoefficients {
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            },
            z: ShellCubicCoefficients {
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            },
            continuity: ShellProfileNodeContinuity::Linear,
        }
    }
}

/// Native shell operation family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShellOpKind {
    /// Straight loft between section profiles.
    LineLoft,
    /// Curved loft between section profiles.
    CurveLoft,
    /// Shell hull with built-in thickness.
    ShellHull,
    /// Cross-section swept along a perimeter.
    PerimeterExtrude,
    /// Revolved profile.
    Revolve,
    /// Extruded profile.
    Extrude,
}

/// Open-top behavior for shell hulls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpenTopPolicy {
    /// No opening is cut into the shell.
    Closed,
    /// Box-like top opening used by the current hull prototype.
    BoxCut {
        /// Z plane for the opening cut.
        cut_z: f32,
        /// Half length of the opening along x.
        half_length: f32,
        /// Half width of the opening along y.
        half_width: f32,
        /// X offset for the opening center.
        offset_x: f32,
    },
}

/// Circular section topology for the first native loft kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellSectionTopology {
    /// Default station coordinate.
    pub station: f32,
    /// Default profile center y coordinate.
    pub center_y: f32,
    /// Default profile center z coordinate.
    pub center_z: f32,
    /// Default profile radius.
    pub radius: f32,
    /// Optional parameter slot overriding `station`.
    pub station_param: Option<usize>,
    /// Optional parameter slot overriding `center_y`.
    pub center_y_param: Option<usize>,
    /// Optional parameter slot overriding `center_z`.
    pub center_z_param: Option<usize>,
    /// Optional parameter slot overriding `radius`.
    pub radius_param: Option<usize>,
}

impl ShellSectionTopology {
    /// Builds a static circular section.
    pub fn circle(
        station: f32,
        center_y: f32,
        center_z: f32,
        radius: f32,
    ) -> Self {
        Self {
            station,
            center_y,
            center_z,
            radius,
            station_param: None,
            center_y_param: None,
            center_z_param: None,
            radius_param: None,
        }
    }

    /// Reads the station value for this section.
    pub fn station(self, params: ShellParamsView<'_>) -> f32 {
        params.get(self.station_param, self.station)
    }

    /// Reads the profile center for this section.
    pub fn center(self, params: ShellParamsView<'_>) -> (f32, f32) {
        (
            params.get(self.center_y_param, self.center_y),
            params.get(self.center_z_param, self.center_z),
        )
    }

    /// Reads the profile radius for this section.
    pub fn radius(self, params: ShellParamsView<'_>) -> f32 {
        params.get(self.radius_param, self.radius).max(1.0e-5)
    }
}

/// Authored 2D half-section controls for a ship-like shell profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellProfileSectionTopology {
    /// Station coordinate along the hull axis.
    pub station: f32,
    /// Keel or bottom centerline height.
    pub keel_z: f32,
    /// Sheer/deck edge height.
    pub sheer_z: f32,
    /// Authored half beam at this station before profile shaping.
    pub beam: f32,
    /// Vertically ordered station profile nodes.
    pub nodes: [ShellProfileNodeTopology; SHELL_MAX_NODES_PER_CURVE],
    /// Number of active profile nodes.
    pub node_count: usize,
    /// Interpolation mode for the following station span.
    pub span_interpolation: ShellProfileSpanInterpolation,
    /// Use the dedicated keel/sheer/beam ship section evaluator for this
    /// section when all sections on the segment opt in.
    pub ship_fast_path: bool,
}

impl ShellProfileSectionTopology {
    /// Builds a ship half-section profile control.
    pub fn ship(station: f32, keel_z: f32, sheer_z: f32, beam: f32) -> Self {
        let height = (sheer_z - keel_z).max(0.060);
        let beam = beam.abs().max(0.012);
        let nodes = [
            ShellProfileNodeTopology::new(
                0.0,
                keel_z,
                ShellProfileNodeContinuity::Linear,
            ),
            ShellProfileNodeTopology::new(
                beam * 0.105,
                keel_z + height * 0.070,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                beam * 0.285,
                keel_z + height * 0.185,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                beam * 0.585,
                keel_z + height * 0.390,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                beam * 0.930,
                keel_z + height * 0.615,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                beam * 1.080,
                keel_z + height * 0.805,
                ShellProfileNodeContinuity::Smooth,
            ),
            ShellProfileNodeTopology::new(
                beam * 0.970,
                sheer_z,
                ShellProfileNodeContinuity::Linear,
            ),
        ];
        Self::station_curve(station, &nodes).with_ship_fast_path()
    }

    /// Builds an authored station curve from vertically ordered profile nodes.
    pub fn station_curve(
        station: f32,
        nodes: &[ShellProfileNodeTopology],
    ) -> Self {
        Self::station_curve_with_span(
            station,
            nodes,
            ShellProfileSpanInterpolation::SmoothCatmullRom,
        )
    }

    /// Builds an authored station curve with explicit span interpolation.
    pub fn station_curve_with_span(
        station: f32,
        nodes: &[ShellProfileNodeTopology],
        span_interpolation: ShellProfileSpanInterpolation,
    ) -> Self {
        assert!(
            (2..=SHELL_MAX_NODES_PER_CURVE).contains(&nodes.len()),
            "station profile needs 2..={SHELL_MAX_NODES_PER_CURVE} nodes"
        );
        let mut ordered = nodes.to_vec();
        ordered.sort_by(|left, right| {
            left.z
                .total_cmp(&right.z)
                .then_with(|| left.half_width.total_cmp(&right.half_width))
        });

        let mut fixed =
            [ShellProfileNodeTopology::default(); SHELL_MAX_NODES_PER_CURVE];
        for (index, node) in ordered.iter().copied().enumerate() {
            fixed[index] = ShellProfileNodeTopology {
                half_width: node.half_width.abs().max(0.0),
                z: node.z,
                continuity: node.continuity,
            };
        }
        let node_count = ordered.len();
        let keel_z = fixed[0].z;
        let sheer_z = fixed[node_count - 1].z;
        let beam = fixed[..node_count]
            .iter()
            .fold(0.0_f32, |acc, node| acc.max(node.half_width.abs()))
            .max(0.012);
        Self {
            station,
            keel_z,
            sheer_z,
            beam,
            nodes: fixed,
            node_count,
            span_interpolation,
            ship_fast_path: false,
        }
    }

    /// Marks this section as compatible with the dedicated ship fast path.
    pub fn with_ship_fast_path(mut self) -> Self {
        self.ship_fast_path = true;
        self
    }

    /// Returns the active station profile nodes.
    pub fn active_nodes(&self) -> &[ShellProfileNodeTopology] {
        &self.nodes[..self.node_count]
    }
}

/// Cubic polynomial coefficients in normalized segment coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellCubicCoefficients {
    /// Constant term.
    pub c0: f32,
    /// Linear term.
    pub c1: f32,
    /// Quadratic term.
    pub c2: f32,
    /// Cubic term.
    pub c3: f32,
}

impl ShellCubicCoefficients {
    /// Builds coefficients from Catmull-Rom control values.
    pub fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32) -> Self {
        Self {
            c0: p1,
            c1: 0.5 * (-p0 + p2),
            c2: 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3),
            c3: 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3),
        }
    }

    /// Evaluates the cubic at `t`.
    pub fn eval(self, t: f32) -> f32 {
        ((self.c3 * t + self.c2) * t + self.c1) * t + self.c0
    }

    fn extrema(self) -> [Option<f32>; 2] {
        let a = 3.0 * self.c3;
        let b = 2.0 * self.c2;
        let c = self.c1;
        if a.abs() <= 1.0e-6 {
            if b.abs() <= 1.0e-6 {
                return [None, None];
            }
            return [Some((-c / b).clamp(0.0, 1.0)), None];
        }

        let discriminant = b.mul_add(b, -4.0 * a * c);
        if discriminant < 0.0 {
            return [None, None];
        }
        let sqrt = discriminant.sqrt();
        [
            Some(((-b - sqrt) / (2.0 * a)).clamp(0.0, 1.0)),
            Some(((-b + sqrt) / (2.0 * a)).clamp(0.0, 1.0)),
        ]
    }
}

/// Segment interpolation data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShellSegmentInterpolation {
    /// Linearly interpolate section center and radius.
    Linear,
    /// Use precomputed cubic coefficients for static section values.
    Cubic {
        /// Cubic center-y coefficients.
        center_y: ShellCubicCoefficients,
        /// Cubic center-z coefficients.
        center_z: ShellCubicCoefficients,
        /// Cubic radius coefficients.
        radius: ShellCubicCoefficients,
    },
}

/// Precomputed station ordering for direct point-to-segment lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShellStationMapping {
    /// Segments do not form a strictly monotonic section chain.
    Unordered,
    /// Section stations are strictly increasing.
    Increasing,
    /// Section stations are strictly decreasing.
    Decreasing,
}

/// Native segment topology connecting two sections.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellSegmentTopology {
    /// Left section index.
    pub left_section: usize,
    /// Right section index.
    pub right_section: usize,
    /// Local interpolation mode.
    pub interpolation: ShellSegmentInterpolation,
}

/// Segment interpolation for ship-profile controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellProfileSegmentTopology {
    /// Left profile section index.
    pub left_section: usize,
    /// Right profile section index.
    pub right_section: usize,
    /// Cubic keel height coefficients.
    pub keel_z: ShellCubicCoefficients,
    /// Cubic sheer height coefficients.
    pub sheer_z: ShellCubicCoefficients,
    /// Cubic beam coefficients.
    pub beam: ShellCubicCoefficients,
    /// Station-to-station interpolation mode.
    pub interpolation: ShellProfileSpanInterpolation,
    /// Matching node interpolation coefficients.
    pub nodes: [ShellProfileSegmentNodeTopology; SHELL_MAX_NODES_PER_CURVE],
    /// Number of matching active nodes on this span.
    pub node_count: usize,
    /// Whether this segment can use the dedicated ship section fast path.
    pub ship_fast_path: bool,
}

/// Native ship-like profile sidecar.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellProfileTopology {
    /// Authored section controls.
    pub sections: Box<[ShellProfileSectionTopology]>,
    /// Precomputed cubic station spans.
    pub segments: Box<[ShellProfileSegmentTopology]>,
    /// Bow cap extension before the first station.
    pub bow_cap_extension: f32,
    /// Stern cap extension after the last station.
    pub stern_cap_extension: f32,
    /// Inset amount applied to caps for inner shell evaluation.
    pub cap_inset_scale: f32,
}

/// Fixed-topology helper shape that can be targeted by future monomorphic JIT
/// shell helpers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShellFixedTopologyHelperKind {
    /// Static ship-profile shell hull with no live shell parameters.
    ShipProfileShellHull,
}

/// Immutable native shell topology.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellTopology {
    /// Native operation kind.
    pub kind: ShellOpKind,
    /// Segment connectivity.
    pub segments: Box<[ShellSegmentTopology]>,
    /// Section definitions.
    pub sections: Box<[ShellSectionTopology]>,
    /// Live parameter layout.
    pub param_layout: ShellParamLayout,
    /// Conservative static bounds.
    pub bounds: ShellBounds,
    /// Open-top behavior.
    pub open_top: OpenTopPolicy,
    /// Shell thickness for `ShellHull`.
    pub shell_thickness: f32,
    /// Precomputed station ordering for fast segment lookup.
    pub station_mapping: ShellStationMapping,
    /// Optional authored 2D profile evaluator for ship-like shell hulls.
    pub profile: Option<ShellProfileTopology>,
}

impl ShellTopology {
    /// Builds a line loft over circular sections.
    pub fn line_loft_circles(
        sections: impl Into<Box<[ShellSectionTopology]>>,
    ) -> Self {
        let sections = sections.into();
        let segments = sequential_segments(sections.len());
        let bounds = compute_bounds(&sections, &segments, 0.0);
        let station_mapping = compute_station_mapping(&sections, &segments);
        Self {
            kind: ShellOpKind::LineLoft,
            segments,
            sections,
            param_layout: ShellParamLayout::default(),
            bounds,
            open_top: OpenTopPolicy::Closed,
            shell_thickness: 0.0,
            station_mapping,
            profile: None,
        }
    }

    /// Builds a curve loft over circular sections with precomputed coefficients.
    pub fn curve_loft_circles(
        sections: impl Into<Box<[ShellSectionTopology]>>,
    ) -> Self {
        let sections = sections.into();
        let segments = curve_segments(&sections);
        let bounds = compute_bounds(&sections, &segments, 0.0);
        let station_mapping = compute_station_mapping(&sections, &segments);
        Self {
            kind: ShellOpKind::CurveLoft,
            segments,
            sections,
            param_layout: ShellParamLayout::default(),
            bounds,
            open_top: OpenTopPolicy::Closed,
            shell_thickness: 0.0,
            station_mapping,
            profile: None,
        }
    }

    /// Builds a closed shell hull over circular sections.
    pub fn shell_hull_circles(
        sections: impl Into<Box<[ShellSectionTopology]>>,
        shell_thickness: f32,
        open_top: OpenTopPolicy,
    ) -> Self {
        let sections = sections.into();
        let segments = sequential_segments(sections.len());
        let thickness = shell_thickness.max(0.0);
        let bounds = compute_bounds(&sections, &segments, thickness);
        let station_mapping = compute_station_mapping(&sections, &segments);
        Self {
            kind: ShellOpKind::ShellHull,
            segments,
            sections,
            param_layout: ShellParamLayout::default(),
            bounds,
            open_top,
            shell_thickness: thickness,
            station_mapping,
            profile: None,
        }
    }

    /// Builds a shell hull over circular sections with precomputed coefficients.
    pub fn curve_shell_hull_circles(
        sections: impl Into<Box<[ShellSectionTopology]>>,
        shell_thickness: f32,
        open_top: OpenTopPolicy,
    ) -> Self {
        let sections = sections.into();
        let segments = curve_segments(&sections);
        let thickness = shell_thickness.max(0.0);
        let bounds = compute_bounds(&sections, &segments, thickness);
        let station_mapping = compute_station_mapping(&sections, &segments);
        Self {
            kind: ShellOpKind::ShellHull,
            segments,
            sections,
            param_layout: ShellParamLayout::default(),
            bounds,
            open_top,
            shell_thickness: thickness,
            station_mapping,
            profile: None,
        }
    }

    /// Builds a shell hull over authored 2D ship profile sections.
    pub fn ship_profile_shell_hull(
        profile_sections: impl Into<Box<[ShellProfileSectionTopology]>>,
        shell_thickness: f32,
        open_top: OpenTopPolicy,
    ) -> Self {
        let profile_sections = profile_sections.into();
        let profile_segments = profile_segments(&profile_sections);
        let sections = profile_circle_sections(&profile_sections);
        let segments = curve_segments(&sections);
        let thickness = shell_thickness.max(0.0);
        let bounds = compute_profile_bounds(
            &profile_sections,
            &profile_segments,
            thickness,
        );
        let station_mapping = compute_station_mapping(&sections, &segments);
        Self {
            kind: ShellOpKind::ShellHull,
            segments,
            sections,
            param_layout: ShellParamLayout::default(),
            bounds,
            open_top,
            shell_thickness: thickness,
            station_mapping,
            profile: Some(ShellProfileTopology {
                sections: profile_sections,
                segments: profile_segments,
                bow_cap_extension: 0.03,
                stern_cap_extension: 0.05,
                cap_inset_scale: 0.20,
            }),
        }
    }
    /// Returns conservative AABBs for each native shell segment.
    pub fn segment_bounds(&self) -> Box<[ShellBounds]> {
        self.segments
            .iter()
            .copied()
            .map(|segment| {
                let mut bounds = ShellBounds::empty();
                include_segment_bounds(
                    &mut bounds,
                    &self.sections,
                    segment,
                    self.shell_thickness,
                );
                bounds
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Returns the fixed helper family this topology can use, if any.
    ///
    /// This is intentionally only a measurement/stub hook for now; it lets the
    /// renderer report how many helper calls could plausibly avoid generic
    /// sidecar dispatch and topology decoding before full codegen exists.
    pub fn fixed_topology_helper_kind(
        &self,
    ) -> Option<ShellFixedTopologyHelperKind> {
        if self.kind != ShellOpKind::ShellHull
            || self.param_layout.parameter_count != 0
            || self.station_mapping == ShellStationMapping::Unordered
        {
            return None;
        }

        let profile = self.profile.as_ref()?;
        if profile.sections.len() != self.sections.len()
            || profile.segments.len() != self.segments.len()
            || profile.segments.is_empty()
        {
            return None;
        }

        let profile_sections_are_static_ship = profile
            .sections
            .iter()
            .all(|section| section.ship_fast_path && section.node_count == 7);
        let profile_segments_are_static_ship =
            profile.segments.iter().all(|segment| {
                segment.ship_fast_path
                    && segment.node_count == 7
                    && segment.right_section == segment.left_section + 1
            });

        (profile_sections_are_static_ship && profile_segments_are_static_ship)
            .then_some(ShellFixedTopologyHelperKind::ShipProfileShellHull)
    }
}

fn compute_station_mapping(
    sections: &[ShellSectionTopology],
    segments: &[ShellSegmentTopology],
) -> ShellStationMapping {
    if sections.len() < 2 || segments.len() != sections.len() - 1 {
        return ShellStationMapping::Unordered;
    }
    for (index, segment) in segments.iter().copied().enumerate() {
        if segment.left_section != index || segment.right_section != index + 1 {
            return ShellStationMapping::Unordered;
        }
    }

    let first = sections[0].station;
    let last = sections[sections.len() - 1].station;
    let increasing = last > first;
    let decreasing = last < first;
    if !increasing && !decreasing {
        return ShellStationMapping::Unordered;
    }

    let mut previous = first;
    for section in sections.iter().skip(1).copied() {
        if increasing {
            if section.station <= previous {
                return ShellStationMapping::Unordered;
            }
        } else if section.station >= previous {
            return ShellStationMapping::Unordered;
        }
        previous = section.station;
    }

    if increasing {
        ShellStationMapping::Increasing
    } else {
        ShellStationMapping::Decreasing
    }
}

fn sequential_segments(section_count: usize) -> Box<[ShellSegmentTopology]> {
    assert!(
        section_count >= 2,
        "shell topology needs at least two sections"
    );
    assert!(
        section_count - 1 <= SHELL_MAX_CANDIDATES,
        "shell topology exceeds fixed candidate capacity"
    );
    (0..section_count - 1)
        .map(|index| ShellSegmentTopology {
            left_section: index,
            right_section: index + 1,
            interpolation: ShellSegmentInterpolation::Linear,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn curve_segments(
    sections: &[ShellSectionTopology],
) -> Box<[ShellSegmentTopology]> {
    assert!(
        sections.len() >= 2,
        "shell topology needs at least two sections"
    );
    assert!(
        sections.len() - 1 <= SHELL_MAX_CANDIDATES,
        "shell topology exceeds fixed candidate capacity"
    );
    (0..sections.len() - 1)
        .map(|index| {
            let p0 = sections[index.saturating_sub(1)];
            let p1 = sections[index];
            let p2 = sections[index + 1];
            let p3 = sections[(index + 2).min(sections.len() - 1)];
            ShellSegmentTopology {
                left_section: index,
                right_section: index + 1,
                interpolation: ShellSegmentInterpolation::Cubic {
                    center_y: ShellCubicCoefficients::catmull_rom(
                        p0.center_y,
                        p1.center_y,
                        p2.center_y,
                        p3.center_y,
                    ),
                    center_z: ShellCubicCoefficients::catmull_rom(
                        p0.center_z,
                        p1.center_z,
                        p2.center_z,
                        p3.center_z,
                    ),
                    radius: ShellCubicCoefficients::catmull_rom(
                        p0.radius, p1.radius, p2.radius, p3.radius,
                    ),
                },
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn profile_segments(
    sections: &[ShellProfileSectionTopology],
) -> Box<[ShellProfileSegmentTopology]> {
    assert!(
        sections.len() >= 2,
        "shell profile topology needs at least two sections"
    );
    assert!(
        sections.len() - 1 <= SHELL_MAX_CANDIDATES,
        "shell profile topology exceeds fixed candidate capacity"
    );
    (0..sections.len() - 1)
        .map(|index| {
            let p0 = sections[index.saturating_sub(1)];
            let p1 = sections[index];
            let p2 = sections[index + 1];
            let p3 = sections[(index + 2).min(sections.len() - 1)];
            let node_count = p1.node_count.min(p2.node_count);
            let interpolation = p1.span_interpolation;
            let mut nodes = [ShellProfileSegmentNodeTopology::default();
                SHELL_MAX_NODES_PER_CURVE];
            for (node_index, node) in
                nodes.iter_mut().enumerate().take(node_count)
            {
                let n0 = p0.nodes[node_index.min(p0.node_count - 1)];
                let n1 = p1.nodes[node_index];
                let n2 = p2.nodes[node_index];
                let n3 = p3.nodes[node_index.min(p3.node_count - 1)];
                let continuity = if n1.continuity
                    == ShellProfileNodeContinuity::Linear
                    || n2.continuity == ShellProfileNodeContinuity::Linear
                {
                    ShellProfileNodeContinuity::Linear
                } else {
                    ShellProfileNodeContinuity::Smooth
                };
                *node = ShellProfileSegmentNodeTopology {
                    half_width: ShellCubicCoefficients::catmull_rom(
                        n0.half_width,
                        n1.half_width,
                        n2.half_width,
                        n3.half_width,
                    ),
                    z: ShellCubicCoefficients::catmull_rom(
                        n0.z, n1.z, n2.z, n3.z,
                    ),
                    continuity,
                };
            }
            ShellProfileSegmentTopology {
                left_section: index,
                right_section: index + 1,
                keel_z: ShellCubicCoefficients::catmull_rom(
                    p0.keel_z, p1.keel_z, p2.keel_z, p3.keel_z,
                ),
                sheer_z: ShellCubicCoefficients::catmull_rom(
                    p0.sheer_z, p1.sheer_z, p2.sheer_z, p3.sheer_z,
                ),
                beam: ShellCubicCoefficients::catmull_rom(
                    p0.beam, p1.beam, p2.beam, p3.beam,
                ),
                interpolation,
                nodes,
                node_count,
                ship_fast_path: p1.ship_fast_path && p2.ship_fast_path,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn profile_circle_sections(
    sections: &[ShellProfileSectionTopology],
) -> Box<[ShellSectionTopology]> {
    sections
        .iter()
        .copied()
        .map(|section| {
            let center_z = (section.keel_z + section.sheer_z) * 0.5;
            let half_height = (section.sheer_z - section.keel_z).abs() * 0.5;
            let radius = section.beam.abs().max(half_height).max(0.035) * 1.35;
            ShellSectionTopology::circle(section.station, 0.0, center_z, radius)
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn compute_bounds(
    sections: &[ShellSectionTopology],
    segments: &[ShellSegmentTopology],
    extra: f32,
) -> ShellBounds {
    let mut bounds = ShellBounds::empty();
    for segment in segments {
        include_segment_bounds(&mut bounds, sections, *segment, extra);
    }
    assert!(bounds.is_valid(), "shell topology bounds must be valid");
    bounds
}

fn compute_profile_bounds(
    sections: &[ShellProfileSectionTopology],
    segments: &[ShellProfileSegmentTopology],
    extra: f32,
) -> ShellBounds {
    let mut bounds = ShellBounds::empty();
    let extra = extra.max(0.0);
    for section in sections {
        let beam = section.beam.abs() * 1.35 + extra;
        bounds.include_point(section.station, -beam, section.keel_z - extra);
        bounds.include_point(section.station, beam, section.sheer_z + extra);
        for node in section.active_nodes() {
            bounds.include_point(
                section.station,
                -node.half_width - extra,
                node.z - extra,
            );
            bounds.include_point(
                section.station,
                node.half_width + extra,
                node.z + extra,
            );
        }
    }
    for segment in segments {
        for t in profile_bound_samples(*segment) {
            if !t.is_finite() {
                continue;
            }
            let left = sections[segment.left_section];
            let right = sections[segment.right_section];
            let station = left.station + (right.station - left.station) * t;
            let keel_z = segment.keel_z.eval(t);
            let sheer_z = segment.sheer_z.eval(t);
            let beam = segment.beam.eval(t).abs() * 1.35 + extra;
            bounds.include_point(station, -beam, keel_z - extra);
            bounds.include_point(station, beam, sheer_z + extra);
            for node_index in 0..segment.node_count {
                let (half_width, z) = eval_profile_segment_node(
                    sections, *segment, node_index, t,
                );
                let half_width = half_width.abs() + extra;
                bounds.include_point(station, -half_width, z - extra);
                bounds.include_point(station, half_width, z + extra);
            }
        }
    }
    bounds.include_point(
        sections[0].station - 0.03 - extra,
        -extra,
        sections[0].keel_z - extra,
    );
    bounds.include_point(
        sections[sections.len() - 1].station + 0.05 + extra,
        extra,
        sections[sections.len() - 1].sheer_z + extra,
    );
    assert!(bounds.is_valid(), "shell profile bounds must be valid");
    bounds
}

fn profile_bound_samples(segment: ShellProfileSegmentTopology) -> [f32; 16] {
    let mut samples = [0.0; 16];
    samples[1] = 1.0;
    samples[2] = 0.25;
    samples[3] = 0.50;
    samples[4] = 0.75;
    let mut index = 5;
    for value in segment
        .keel_z
        .extrema()
        .into_iter()
        .chain(segment.sheer_z.extrema())
        .chain(segment.beam.extrema())
        .flatten()
    {
        samples[index] = value;
        index += 1;
        if index == samples.len() {
            return samples;
        }
    }
    samples
}

fn eval_profile_segment_node(
    sections: &[ShellProfileSectionTopology],
    segment: ShellProfileSegmentTopology,
    node_index: usize,
    t: f32,
) -> (f32, f32) {
    let left = sections[segment.left_section].nodes[node_index];
    let right = sections[segment.right_section].nodes[node_index];
    match segment.interpolation {
        ShellProfileSpanInterpolation::Linear => (
            lerp_f32(left.half_width, right.half_width, t),
            lerp_f32(left.z, right.z, t),
        ),
        ShellProfileSpanInterpolation::SmoothCatmullRom => {
            let node = segment.nodes[node_index];
            (node.half_width.eval(t), node.z.eval(t))
        }
    }
}

fn lerp_f32(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t
}

fn include_segment_bounds(
    bounds: &mut ShellBounds,
    sections: &[ShellSectionTopology],
    segment: ShellSegmentTopology,
    extra: f32,
) {
    let left = sections
        .get(segment.left_section)
        .expect("segment section index must be valid");
    let right = sections
        .get(segment.right_section)
        .expect("segment section index must be valid");
    let extra = extra.max(0.0);
    let min_x = left.station.min(right.station);
    let max_x = left.station.max(right.station);

    let mut min_y = left.center_y.min(right.center_y);
    let mut max_y = left.center_y.max(right.center_y);
    let mut min_z = left.center_z.min(right.center_z);
    let mut max_z = left.center_z.max(right.center_z);
    let mut max_radius = left.radius.max(right.radius).max(0.0);

    if let ShellSegmentInterpolation::Cubic {
        center_y,
        center_z,
        radius,
    } = segment.interpolation
    {
        for t in cubic_bound_samples(center_y, center_z, radius) {
            let y = center_y.eval(t);
            let z = center_z.eval(t);
            let r = radius.eval(t).max(0.0);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
            max_radius = max_radius.max(r);
        }
    }

    let radius = max_radius + extra;
    bounds.include_point(min_x, min_y - radius, min_z - radius);
    bounds.include_point(max_x, max_y + radius, max_z + radius);
}

fn cubic_bound_samples(
    center_y: ShellCubicCoefficients,
    center_z: ShellCubicCoefficients,
    radius: ShellCubicCoefficients,
) -> [f32; 8] {
    let mut samples = [0.0; 8];
    samples[1] = 1.0;
    for (index, value) in (2..).zip(
        center_y
            .extrema()
            .into_iter()
            .chain(center_z.extrema())
            .chain(radius.extrema())
            .flatten(),
    ) {
        samples[index] = value;
    }
    samples
}
