use std::sync::Arc;

use fidget::{
    context::Tree,
    shell::{
        OpenTopPolicy, ShellProfileSectionTopology,
        ShellProfileSpanInterpolation, ShellSectionTopology, ShellTopology,
    },
};

const HULL_X_MIN: f32 = -1.34;
const HULL_X_MAX: f32 = 1.18;
// Hercules-converged hull lines: bow stem rises above the baseline, long
// parallel middle body holds near maximum draft, and counter stern lifts back
// up at the after overhang. Keep in sync with `src/spline_ship.rs`.
const KEEL_CTRL: [f32; 11] = [
    0.06, -0.12, -0.26, -0.33, -0.36, -0.37, -0.36, -0.32, -0.22, -0.06, 0.10,
];
const SHEER_CTRL: [f32; 11] = [
    0.32, 0.24, 0.19, 0.17, 0.16, 0.155, 0.16, 0.17, 0.19, 0.22, 0.27,
];
const BEAM_CTRL: [f32; 11] = [
    0.000, 0.020, 0.085, 0.190, 0.275, 0.330, 0.345, 0.325, 0.245, 0.145, 0.055,
];
// Native profile controls are oriented like the Hercules drawing: rounded
// counter stern at x-min, fuller middle body, fine rising bow stem at x-max,
// and a long flat run.
const NATIVE_KEEL_CTRL: [f32; 11] = [
    0.015, -0.120, -0.245, -0.330, -0.365, -0.372, -0.368, -0.335, -0.255,
    -0.075, 0.110,
];
const NATIVE_SHEER_CTRL: [f32; 11] = [
    0.280, 0.238, 0.198, 0.174, 0.162, 0.158, 0.162, 0.180, 0.212, 0.262, 0.326,
];
const NATIVE_BEAM_CTRL: [f32; 11] = [
    0.165, 0.205, 0.280, 0.330, 0.355, 0.360, 0.342, 0.292, 0.190, 0.066, 0.000,
];

#[derive(Clone, Copy, Debug)]
struct HullShellOptions {
    shell_thickness: f32,
    open_top: bool,
    open_top_cut_z: f32,
    open_top_half_length: f32,
    open_top_half_width: f32,
}

const CLOSED_THIN_OPTIONS: HullShellOptions = HullShellOptions {
    shell_thickness: 0.055,
    open_top: false,
    open_top_cut_z: 0.0,
    open_top_half_length: 0.0,
    open_top_half_width: 0.0,
};

const CLOSED_THICK_OPTIONS: HullShellOptions = HullShellOptions {
    shell_thickness: 0.115,
    open_top: false,
    open_top_cut_z: 0.0,
    open_top_half_length: 0.0,
    open_top_half_width: 0.0,
};

const OPEN_TOP_OPTIONS: HullShellOptions = HullShellOptions {
    shell_thickness: 0.070,
    open_top: true,
    open_top_cut_z: 0.015,
    open_top_half_length: 0.84,
    open_top_half_width: 0.20,
};

pub(crate) fn build_builtin_ship_tree(target: &str) -> Option<Tree> {
    let normalized = target.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "ship-native-line-loft" => Some(build_native_ship_line_loft_tree()),
        "ship-native-curve-loft" => Some(build_native_ship_curve_loft_tree()),
        "ship-native-shell"
        | "ship-native-profile-shell"
        | "ship-native-hull" => {
            Some(build_native_ship_shell_tree(CLOSED_THICK_OPTIONS))
        }
        _ => {
            builtin_ship_options(&normalized).map(build_spline_ship_shell_tree)
        }
    }
}

pub(crate) fn is_builtin_ship_target(target: &str) -> bool {
    let normalized = target.trim().to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "ship-native-line-loft"
            | "ship-native-curve-loft"
            | "ship-native-shell"
            | "ship-native-profile-shell"
            | "ship-native-hull"
    ) || builtin_ship_options(&normalized).is_some()
}

fn builtin_ship_options(normalized: &str) -> Option<HullShellOptions> {
    match normalized {
        "ship" | "ship-expression" | "ship-thick" | "ship-closed-thick" => {
            CLOSED_THICK_OPTIONS
        }
        "ship-thin" | "ship-closed-thin" => CLOSED_THIN_OPTIONS,
        "ship-open" | "ship-open-top" => OPEN_TOP_OPTIONS,
        _ => return None,
    }
    .into()
}

fn build_native_ship_shell_tree(options: HullShellOptions) -> Tree {
    let open_top = if options.open_top {
        OpenTopPolicy::BoxCut {
            cut_z: options.open_top_cut_z,
            half_length: options.open_top_half_length,
            half_width: options.open_top_half_width,
            offset_x: -0.02,
        }
    } else {
        OpenTopPolicy::Closed
    };
    Tree::shell_hull(Arc::new(ShellTopology::ship_profile_shell_hull(
        native_ship_profile_sections().into_boxed_slice(),
        options.shell_thickness.max(0.008),
        open_top,
    )))
}

fn build_native_ship_line_loft_tree() -> Tree {
    Tree::line_loft_shell(Arc::new(ShellTopology::line_loft_circles(
        native_ship_sections().into_boxed_slice(),
    )))
}

fn build_native_ship_curve_loft_tree() -> Tree {
    Tree::curve_loft_shell(Arc::new(ShellTopology::curve_loft_circles(
        native_ship_sections().into_boxed_slice(),
    )))
}

fn build_spline_ship_shell_tree(options: HullShellOptions) -> Tree {
    let outer = build_spline_ship_outer_tree(0.0);
    let inner =
        build_spline_ship_outer_tree(options.shell_thickness.max(0.008));
    let mut shell = outer.max(-inner);

    if options.open_top {
        let opening = build_open_top_cutter(options);
        shell = shell.max(-opening);
    }

    shell
}

fn build_spline_ship_outer_tree(inset: f32) -> Tree {
    let x_axis = Tree::x();
    let y_axis = Tree::y();
    let z_axis = Tree::z();
    let inset = inset.max(0.0);

    let t = ((x_axis.clone() - HULL_X_MIN) / (HULL_X_MAX - HULL_X_MIN))
        .max(0.0)
        .min(1.0);
    let keel_z =
        sample_catmull_rom_tree(&KEEL_CTRL, t.clone()) + inset as f64 * 0.92;
    let sheer_z =
        sample_catmull_rom_tree(&SHEER_CTRL, t.clone()) - inset as f64 * 0.92;
    let bow_pinch = lerp_tree(0.12, 1.0, t.clone().sqrt());
    let stern_taper =
        lerp_tree(1.0, 0.88, ((t.clone() - 0.82) / 0.18).max(0.0).min(1.0));
    let beam = (sample_catmull_rom_tree(&BEAM_CTRL, t) - inset as f64 * 0.96)
        * bow_pinch
        * stern_taper;
    let beam = beam.max(0.012);

    let height = (sheer_z.clone() - keel_z.clone()).max(0.060);
    let z_rel = ((z_axis.clone() - keel_z.clone()) / height)
        .max(0.0)
        .min(1.0);
    let sheer_tuck = (Tree::constant(1.0)
        - z_rel.clone().square() * z_rel.clone() * 0.30)
        .max(0.62);
    let half_width = beam
        * (Tree::constant(0.045) + z_rel.clone().sqrt() * 1.20)
        * sheer_tuck;
    let side = y_axis.abs() - half_width;
    let vertical_profile = (keel_z - z_axis.clone()).max(z_axis - sheer_z);
    let hull = side.max(vertical_profile);
    let bow_cap = (HULL_X_MIN - 0.03 + inset * 0.20) - x_axis.clone();
    let stern_cap = x_axis - (HULL_X_MAX + 0.05 - inset * 0.20);
    hull.max(bow_cap.max(stern_cap))
}

fn build_open_top_cutter(options: HullShellOptions) -> Tree {
    let x_axis = Tree::x();
    let y_axis = Tree::y();
    let z_axis = Tree::z();

    let x_window = (x_axis + 0.02).abs() - options.open_top_half_length as f64;
    let y_window = y_axis.abs() - options.open_top_half_width as f64;
    let z_cut = Tree::constant(options.open_top_cut_z as f64) - z_axis;

    z_cut.max(x_window).max(y_window)
}

fn native_ship_sections() -> Vec<ShellSectionTopology> {
    NATIVE_KEEL_CTRL
        .iter()
        .zip(NATIVE_SHEER_CTRL.iter())
        .zip(NATIVE_BEAM_CTRL.iter())
        .enumerate()
        .map(|(i, ((keel, sheer), beam))| {
            let t = i as f32 / (NATIVE_KEEL_CTRL.len() - 1) as f32;
            let station = HULL_X_MIN + t * (HULL_X_MAX - HULL_X_MIN);
            let center_z = (keel + sheer) * 0.5;
            let half_height = (sheer - keel).abs() * 0.5;
            let half_width = (*beam).max(0.012);
            let radius = half_width.max(half_height * 0.35).max(0.035);
            ShellSectionTopology::circle(station, 0.0, center_z, radius)
        })
        .collect()
}

fn native_ship_profile_sections() -> Vec<ShellProfileSectionTopology> {
    let last_index = NATIVE_KEEL_CTRL.len() - 1;
    NATIVE_KEEL_CTRL
        .iter()
        .zip(NATIVE_SHEER_CTRL.iter())
        .zip(NATIVE_BEAM_CTRL.iter())
        .enumerate()
        .map(|(i, ((keel, sheer), beam))| {
            let t = i as f32 / (NATIVE_KEEL_CTRL.len() - 1) as f32;
            let station = HULL_X_MIN + t * (HULL_X_MAX - HULL_X_MIN);
            let half_width = (*beam).max(0.012);
            let span = if i + 2 >= last_index {
                ShellProfileSpanInterpolation::Linear
            } else {
                ShellProfileSpanInterpolation::SmoothCatmullRom
            };
            let mut section = ShellProfileSectionTopology::ship(
                station, *keel, *sheer, half_width,
            );
            section.span_interpolation = span;
            section
        })
        .collect()
}

fn sample_catmull_rom_tree(control_points: &[f32], t: Tree) -> Tree {
    assert!(!control_points.is_empty());
    if control_points.len() == 1 {
        return Tree::constant(control_points[0] as f64);
    }

    let segment_count = (control_points.len() - 1) as f64;
    let scaled = t.max(0.0).min(1.0) * segment_count;
    let segment = scaled.clone().floor().min(segment_count - 1.0).max(0.0);
    let u = scaled - segment.clone();
    let u2 = u.square();
    let u3 = u2.clone() * u.clone();

    let get = |idx: isize| -> f32 {
        let clamped = idx.clamp(0, control_points.len() as isize - 1) as usize;
        control_points[clamped]
    };

    let mut out = Tree::constant(0.0);
    for i in 0..control_points.len() - 1 {
        let i = i as isize;
        let p0 = get(i - 1) as f64;
        let p1 = get(i) as f64;
        let p2 = get(i + 1) as f64;
        let p3 = get(i + 2) as f64;
        let constant_term = Tree::constant(2.0 * p1);
        let linear_term = u.clone() * (-p0 + p2);
        let quadratic_term = u2.clone() * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
        let cubic_term = u3.clone() * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
        let value =
            (constant_term + linear_term + quadratic_term + cubic_term) * 0.5;
        let mask = (Tree::constant(1.0)
            - segment.clone().compare(i as f64).abs())
        .max(0.0);
        out = out + value * mask;
    }
    out
}

fn lerp_tree(start: f64, end: f64, t: Tree) -> Tree {
    Tree::constant(start) + t * (end - start)
}
