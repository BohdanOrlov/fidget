use std::sync::Arc;

use crate::{
    context::{Context, Tree},
    eval::{Function, MathFunction, Tape, TracingEvaluator},
    shell::{
        OpenTopPolicy, ShellEvalScratch, ShellParamsView,
        ShellProfileSectionTopology, ShellSectionTopology, ShellTopology,
        eval::SHELL_EVAL_STATS_TEST_LOCK, eval_shell_distance,
        eval_shell_interval, eval_shell_interval_with_trace,
        reset_shell_eval_stats, set_shell_eval_stats_enabled, shell_eval_stats,
    },
    types::Interval,
    var::Var,
    vm::VmFunction,
};

fn shell() -> Arc<ShellTopology> {
    Arc::new(ShellTopology::line_loft_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
    ))
}

#[test]
fn shell_interval_rejects_tiles_outside_global_bounds() {
    let tree = Tree::line_loft_shell(shell());
    let mut ctx = Context::new();
    let root = ctx.import(&tree);
    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.interval_tape(Default::default());
    let mut args = vec![Interval::new(0.0, 0.0); tape.vars().len()];
    args[tape.vars()[&Var::X]] = Interval::new(4.0, 5.0);
    args[tape.vars()[&Var::Y]] = Interval::new(-0.25, 0.25);
    args[tape.vars()[&Var::Z]] = Interval::new(-0.25, 0.25);

    let mut eval = VmFunction::new_interval_eval();
    let (out, trace) = eval.eval(&tape, &args).unwrap();

    assert!(out[0].lower() > 0.0, "expected positive rejection interval");
    assert_eq!(out[0].upper(), f32::INFINITY);
    assert!(trace.is_none());
}

#[test]
fn shell_interval_stays_conservative_inside_global_bounds() {
    let tree = Tree::line_loft_shell(shell());
    let mut ctx = Context::new();
    let root = ctx.import(&tree);
    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.interval_tape(Default::default());
    let mut args = vec![Interval::new(0.0, 0.0); tape.vars().len()];
    args[tape.vars()[&Var::X]] = Interval::new(0.0, 2.0);
    args[tape.vars()[&Var::Y]] = Interval::new(-1.0, 1.0);
    args[tape.vars()[&Var::Z]] = Interval::new(-1.0, 1.0);

    let mut eval = VmFunction::new_interval_eval();
    let (out, trace) = eval.eval(&tape, &args).unwrap();

    assert_eq!(out[0].lower(), f32::NEG_INFINITY);
    assert_eq!(out[0].upper(), f32::INFINITY);
    assert!(trace.is_none());
}

#[test]
fn shell_hull_interval_contains_sampled_points() {
    let topology = ShellTopology::shell_hull_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
        0.2,
        OpenTopPolicy::Closed,
    );
    let tiles = [
        (
            Interval::new(0.25, 0.75),
            Interval::new(-0.1, 0.1),
            Interval::new(-0.1, 0.1),
        ),
        (
            Interval::new(0.25, 0.75),
            Interval::new(0.88, 1.02),
            Interval::new(-0.05, 0.05),
        ),
        (
            Interval::new(2.4, 2.8),
            Interval::new(-0.2, 0.2),
            Interval::new(-0.2, 0.2),
        ),
    ];

    let mut scratch = ShellEvalScratch::default();
    for (x, y, z) in tiles {
        let interval = eval_shell_interval(&topology, x, y, z);
        for xi in [x.lower(), (x.lower() + x.upper()) * 0.5, x.upper()] {
            for yi in [y.lower(), (y.lower() + y.upper()) * 0.5, y.upper()] {
                for zi in [z.lower(), (z.lower() + z.upper()) * 0.5, z.upper()]
                {
                    let distance = eval_shell_distance(
                        &topology,
                        ShellParamsView::empty(),
                        &mut scratch,
                        xi,
                        yi,
                        zi,
                    )
                    .distance;
                    assert!(
                        interval.lower() <= distance,
                        "interval lower {} excludes sample {distance}",
                        interval.lower()
                    );
                    assert!(
                        distance <= interval.upper(),
                        "interval upper {} excludes sample {distance}",
                        interval.upper()
                    );
                }
            }
        }
    }
}

#[test]
fn shell_interval_trace_records_local_active_segment_mask() {
    let topology = ShellTopology::line_loft_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 0.5),
            ShellSectionTopology::circle(1.0, 0.0, 0.0, 0.5),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 0.5),
            ShellSectionTopology::circle(3.0, 0.0, 0.0, 0.5),
        ]
        .into_boxed_slice(),
    );

    let (interval, trace) = eval_shell_interval_with_trace(
        &topology,
        Interval::new(1.20, 1.30),
        Interval::new(-0.10, 0.10),
        Interval::new(-0.10, 0.10),
    );

    assert_eq!(interval.lower(), f32::NEG_INFINITY);
    assert_eq!(interval.upper(), f32::INFINITY);
    assert_eq!(trace.segment_count, 3);
    assert_eq!(
        trace.active_segment_mask, 0b010,
        "tile inside the middle station span should only keep that span active"
    );
    assert!(
        trace.sidecar_reduction_eligible,
        "simple line-loft traces may be used for reduced sidecar simplification"
    );
}

#[test]
fn profile_shell_interval_trace_records_local_active_segment_mask() {
    let _stats_guard = SHELL_EVAL_STATS_TEST_LOCK.lock().unwrap();
    let topology = ShellTopology::ship_profile_shell_hull(
        vec![
            ShellProfileSectionTopology::ship(0.0, -0.4, 0.7, 0.6),
            ShellProfileSectionTopology::ship(1.0, -0.4, 0.7, 0.6),
            ShellProfileSectionTopology::ship(2.0, -0.4, 0.7, 0.6),
            ShellProfileSectionTopology::ship(3.0, -0.4, 0.7, 0.6),
        ]
        .into_boxed_slice(),
        0.08,
        OpenTopPolicy::Closed,
    );

    let (interval, trace) = eval_shell_interval_with_trace(
        &topology,
        Interval::new(1.20, 1.30),
        Interval::new(-0.10, 0.10),
        Interval::new(0.00, 0.10),
    );

    assert_eq!(interval.lower(), f32::NEG_INFINITY);
    assert_eq!(interval.upper(), f32::INFINITY);
    assert_eq!(trace.segment_count, 3);
    assert_eq!(
        trace.active_segment_mask, 0b010,
        "profile tile inside the middle station span should only keep that span active"
    );
    assert!(
        !trace.sidecar_reduction_eligible,
        "profile-shell active masks are diagnostic until reduced sidecars are benchmarked as a win"
    );
}

#[test]
fn shell_interval_stats_assert_no_hot_loop_allocations() {
    let _stats_guard = SHELL_EVAL_STATS_TEST_LOCK.lock().unwrap();
    let topology = ShellTopology::shell_hull_circles(
        vec![
            ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
            ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
        ]
        .into_boxed_slice(),
        0.2,
        OpenTopPolicy::Closed,
    );

    set_shell_eval_stats_enabled(true);
    reset_shell_eval_stats();

    let intervals = [
        eval_shell_interval(
            &topology,
            Interval::new(0.25, 0.75),
            Interval::new(-0.1, 0.1),
            Interval::new(-0.1, 0.1),
        ),
        eval_shell_interval(
            &topology,
            Interval::new(3.0, 4.0),
            Interval::new(-0.1, 0.1),
            Interval::new(-0.1, 0.1),
        ),
    ];
    let stats = shell_eval_stats();
    set_shell_eval_stats_enabled(false);

    assert!(intervals.iter().all(|interval| interval.lower() > 0.0));
    assert!(
        stats.interval_calls >= intervals.len() as u64,
        "expected at least this test's interval calls, got {}",
        stats.interval_calls
    );
    assert_eq!(stats.interval_hot_loop_allocations, 0);
    assert_eq!(stats.float_slice_hot_loop_allocations, 0);
    assert_eq!(stats.grad_slice_hot_loop_allocations, 0);
    assert_eq!(stats.hot_loop_allocations, 0);
}
