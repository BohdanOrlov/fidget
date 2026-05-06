use std::sync::Arc;

use crate::{
    compiler::{RegOp, SsaOp, SsaTape},
    context::{Context, Tree},
    eval::{BulkEvaluator, Function, MathFunction, Tape, TracingEvaluator},
    shell::{
        ShellEvalScratch, ShellParamsView, ShellSectionTopology, ShellTopology,
        eval_shell_distance,
    },
    types::{Grad, Interval},
    var::Var,
    vm::{VmData, VmFunction},
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

fn shell_tree() -> Tree {
    Tree::line_loft_shell(shell())
}

fn import_tree(tree: &Tree) -> (Context, crate::context::Node) {
    let mut ctx = Context::new();
    let root = ctx.import(tree);
    (ctx, root)
}

fn point_args(vars: &crate::var::VarMap, x: f32, y: f32, z: f32) -> Vec<f32> {
    let mut args = vec![0.0; vars.len()];
    args[vars[&Var::X]] = x;
    args[vars[&Var::Y]] = y;
    args[vars[&Var::Z]] = z;
    args
}

#[test]
fn shell_lowers_to_one_ssa_shell_op() {
    let (ctx, root) = import_tree(&shell_tree());
    let (tape, vars) = SsaTape::new(&ctx, &[root]).unwrap();

    let shell_ops: Vec<_> = tape
        .iter()
        .filter_map(|op| match op {
            SsaOp::ShellDistance(out, shell, x, y, z) => {
                Some((*out, *shell, *x, *y, *z))
            }
            _ => None,
        })
        .collect();

    assert_eq!(shell_ops.len(), 1);
    assert_eq!(shell_ops[0].1, 0);
    assert_eq!(tape.shells.len(), 1);
    assert_eq!(vars.len(), 3);
}

#[test]
fn shell_lowers_to_one_reg_shell_op() {
    let (ctx, root) = import_tree(&shell_tree());
    let data = VmData::<255>::new(&ctx, &[root]).unwrap();
    let shell_ops: Vec<_> = data
        .iter_asm()
        .filter_map(|op| match op {
            RegOp::ShellDistance(out, shell, x, y, z) => {
                Some((out, shell, x, y, z))
            }
            _ => None,
        })
        .collect();

    assert_eq!(shell_ops.len(), 1);
    assert_eq!(shell_ops[0].1, 0);
    assert!(data.shell_topology(0).is_some());
}

#[test]
fn shell_vm_point_eval_matches_kernel() {
    let expected = {
        let shell = shell();
        let mut scratch = ShellEvalScratch::default();
        eval_shell_distance(
            &shell,
            ShellParamsView::empty(),
            &mut scratch,
            1.0,
            1.25,
            0.0,
        )
        .distance
    };

    let (ctx, root) = import_tree(&shell_tree());
    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.point_tape(Default::default());
    let mut eval = VmFunction::new_point_eval();
    let args = point_args(tape.vars(), 1.0, 1.25, 0.0);

    let (out, trace) = eval.eval(&tape, &args).unwrap();
    assert_approx_eq(out[0], expected);
    assert!(trace.is_none());
}

#[test]
fn shell_vm_float_slice_eval_matches_kernel() {
    let samples = [(1.0, 0.0, 0.0), (1.0, 1.25, 0.0), (3.0, 0.0, 0.0)];
    let shell = shell();
    let mut scratch = ShellEvalScratch::default();
    let expected: Vec<_> = samples
        .iter()
        .map(|&(x, y, z)| {
            eval_shell_distance(
                &shell,
                ShellParamsView::empty(),
                &mut scratch,
                x,
                y,
                z,
            )
            .distance
        })
        .collect();

    let (ctx, root) = import_tree(&Tree::line_loft_shell(shell));
    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.float_slice_tape(Default::default());
    let mut args = vec![vec![0.0; samples.len()]; tape.vars().len()];
    for (i, &(x, y, z)) in samples.iter().enumerate() {
        args[tape.vars()[&Var::X]][i] = x;
        args[tape.vars()[&Var::Y]][i] = y;
        args[tape.vars()[&Var::Z]][i] = z;
    }

    let mut eval = VmFunction::new_float_slice_eval();
    let out = eval.eval(&tape, &args).unwrap();
    for (&actual, &expected) in out[0].iter().zip(&expected) {
        assert_approx_eq(actual, expected);
    }
}

#[test]
fn shell_vm_grad_slice_eval_returns_native_distance_and_gradient() {
    let (ctx, root) = import_tree(&shell_tree());
    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.grad_slice_tape(Default::default());
    let mut args = vec![vec![Grad::from(0.0); 1]; tape.vars().len()];
    args[tape.vars()[&Var::X]][0] = Grad::new(1.0, 1.0, 0.0, 0.0);
    args[tape.vars()[&Var::Y]][0] = Grad::new(1.25, 0.0, 1.0, 0.0);
    args[tape.vars()[&Var::Z]][0] = Grad::new(0.0, 0.0, 0.0, 1.0);

    let mut eval = VmFunction::new_grad_slice_eval();
    let out = eval.eval(&tape, &args).unwrap()[0][0];

    assert_approx_eq(out.v, 0.25);
    assert!(out.dx.abs() <= 1.0e-3, "unexpected dx={}", out.dx);
    assert!((out.dy - 1.0).abs() <= 1.0e-3, "unexpected dy={}", out.dy);
    assert!(out.dz.abs() <= 1.0e-3, "unexpected dz={}", out.dz);
}

#[test]
fn shell_vm_interval_eval_is_conservative() {
    let (ctx, root) = import_tree(&shell_tree());
    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.interval_tape(Default::default());
    let mut args = vec![Interval::new(0.0, 0.0); tape.vars().len()];
    args[tape.vars()[&Var::X]] = Interval::new(0.0, 2.0);
    args[tape.vars()[&Var::Y]] = Interval::new(-2.0, 2.0);
    args[tape.vars()[&Var::Z]] = Interval::new(-2.0, 2.0);

    let mut eval = VmFunction::new_interval_eval();
    let (out, trace) = eval.eval(&tape, &args).unwrap();

    assert_eq!(out[0].lower(), f32::NEG_INFINITY);
    assert_eq!(out[0].upper(), f32::INFINITY);
    assert!(trace.is_none());
}

#[test]
fn shell_simplification_preserves_shell_sidecar() {
    let (ctx, root) = import_tree(&shell_tree().min(0.5));
    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.point_tape(Default::default());
    let mut eval = VmFunction::new_point_eval();
    let args = point_args(tape.vars(), 1.0, 0.0, 0.0);
    let (_out, trace) = eval.eval(&tape, &args).unwrap();

    let simplified = shape
        .simplify(
            trace.unwrap(),
            VmData::<255>::default(),
            &mut Default::default(),
        )
        .unwrap();

    let shell_op_count = simplified
        .data()
        .iter_asm()
        .filter(|op| matches!(op, RegOp::ShellDistance(..)))
        .count();
    assert_eq!(shell_op_count, 1);
    assert!(simplified.data().shell_topology(0).is_some());
}

#[test]
fn shell_constant_coordinates_are_materialized_before_vm_eval() {
    let tree = Tree::shell_distance(shell(), 1.0.into(), Tree::y(), 0.0.into());
    let (ctx, root) = import_tree(&tree);
    let data = VmData::<255>::new(&ctx, &[root]).unwrap();
    let copy_imm_count = data
        .iter_asm()
        .filter(|op| matches!(op, RegOp::CopyImm(..)))
        .count();
    assert_eq!(copy_imm_count, 2);

    let shape = VmFunction::new(&ctx, &[root]).unwrap();
    let tape = shape.point_tape(Default::default());
    let mut eval = VmFunction::new_point_eval();
    let args = vec![1.25; tape.vars().len()];
    let (out, _trace) = eval.eval(&tape, &args).unwrap();
    assert_approx_eq(out[0], 0.25);
}

fn assert_approx_eq(left: f32, right: f32) {
    let diff = (left - right).abs();
    assert!(
        diff <= 1.0e-5,
        "expected {left} to be within tolerance of {right}; diff={diff}"
    );
}
