use std::sync::Arc;

use crate::{
    context::{BinaryOpcode, Context, Op, Tree},
    shell::{ShellSectionTopology, ShellTopology},
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
fn shell_tree_imports_as_native_context_op() {
    let shell = shell();
    let tree = Tree::line_loft_shell(shell.clone());
    let mut ctx = Context::new();
    let root = ctx.import(&tree);

    let Op::Shell(key, x, y, z) = *ctx.get_op(root).unwrap() else {
        panic!("expected native shell op");
    };
    assert!(Arc::ptr_eq(ctx.shell_topology(key).unwrap(), &shell));
    assert!(matches!(ctx.get_op(x), Some(Op::Input(_))));
    assert!(matches!(ctx.get_op(y), Some(Op::Input(_))));
    assert!(matches!(ctx.get_op(z), Some(Op::Input(_))));
}

#[test]
fn shell_context_eval_matches_kernel_distance() {
    let tree = Tree::line_loft_shell(shell());
    let mut ctx = Context::new();
    let root = ctx.import(&tree);

    assert_approx_eq(ctx.eval_xyz(root, 1.0, 0.0, 0.0).unwrap(), -1.0);
    assert_approx_eq(ctx.eval_xyz(root, 1.0, 1.25, 0.0).unwrap(), 0.25);
}

#[test]
fn shell_context_export_roundtrips_tree() {
    let shell = shell();
    let tree = Tree::shell_distance(shell, Tree::x(), Tree::y(), Tree::z());
    let mut ctx = Context::new();
    let root = ctx.import(&tree);
    let exported = ctx.export(root).unwrap();

    assert_eq!(exported, tree);
}

#[test]
fn shell_context_dedups_identical_topology_references() {
    let shell = shell();
    let tree_a = Tree::line_loft_shell(shell.clone());
    let tree_b = Tree::line_loft_shell(shell);
    let mut ctx = Context::new();

    let root_a = ctx.import(&tree_a);
    let root_b = ctx.import(&tree_b);

    assert_eq!(root_a, root_b);
}

#[test]
fn shell_context_respects_remapped_coordinates() {
    let tree = Tree::line_loft_shell(shell()).remap_xyz(
        Tree::y(),
        Tree::x(),
        Tree::z(),
    );
    let mut ctx = Context::new();
    let root = ctx.import(&tree);

    assert_approx_eq(ctx.eval_xyz(root, 0.0, 1.0, 0.0).unwrap(), -1.0);
    assert_approx_eq(ctx.eval_xyz(root, 1.25, 1.0, 0.0).unwrap(), 0.25);
}

#[test]
fn shell_context_respects_affine_remap_coordinates() {
    let translate_x =
        nalgebra::convert(nalgebra::Translation3::<f64>::new(1.0, 0.0, 0.0));
    let tree = Tree::line_loft_shell(shell()).remap_affine(translate_x);
    let mut ctx = Context::new();
    let root = ctx.import(&tree);

    assert_approx_eq(ctx.eval_xyz(root, 0.0, 0.0, 0.0).unwrap(), -1.0);
    assert_approx_eq(ctx.eval_xyz(root, -1.25, 0.0, 0.0).unwrap(), 0.25);
}

#[test]
fn shell_context_composes_with_regular_csg() {
    let tree = Tree::line_loft_shell(shell()).max(0.25);
    let mut ctx = Context::new();
    let root = ctx.import(&tree);

    assert!(matches!(
        ctx.get_op(root),
        Some(Op::Binary(BinaryOpcode::Max, ..))
    ));
    assert_approx_eq(ctx.eval_xyz(root, 1.0, 0.0, 0.0).unwrap(), 0.25);
}

fn assert_approx_eq(left: f64, right: f64) {
    let diff = (left - right).abs();
    assert!(
        diff <= 1.0e-6,
        "expected {left} to be within tolerance of {right}; diff={diff}"
    );
}
