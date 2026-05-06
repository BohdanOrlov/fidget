//! Native shell and loft math kernels.
//!
//! This module is intentionally independent from `Tree`, `Context`, and tape
//! lowering.  It provides deterministic pure kernels that later native shell
//! operations can call from VM/JIT evaluation paths.

pub mod bounds;
pub mod eval;
pub mod interval;
pub mod params;
pub mod topology;

pub use bounds::ShellBounds;
pub use eval::{
    ShellEvalScratch, ShellEvalStats, ShellSample, eval_shell_distance,
    reset_shell_eval_stats, set_shell_eval_stats_enabled, shell_eval_stats,
};
pub use interval::eval_shell_interval;
pub use params::{ShellParamLayout, ShellParamsView};
pub use topology::{
    OpenTopPolicy, SHELL_MAX_CANDIDATES, SHELL_MAX_CURVES,
    SHELL_MAX_NODES_PER_CURVE, ShellCubicCoefficients, ShellOpKind,
    ShellProfileNodeContinuity, ShellProfileNodeTopology,
    ShellProfileSectionTopology, ShellProfileSegmentTopology,
    ShellProfileSpanInterpolation, ShellProfileTopology, ShellSectionTopology,
    ShellSegmentInterpolation, ShellSegmentTopology, ShellTopology,
};
