//! Simple virtual machine for shape evaluation
use crate::{
    Context,
    compiler::RegOp,
    context::{BadNode, Node},
    eval::{
        BulkEvalError, BulkEvaluator, BulkOutput, Function, MathFunction, Tape,
        Trace, TracingEvalError, TracingEvaluator,
    },
    render::{
        NativeRenderMetadata, RenderHints, ShellActiveSegmentTraceSummary,
        TileSizes,
    },
    shape::Shape,
    shell::{
        ShellEvalScratch, ShellIntervalTrace, ShellTopology,
        eval_shell_distance, eval_shell_interval,
        eval_shell_interval_with_trace,
    },
    types::{Grad, Interval},
    var::VarMap,
};
use std::sync::Arc;

mod choice;
mod data;

pub use choice::Choice;
use data::BadChoiceSlice;
pub use data::{VmData, VmWorkspace};

////////////////////////////////////////////////////////////////////////////////

/// Function which uses the VM backend for evaluation
///
/// Internally, the [`VmFunction`] stores an [`Arc<VmData>`](VmData), and
/// iterates over a [`Vec<RegOp>`](RegOp) to perform evaluation.
///
/// All of the associated [`Tape`] types simply clone the internal `Arc`;
/// there's no separate planning required to generate a tape.
pub type VmFunction = GenericVmFunction<{ u8::MAX as usize }>;

/// Shape that uses the [`VmFunction`] backend for evaluation
pub type VmShape = Shape<VmFunction>;

/// Tape storage type which indicates that there's no actual backing storage
#[derive(Default)]
pub struct EmptyTapeStorage;

/// Tape which uses the VM backend for evaluation
///
/// This tape type is equivalent to a [`GenericVmFunction`], but implements
/// different traits ([`Tape`] instead of [`Function`]).
#[derive(Clone)]
pub struct GenericVmTape<const N: usize>(Arc<VmData<N>>);

impl<const N: usize> GenericVmTape<N> {
    /// Returns a handle to the inner [`VmData`] used by the tape
    pub fn data(&self) -> &VmData<N> {
        &self.0
    }
}

impl<const N: usize> Tape for GenericVmTape<N> {
    type Storage = EmptyTapeStorage;
    fn recycle(self) -> Option<Self::Storage> {
        Some(EmptyTapeStorage)
    }

    fn vars(&self) -> &VarMap {
        &self.0.vars
    }

    fn output_count(&self) -> usize {
        self.0.output_count()
    }
}

/// A trace captured by a VM evaluation
///
/// This stores regular min/max choices plus native shell trace payloads.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct VmTrace {
    choices: Vec<Choice>,
    shell_traces: Vec<Option<ShellIntervalTrace>>,
}

impl VmTrace {
    /// Fills the trace with the given value
    pub fn fill(&mut self, v: Choice) {
        self.choices.fill(v);
    }
    /// Resizes the trace, using the new value if it needs to be extended
    pub fn resize(&mut self, n: usize, v: Choice) {
        self.choices.resize(n, v);
    }
    /// Resizes the shell trace sidecar table and clears previous payloads.
    pub fn resize_shells(&mut self, n: usize) {
        self.shell_traces.clear();
        self.shell_traces.resize(n, None);
    }
    /// Records a shell interval trace for a topology sidecar.
    pub fn record_shell_trace(
        &mut self,
        shell: u32,
        trace: ShellIntervalTrace,
    ) {
        if let Some(slot) = self.shell_traces.get_mut(shell as usize) {
            *slot = Some(trace);
        }
    }
    /// Returns true if any shell trace can narrow a sidecar.
    pub fn has_shell_simplification(&self) -> bool {
        self.shell_traces
            .iter()
            .flatten()
            .any(shell_trace_can_simplify)
    }
    /// Returns shell trace payloads indexed by shell sidecar.
    pub fn shell_traces(&self) -> &[Option<ShellIntervalTrace>] {
        &self.shell_traces
    }
    /// Returns active segment trace summaries for render metadata.
    pub fn active_segment_trace_summaries(
        &self,
    ) -> Vec<ShellActiveSegmentTraceSummary> {
        self.shell_traces
            .iter()
            .enumerate()
            .filter_map(|(shell_index, trace)| {
                trace.map(|trace| ShellActiveSegmentTraceSummary {
                    shell_index: shell_index as u32,
                    active_segment_mask: trace.active_segment_mask,
                    segment_count: trace.segment_count,
                    sidecar_reduction_eligible: trace
                        .sidecar_reduction_eligible,
                })
            })
            .collect()
    }

    /// Returns the inner choice slice
    pub fn as_slice(&self) -> &[Choice] {
        self.choices.as_slice()
    }
    /// Returns the inner choice slice as a mutable reference
    pub fn as_mut_slice(&mut self) -> &mut [Choice] {
        self.choices.as_mut_slice()
    }
    /// Returns a pointer to the allocated choice array
    pub fn as_mut_ptr(&mut self) -> *mut Choice {
        self.choices.as_mut_ptr()
    }
}

impl Trace for VmTrace {
    fn copy_from(&mut self, other: &VmTrace) {
        self.choices.resize(other.choices.len(), Choice::Unknown);
        self.choices.copy_from_slice(&other.choices);
        self.shell_traces.clear();
        self.shell_traces.extend_from_slice(&other.shell_traces);
    }

    fn keep_simplified_shape(&self) -> bool {
        self.has_shell_simplification()
    }
}

#[cfg(any(test, feature = "eval-tests"))]
impl From<Vec<Choice>> for VmTrace {
    fn from(v: Vec<Choice>) -> Self {
        Self {
            choices: v,
            shell_traces: Vec::new(),
        }
    }
}

#[cfg(any(test, feature = "eval-tests"))]
impl AsRef<[Choice]> for VmTrace {
    fn as_ref(&self) -> &[Choice] {
        &self.choices
    }
}

fn shell_trace_can_simplify(trace: &ShellIntervalTrace) -> bool {
    let all_segments = if trace.segment_count >= u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << trace.segment_count) - 1
    };
    trace.sidecar_reduction_eligible
        && trace.active_segment_mask != 0
        && trace.active_segment_mask != all_segments
        && trace.segment_count > 1
}

/// VM-backed shape with a configurable number of registers
///
/// You are unlikely to use this directly; [`VmShape`] should be used for
/// VM-based evaluation.
#[derive(Clone)]
pub struct GenericVmFunction<const N: usize>(Arc<VmData<N>>);

impl<const N: usize> From<VmData<N>> for GenericVmFunction<N> {
    fn from(d: VmData<N>) -> Self {
        Self(d.into())
    }
}

impl<const N: usize> GenericVmFunction<N> {
    /// Returns a characteristic size (the length of the inner assembly tape)
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Reclaim the inner `VmData` if there's only a single reference
    pub fn recycle(self) -> Option<VmData<N>> {
        Arc::try_unwrap(self.0).ok()
    }

    /// Borrows the inner [`VmData`]
    pub fn data(&self) -> &VmData<N> {
        self.0.as_ref()
    }

    /// Returns a [`GenericVmTape`] for the given function
    pub fn tape(&self) -> GenericVmTape<N> {
        GenericVmTape(self.0.clone())
    }

    /// Returns the number of choices (i.e. `min` and `max` nodes) in the tape
    pub fn choice_count(&self) -> usize {
        self.0.choice_count()
    }

    /// Returns the number of outputs in the tape
    pub fn output_count(&self) -> usize {
        self.0.output_count()
    }

    /// Simplifies the function with the given trace and a new register count
    pub fn simplify_with<const M: usize>(
        &self,
        trace: &VmTrace,
        storage: VmData<M>,
        workspace: &mut VmWorkspace<M>,
    ) -> Result<GenericVmFunction<M>, BadTrace> {
        let d = self.0.simplify::<M>(
            trace.as_slice(),
            trace.shell_traces(),
            workspace,
            storage,
        )?;
        Ok(GenericVmFunction(Arc::new(d)))
    }
}

/// Error type for simplification
#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub struct BadTrace(#[from] pub BadChoiceSlice);

impl<const N: usize> Function for GenericVmFunction<N> {
    type Storage = VmData<N>;
    type Workspace = VmWorkspace<N>;

    type TapeStorage = EmptyTapeStorage;

    type FloatSliceEval = VmFloatSliceEval<N>;
    type GradSliceEval = VmGradSliceEval<N>;
    type PointEval = VmPointEval<N>;
    type IntervalEval = VmIntervalEval<N>;
    type Trace = VmTrace;

    #[inline]
    fn float_slice_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn grad_slice_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn point_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn interval_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Self::Storage,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, BadTrace> {
        self.simplify_with(trace, storage, workspace)
    }

    #[inline]
    fn recycle(self) -> Option<Self::Storage> {
        GenericVmFunction::recycle(self)
    }

    #[inline]
    fn native_render_metadata(&self) -> Option<NativeRenderMetadata> {
        self.0.native_render_metadata()
    }

    #[inline]
    fn has_native_render_metadata(&self) -> bool {
        !self.0.shell_topologies().is_empty()
    }

    fn size(&self) -> usize {
        GenericVmFunction::size(self)
    }

    #[inline]
    fn vars(&self) -> &VarMap {
        &self.0.vars
    }

    #[inline]
    fn can_simplify(&self) -> bool {
        self.0.choice_count() > 0
    }
}

impl<const N: usize> RenderHints for GenericVmFunction<N> {
    fn tile_sizes_3d() -> TileSizes {
        TileSizes::new(&[128, 64, 32, 16, 8]).unwrap()
    }

    fn tile_sizes_2d() -> TileSizes {
        TileSizes::new(&[128, 32, 8]).unwrap()
    }
}

impl<const N: usize> MathFunction for GenericVmFunction<N> {
    fn new(ctx: &Context, nodes: &[Node]) -> Result<Self, BadNode> {
        let d = VmData::new(ctx, nodes)?;
        Ok(Self(d.into()))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Helper struct to reduce boilerplate conversions
struct SlotArray<'a, T>(&'a mut [T]);
impl<T> std::ops::Index<u8> for SlotArray<'_, T> {
    type Output = T;
    fn index(&self, i: u8) -> &Self::Output {
        &self.0[i as usize]
    }
}
impl<T> std::ops::IndexMut<u8> for SlotArray<'_, T> {
    fn index_mut(&mut self, i: u8) -> &mut T {
        &mut self.0[i as usize]
    }
}
impl<T> std::ops::Index<u32> for SlotArray<'_, T> {
    type Output = T;
    fn index(&self, i: u32) -> &Self::Output {
        &self.0[i as usize]
    }
}
impl<T> std::ops::IndexMut<u32> for SlotArray<'_, T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        &mut self.0[i as usize]
    }
}

////////////////////////////////////////////////////////////////////////////////

fn eval_shell_sample(
    shell: &ShellTopology,
    scratch: &mut ShellEvalScratch,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    eval_shell_distance(
        shell,
        crate::shell::ShellParamsView::empty(),
        scratch,
        x,
        y,
        z,
    )
    .distance
}

/// Generic VM evaluator for tracing evaluation
struct TracingVmEval<T> {
    slots: Vec<T>,
    out: Vec<T>,
    choices: VmTrace,
    shell_scratch: ShellEvalScratch,
}

impl<T> Default for TracingVmEval<T> {
    fn default() -> Self {
        Self {
            slots: Vec::default(),
            out: Vec::default(),
            choices: VmTrace::default(),
            shell_scratch: ShellEvalScratch::default(),
        }
    }
}

impl<T: From<f32> + Clone> TracingVmEval<T> {
    fn resize_slots<const N: usize>(&mut self, tape: &VmData<N>) {
        self.slots.resize(tape.slot_count(), f32::NAN.into());
        self.choices.resize(tape.choice_count(), Choice::Unknown);
        self.choices.resize_shells(tape.shell_topologies().len());
        self.out.resize(tape.output_count(), f32::NAN.into());
        self.choices.fill(Choice::Unknown);
    }
}

/// VM-based tracing evaluator for intervals
#[derive(Default)]
pub struct VmIntervalEval<const N: usize>(TracingVmEval<Interval>);
impl<const N: usize> TracingEvaluator for VmIntervalEval<N> {
    type Data = Interval;
    type Tape = GenericVmTape<N>;
    type Trace = VmTrace;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval(
        &mut self,
        tape: &Self::Tape,
        vars: &[Interval],
    ) -> Result<(&[Interval], Option<&VmTrace>), TracingEvalError> {
        tape.vars().check_tracing_arguments(vars)?;
        let tape = tape.data();
        self.0.resize_slots(tape);

        let mut simplify = false;
        let mut v = SlotArray(&mut self.0.slots);
        let trace = &mut self.0.choices;
        let shell_traces = &mut trace.shell_traces;
        let mut choices = trace.choices.as_mut_slice().iter_mut();
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize] = v[arg];
                }
                RegOp::Input(out, i) => {
                    v[out] = vars[i as usize];
                }
                RegOp::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                RegOp::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                RegOp::RecipReg(out, arg) => {
                    v[out] = v[arg].recip();
                }
                RegOp::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                RegOp::SquareReg(out, arg) => {
                    v[out] = v[arg].square();
                }
                RegOp::FloorReg(out, arg) => {
                    v[out] = v[arg].floor();
                }
                RegOp::CeilReg(out, arg) => {
                    v[out] = v[arg].ceil();
                }
                RegOp::RoundReg(out, arg) => {
                    v[out] = v[arg].round();
                }
                RegOp::SinReg(out, arg) => {
                    v[out] = v[arg].sin();
                }
                RegOp::CosReg(out, arg) => {
                    v[out] = v[arg].cos();
                }
                RegOp::TanReg(out, arg) => {
                    v[out] = v[arg].tan();
                }
                RegOp::AsinReg(out, arg) => {
                    v[out] = v[arg].asin();
                }
                RegOp::AcosReg(out, arg) => {
                    v[out] = v[arg].acos();
                }
                RegOp::AtanReg(out, arg) => {
                    v[out] = v[arg].atan();
                }
                RegOp::ExpReg(out, arg) => {
                    v[out] = v[arg].exp();
                }
                RegOp::LnReg(out, arg) => {
                    v[out] = v[arg].ln();
                }
                RegOp::NotReg(out, arg) => {
                    v[out] = if !v[arg].contains(0.0) && !v[arg].has_nan() {
                        Interval::new(0.0, 0.0)
                    } else if v[arg].lower() == 0.0 && v[arg].upper() == 0.0 {
                        Interval::new(1.0, 1.0)
                    } else {
                        Interval::new(0.0, 1.0)
                    };
                }
                RegOp::CopyReg(out, arg) => v[out] = v[arg],
                RegOp::ShellDistance(out, shell, x, y, z) => {
                    let shell_index = shell;
                    let shell = tape.shell_topology(shell_index).expect(
                        "shell sidecar should exist during interval eval",
                    );
                    if shell.profile.is_some() {
                        v[out] = eval_shell_interval(shell, v[x], v[y], v[z]);
                        continue;
                    }
                    let (interval, shell_trace) =
                        eval_shell_interval_with_trace(shell, v[x], v[y], v[z]);
                    v[out] = interval;
                    if let Some(slot) =
                        shell_traces.get_mut(shell_index as usize)
                    {
                        *slot = Some(shell_trace);
                    }
                    simplify |= shell_trace_can_simplify(&shell_trace);
                }
                RegOp::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm.into();
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm;
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm.into();
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    let imm: Interval = imm.into();
                    v[out] = imm / v[arg];
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    v[out] = v[arg].atan2(imm.into());
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    let imm: Interval = imm.into();
                    v[out] = imm.atan2(v[arg]);
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].atan2(v[rhs]);
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    v[out] = Interval::from(imm) - v[arg];
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm.into();
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].min_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].and_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].and_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].or_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].or_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].rem_euclid(v[rhs]);
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    v[out] = v[arg].rem_euclid(imm.into());
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    v[out] = Interval::from(imm).rem_euclid(v[arg]);
                }
                RegOp::AddRegReg(out, lhs, rhs) => v[out] = v[lhs] + v[rhs],
                RegOp::MulRegReg(out, lhs, rhs) => v[out] = v[lhs] * v[rhs],
                RegOp::DivRegReg(out, lhs, rhs) => v[out] = v[lhs] / v[rhs],
                RegOp::SubRegReg(out, lhs, rhs) => v[out] = v[lhs] - v[rhs],
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    v[out] = if v[lhs].has_nan() || v[rhs].has_nan() {
                        f32::NAN.into()
                    } else if v[lhs].upper() < v[rhs].lower() {
                        Interval::from(-1.0)
                    } else if v[lhs].lower() > v[rhs].upper() {
                        Interval::from(1.0)
                    } else {
                        Interval::new(-1.0, 1.0)
                    };
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    v[out] = if v[arg].has_nan() || imm.is_nan() {
                        f32::NAN.into()
                    } else if v[arg].upper() < imm {
                        Interval::from(-1.0)
                    } else if v[arg].lower() > imm {
                        Interval::from(1.0)
                    } else {
                        Interval::new(-1.0, 1.0)
                    };
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    v[out] = if v[arg].has_nan() || imm.is_nan() {
                        f32::NAN.into()
                    } else if imm < v[arg].lower() {
                        Interval::from(-1.0)
                    } else if imm > v[arg].upper() {
                        Interval::from(1.0)
                    } else {
                        Interval::new(-1.0, 1.0)
                    };
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::CopyImm(out, imm) => {
                    v[out] = imm.into();
                }
                RegOp::Load(out, mem) => {
                    v[out] = v[mem];
                }
                RegOp::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        Ok((
            &self.0.out,
            if simplify {
                Some(&self.0.choices)
            } else {
                None
            },
        ))
    }
}

/// VM-based tracing evaluator for single points
#[derive(Default)]
pub struct VmPointEval<const N: usize>(TracingVmEval<f32>);
impl<const N: usize> TracingEvaluator for VmPointEval<N> {
    type Data = f32;
    type Tape = GenericVmTape<N>;
    type Trace = VmTrace;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval(
        &mut self,
        tape: &Self::Tape,
        vars: &[f32],
    ) -> Result<(&[f32], Option<&VmTrace>), TracingEvalError> {
        tape.vars().check_tracing_arguments(vars)?;
        let tape = tape.data();
        self.0.resize_slots(tape);

        let mut choices = self.0.choices.as_mut_slice().iter_mut();
        let mut simplify = false;
        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize] = v[arg];
                }
                RegOp::Input(out, i) => {
                    v[out] = vars[i as usize];
                }
                RegOp::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                RegOp::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                RegOp::RecipReg(out, arg) => {
                    v[out] = 1.0 / v[arg];
                }
                RegOp::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                RegOp::SquareReg(out, arg) => {
                    let s = v[arg];
                    v[out] = s * s;
                }
                RegOp::FloorReg(out, arg) => {
                    v[out] = v[arg].floor();
                }
                RegOp::CeilReg(out, arg) => {
                    v[out] = v[arg].ceil();
                }
                RegOp::RoundReg(out, arg) => {
                    v[out] = v[arg].round();
                }
                RegOp::SinReg(out, arg) => {
                    v[out] = v[arg].sin();
                }
                RegOp::CosReg(out, arg) => {
                    v[out] = v[arg].cos();
                }
                RegOp::TanReg(out, arg) => {
                    v[out] = v[arg].tan();
                }
                RegOp::AsinReg(out, arg) => {
                    v[out] = v[arg].asin();
                }
                RegOp::AcosReg(out, arg) => {
                    v[out] = v[arg].acos();
                }
                RegOp::AtanReg(out, arg) => {
                    v[out] = v[arg].atan();
                }
                RegOp::ExpReg(out, arg) => {
                    v[out] = v[arg].exp();
                }
                RegOp::LnReg(out, arg) => {
                    v[out] = v[arg].ln();
                }
                RegOp::NotReg(out, arg) => v[out] = (v[arg] == 0.0).into(),
                RegOp::CopyReg(out, arg) => {
                    v[out] = v[arg];
                }
                RegOp::ShellDistance(out, shell, x, y, z) => {
                    let shell = tape
                        .shell_topology(shell)
                        .expect("shell sidecar should exist during point eval");
                    v[out] = eval_shell_sample(
                        shell,
                        &mut self.0.shell_scratch,
                        v[x],
                        v[y],
                        v[z],
                    );
                }
                RegOp::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm;
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm;
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm;
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    v[out] = imm / v[arg];
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    v[out] = v[arg].atan2(imm);
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    v[out] = imm.atan2(v[arg]);
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].atan2(v[rhs]);
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    v[out] = imm - v[arg];
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm;
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let a = v[arg];
                    let (choice, value) = if a < imm {
                        (Choice::Left, a)
                    } else if imm < a {
                        (Choice::Right, imm)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || imm.is_nan() {
                                f32::NAN
                            } else {
                                imm
                            },
                        )
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let a = v[arg];
                    let (choice, value) = if a > imm {
                        (Choice::Left, a)
                    } else if imm > a {
                        (Choice::Right, imm)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || imm.is_nan() {
                                f32::NAN
                            } else {
                                imm
                            },
                        )
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    let a = v[arg];
                    let (choice, value) = if a == 0.0 {
                        (Choice::Left, a)
                    } else {
                        (Choice::Right, imm)
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    let a = v[arg];
                    let (choice, value) = if a != 0.0 {
                        (Choice::Left, a)
                    } else {
                        (Choice::Right, imm)
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].rem_euclid(v[rhs]);
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    v[out] = v[arg].rem_euclid(imm);
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    v[out] = imm.rem_euclid(v[arg]);
                }
                RegOp::AddRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] + v[rhs];
                }
                RegOp::MulRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] * v[rhs];
                }
                RegOp::DivRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] / v[rhs];
                }
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs]
                        .partial_cmp(&v[rhs])
                        .map(|c| c as i8 as f32)
                        .unwrap_or(f32::NAN)
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    v[out] = v[arg]
                        .partial_cmp(&imm)
                        .map(|c| c as i8 as f32)
                        .unwrap_or(f32::NAN)
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    v[out] = imm
                        .partial_cmp(&v[arg])
                        .map(|c| c as i8 as f32)
                        .unwrap_or(f32::NAN)
                }
                RegOp::SubRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] - v[rhs];
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    let (choice, value) = if a < b {
                        (Choice::Left, a)
                    } else if b < a {
                        (Choice::Right, b)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || b.is_nan() {
                                f32::NAN
                            } else {
                                b
                            },
                        )
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    let (choice, value) = if a > b {
                        (Choice::Left, a)
                    } else if b > a {
                        (Choice::Right, b)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || b.is_nan() {
                                f32::NAN
                            } else {
                                b
                            },
                        )
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    let (choice, value) = if a == 0.0 {
                        (Choice::Left, a)
                    } else {
                        (Choice::Right, b)
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    let (choice, value) = if a != 0.0 {
                        (Choice::Left, a)
                    } else {
                        (Choice::Right, b)
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::CopyImm(out, imm) => {
                    v[out] = imm;
                }
                RegOp::Load(out, mem) => {
                    v[out] = v[mem];
                }
                RegOp::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        Ok((
            &self.0.out,
            if simplify {
                Some(&self.0.choices)
            } else {
                None
            },
        ))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Bulk evaluator for VM tapes
#[derive(Default)]
struct BulkVmEval<T> {
    /// Workspace for data
    slots: Vec<Vec<T>>,

    /// Output array
    out: Vec<Vec<T>>,

    /// Reusable native shell scratch.
    shell_scratch: ShellEvalScratch,
}

impl<T: From<f32> + Clone> BulkVmEval<T> {
    /// Reserves slots for the given tape and slice size
    fn resize_slots<const N: usize>(&mut self, tape: &VmData<N>, size: usize) {
        self.slots
            .resize_with(tape.slot_count(), || vec![f32::NAN.into(); size]);
        for s in self.slots.iter_mut() {
            s.resize(size, f32::NAN.into());
        }

        self.out
            .resize_with(tape.output_count(), || vec![f32::NAN.into(); size]);
        for o in self.out.iter_mut() {
            o.resize(size, f32::NAN.into());
        }
    }
}

/// VM-based bulk evaluator for arrays of points, yielding point values
#[derive(Default)]
pub struct VmFloatSliceEval<const N: usize>(BulkVmEval<f32>);
impl<const N: usize> BulkEvaluator for VmFloatSliceEval<N> {
    type Data = f32;
    type Tape = GenericVmTape<N>;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<BulkOutput<'_, f32>, BulkEvalError> {
        tape.vars().check_bulk_arguments(vars)?;
        let tape = tape.data();

        let size = vars.first().map(|v| v.len()).unwrap_or(0);
        self.0.resize_slots(tape, size);

        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize][0..size]
                        .copy_from_slice(&v[arg][0..size]);
                }
                RegOp::Input(out, i) => {
                    v[out][0..size].copy_from_slice(&vars[i as usize]);
                }
                RegOp::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                RegOp::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                RegOp::RecipReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = 1.0 / v[arg][i];
                    }
                }
                RegOp::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                RegOp::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                RegOp::FloorReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].floor();
                    }
                }
                RegOp::CeilReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ceil();
                    }
                }
                RegOp::RoundReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].round();
                    }
                }
                RegOp::SinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sin();
                    }
                }
                RegOp::CosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].cos();
                    }
                }
                RegOp::TanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].tan();
                    }
                }
                RegOp::AsinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].asin();
                    }
                }
                RegOp::AcosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].acos();
                    }
                }
                RegOp::AtanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan();
                    }
                }
                RegOp::ExpReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].exp();
                    }
                }
                RegOp::LnReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ln();
                    }
                }
                RegOp::NotReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = (v[arg][i] == 0.0).into();
                    }
                }
                RegOp::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                RegOp::ShellDistance(out, shell, x, y, z) => {
                    let shell = tape
                        .shell_topology(shell)
                        .expect("shell sidecar should exist during float eval");
                    for i in 0..size {
                        v[out][i] = eval_shell_sample(
                            shell,
                            &mut self.0.shell_scratch,
                            v[x][i],
                            v[y][i],
                            v[z][i],
                        );
                    }
                }
                RegOp::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm;
                    }
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm;
                    }
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan2(imm);
                    }
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm.atan2(v[arg][i]);
                    }
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].atan2(v[rhs][i]);
                    }
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm
                            .partial_cmp(&v[arg][i])
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN)
                    }
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i]
                            .partial_cmp(&imm)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN)
                    }
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = if v[arg][i].is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            v[arg][i].min(imm)
                        };
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = if v[arg][i].is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            v[arg][i].max(imm)
                        };
                    }
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] =
                            if v[arg][i] == 0.0 { v[arg][i] } else { imm };
                    }
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] =
                            if v[arg][i] != 0.0 { v[arg][i] } else { imm };
                    }
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].rem_euclid(v[rhs][i]);
                    }
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].rem_euclid(imm);
                    }
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm.rem_euclid(v[arg][i]);
                    }
                }
                RegOp::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                RegOp::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                RegOp::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                RegOp::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i]
                            .partial_cmp(&v[rhs][i])
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN)
                    }
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i].is_nan() || v[rhs][i].is_nan()
                        {
                            f32::NAN
                        } else {
                            v[lhs][i].min(v[rhs][i])
                        };
                    }
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i].is_nan() || v[rhs][i].is_nan()
                        {
                            f32::NAN
                        } else {
                            v[lhs][i].max(v[rhs][i])
                        };
                    }
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i] == 0.0 {
                            v[lhs][i]
                        } else {
                            v[rhs][i]
                        };
                    }
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i] != 0.0 {
                            v[lhs][i]
                        } else {
                            v[rhs][i]
                        };
                    }
                }
                RegOp::CopyImm(out, imm) => {
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                RegOp::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                RegOp::Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        Ok(BulkOutput::new(&self.0.out, size))
    }
}

/// VM-based bulk evaluator for arrays of points, yielding gradient values
#[derive(Default)]
pub struct VmGradSliceEval<const N: usize>(BulkVmEval<Grad>);
impl<const N: usize> BulkEvaluator for VmGradSliceEval<N> {
    type Data = Grad;
    type Tape = GenericVmTape<N>;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<BulkOutput<'_, Grad>, BulkEvalError> {
        tape.vars().check_bulk_arguments(vars)?;
        let tape = tape.data();
        let size = vars.first().map(|v| v.len()).unwrap_or(0);
        self.0.resize_slots(tape, size);

        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize][0..size]
                        .copy_from_slice(&v[arg][0..size]);
                }
                RegOp::Input(out, i) => {
                    v[out][0..size].copy_from_slice(&vars[i as usize]);
                }
                RegOp::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                RegOp::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                RegOp::RecipReg(out, arg) => {
                    let one: Grad = 1.0.into();
                    for i in 0..size {
                        v[out][i] = one / v[arg][i];
                    }
                }
                RegOp::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                RegOp::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                RegOp::FloorReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].floor();
                    }
                }
                RegOp::CeilReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ceil();
                    }
                }
                RegOp::RoundReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].round();
                    }
                }
                RegOp::SinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sin();
                    }
                }
                RegOp::CosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].cos();
                    }
                }
                RegOp::TanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].tan();
                    }
                }
                RegOp::AsinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].asin();
                    }
                }
                RegOp::AcosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].acos();
                    }
                }
                RegOp::AtanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan();
                    }
                }
                RegOp::ExpReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].exp();
                    }
                }
                RegOp::LnReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ln();
                    }
                }
                RegOp::NotReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = f32::from(v[arg][i].v == 0.0).into();
                    }
                }
                RegOp::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                RegOp::ShellDistance(out, shell, x, y, z) => {
                    let shell = tape
                        .shell_topology(shell)
                        .expect("shell sidecar should exist during grad eval");
                    for i in 0..size {
                        v[out][i] = crate::shell::eval_shell_grad(
                            shell,
                            crate::shell::ShellParamsView::empty(),
                            &mut self.0.shell_scratch,
                            v[x][i],
                            v[y][i],
                            v[z][i],
                        );
                    }
                }
                RegOp::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm.into();
                    }
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm.into();
                    }
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    let imm = Grad::from(imm);
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    let imm = Grad::from(imm);
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan2(imm);
                    }
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    let imm = Grad::from(imm);
                    for i in 0..size {
                        v[out][i] = imm.atan2(v[arg][i]);
                    }
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].atan2(v[rhs][i]);
                    }
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    for i in 0..size {
                        let p = imm
                            .partial_cmp(&v[arg][i].v)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN);
                        v[out][i] = Grad::new(p, 0.0, 0.0, 0.0);
                    }
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    for i in 0..size {
                        let p = v[arg][i]
                            .v
                            .partial_cmp(&imm)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN);
                        v[out][i] = Grad::new(p, 0.0, 0.0, 0.0);
                    }
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = if v[arg][i].v.is_nan() || imm.v.is_nan() {
                            f32::NAN.into()
                        } else {
                            v[arg][i].min(imm)
                        };
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = if v[arg][i].v.is_nan() || imm.v.is_nan() {
                            f32::NAN.into()
                        } else {
                            v[arg][i].max(imm)
                        };
                    }
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].rem_euclid(v[rhs][i]);
                    }
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].rem_euclid(imm.into());
                    }
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = Grad::from(imm).rem_euclid(v[arg][i]);
                    }
                }
                RegOp::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                RegOp::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i].v == 0.0 {
                            v[lhs][i]
                        } else {
                            v[rhs][i]
                        };
                    }
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = if v[arg][i].v == 0.0 {
                            v[arg][i]
                        } else {
                            imm.into()
                        };
                    }
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i].v != 0.0 {
                            v[lhs][i]
                        } else {
                            v[rhs][i]
                        };
                    }
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = if v[arg][i].v != 0.0 {
                            v[arg][i]
                        } else {
                            imm.into()
                        };
                    }
                }
                RegOp::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                RegOp::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        let p = v[lhs][i]
                            .v
                            .partial_cmp(&v[rhs][i].v)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN);
                        v[out][i] = Grad::new(p, 0.0, 0.0, 0.0);
                    }
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] =
                            if v[lhs][i].v.is_nan() || v[rhs][i].v.is_nan() {
                                f32::NAN.into()
                            } else {
                                v[lhs][i].min(v[rhs][i])
                            };
                    }
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] =
                            if v[lhs][i].v.is_nan() || v[rhs][i].v.is_nan() {
                                f32::NAN.into()
                            } else {
                                v[lhs][i].max(v[rhs][i])
                            };
                    }
                }
                RegOp::CopyImm(out, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                RegOp::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                RegOp::Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        Ok(BulkOutput::new(&self.0.out, size))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(VmFunction);
    crate::interval_tests!(VmFunction);
    crate::float_slice_tests!(VmFunction);
    crate::point_tests!(VmFunction);

    #[test]
    fn vm_native_shell_render_metadata_reports_bounds_and_segments() {
        use crate::{
            context::Tree,
            shell::{ShellSectionTopology, ShellTopology},
        };

        let shell = Arc::new(ShellTopology::line_loft_circles(
            vec![
                ShellSectionTopology::circle(0.0, 0.0, 0.0, 1.0),
                ShellSectionTopology::circle(1.0, 0.0, 0.0, 1.0),
                ShellSectionTopology::circle(2.0, 0.0, 0.0, 1.0),
            ]
            .into_boxed_slice(),
        ));
        let tree = Tree::line_loft_shell(shell);
        let mut ctx = Context::new();
        let root = ctx.import(&tree);
        let function = VmFunction::new(&ctx, &[root]).unwrap();
        let metadata = function.native_render_metadata().unwrap();

        let global = metadata.global_aabb.unwrap();
        assert_eq!(metadata.shell_segment_aabbs.len(), 2);
        assert!(global.min_x <= 0.0 && global.max_x >= 2.0);
        assert!(global.min_y <= -1.0 && global.max_y >= 1.0);
        assert_eq!(metadata.shell_segment_aabbs[0].shell_index, 0);
        assert_eq!(metadata.shell_segment_aabbs[0].segment_index, 0);
        assert!(metadata.shell_segment_aabbs[0].bounds.max_x <= 1.0);
        assert_eq!(metadata.shell_segment_aabbs[1].segment_index, 1);
        assert!(metadata.shell_segment_aabbs[1].bounds.min_x >= 1.0);
    }

    #[test]
    fn vm_trace_reports_active_segment_summaries() {
        let mut trace = VmTrace::default();
        trace.resize_shells(2);
        trace.record_shell_trace(
            1,
            ShellIntervalTrace {
                active_segment_mask: 0b010,
                segment_count: 3,
                sidecar_reduction_eligible: true,
            },
        );

        let summaries = trace.active_segment_trace_summaries();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].shell_index, 1);
        assert_eq!(summaries[0].active_segment_mask, 0b010);
        assert_eq!(summaries[0].segment_count, 3);
        assert!(summaries[0].sidecar_reduction_eligible);
    }

    #[test]
    fn vm_3d_tile_hints_preserve_general_render_defaults() {
        let sizes = VmFunction::tile_sizes_3d()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(sizes, [128, 64, 32, 16, 8]);
    }
}
