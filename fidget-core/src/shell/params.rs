//! Runtime shell parameter views.

/// Immutable shell parameter layout metadata.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShellParamLayout {
    /// Number of live parameter slots consumed by this shell topology.
    pub parameter_count: usize,
}

/// Borrowed view of live shell parameter values.
#[derive(Clone, Copy, Debug)]
pub struct ShellParamsView<'a> {
    values: &'a [f32],
}

impl<'a> ShellParamsView<'a> {
    /// Builds a parameter view.
    pub fn new(values: &'a [f32]) -> Self {
        Self { values }
    }

    /// Builds an empty parameter view.
    pub fn empty() -> Self {
        Self { values: &[] }
    }

    /// Reads a parameter, falling back to the supplied default when the slot is
    /// absent.
    pub fn get(self, slot: Option<usize>, default: f32) -> f32 {
        slot.and_then(|index| self.values.get(index).copied())
            .unwrap_or(default)
    }

    /// Returns the number of parameter values in this view.
    pub fn len(self) -> usize {
        self.values.len()
    }

    /// Returns true when this view contains no parameter values.
    pub fn is_empty(self) -> bool {
        self.values.is_empty()
    }
}
