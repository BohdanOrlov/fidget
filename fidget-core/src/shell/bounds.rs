//! Conservative shell bounds.

/// Axis-aligned shell bounds in model space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellBounds {
    /// Minimum x coordinate.
    pub min_x: f32,
    /// Minimum y coordinate.
    pub min_y: f32,
    /// Minimum z coordinate.
    pub min_z: f32,
    /// Maximum x coordinate.
    pub max_x: f32,
    /// Maximum y coordinate.
    pub max_y: f32,
    /// Maximum z coordinate.
    pub max_z: f32,
}

impl ShellBounds {
    /// Builds empty bounds.
    pub fn empty() -> Self {
        Self {
            min_x: f32::INFINITY,
            min_y: f32::INFINITY,
            min_z: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            max_y: f32::NEG_INFINITY,
            max_z: f32::NEG_INFINITY,
        }
    }

    /// Expands the bounds to include a point.
    pub fn include_point(&mut self, x: f32, y: f32, z: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.min_z = self.min_z.min(z);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.max_z = self.max_z.max(z);
    }

    /// Expands the bounds to include another bounds box.
    pub fn include_bounds(&mut self, other: ShellBounds) {
        self.include_point(other.min_x, other.min_y, other.min_z);
        self.include_point(other.max_x, other.max_y, other.max_z);
    }

    /// Expands the bounds uniformly in every direction.
    pub fn inflate(self, amount: f32) -> Self {
        let amount = amount.max(0.0);
        Self {
            min_x: self.min_x - amount,
            min_y: self.min_y - amount,
            min_z: self.min_z - amount,
            max_x: self.max_x + amount,
            max_y: self.max_y + amount,
            max_z: self.max_z + amount,
        }
    }

    /// Returns true if all bounds are finite and ordered.
    pub fn is_valid(self) -> bool {
        self.min_x.is_finite()
            && self.min_y.is_finite()
            && self.min_z.is_finite()
            && self.max_x.is_finite()
            && self.max_y.is_finite()
            && self.max_z.is_finite()
            && self.min_x <= self.max_x
            && self.min_y <= self.max_y
            && self.min_z <= self.max_z
    }
}
