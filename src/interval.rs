use std::ops::{Add, Bound, Range, RangeBounds, RangeInclusive};
use num_traits::NumRef;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_derive", derive(serde::Deserialize, serde::Serialize))]
pub struct Interval<N>(RangeInclusive<N>);

impl<N: Ord> From<RangeInclusive<N>> for Interval<N> {
    fn from(range: RangeInclusive<N>) -> Self {
        if range.end() < range.start() {
            panic!("interval end bound must be >= start bound")
        }
        Self(range)
    }
}

impl<N: NumRef + Ord> From<Range<N>> for Interval<N> {
    fn from(range: Range<N>) -> Self {
        if range.end <= range.start {
            panic!("interval end bound must be > start bound")
        }
        Self(range.start..=range.end.sub(N::one()))
    }
}

impl<N: Ord + Clone> From<&'_ Interval<N>> for Interval<N> {
    fn from(interval: &Interval<N>) -> Self {
        interval.clone()
    }
}

impl<N: Ord + Clone> From<N> for Interval<N> {
    fn from(point: N) -> Self {
        Self::from(point.clone()..=point)
    }
}

impl<N> RangeBounds<N> for Interval<N> {
    fn start_bound(&self) -> Bound<&N> {
        self.0.start_bound()
    }

    fn end_bound(&self) -> Bound<&N> {
        self.0.end_bound()
    }
}

impl<N> Interval<N> {
    pub fn start(&self) -> &N {
        self.0.start()
    }

    pub fn end(&self) -> &N {
        self.0.end()
    }

    pub fn into_inner(self) -> (N, N) {
        self.0.into_inner()
    }
}

impl<N: Clone + Ord> Interval<N> {
    pub fn point(value: N) -> Self {
        Self::from(value.clone()..=value)
    }
}

impl<N: Ord> Interval<N> {
    pub fn contains(&self, other: &Interval<N>) -> bool {
        self.start() <= other.start() && self.end() >= other.end()
    }

    pub fn contains_point(&self, point: &N) -> bool {
        self.start() <= point && self.end() >= point
    }

    fn overlaps_aux(&self, other: &Interval<N>) -> bool {
        (self.start() <= other.start() && self.end() >= other.start())
            || (self.start() <= other.end() && self.end() >= other.end())
    }

    pub fn overlaps(&self, other: &Interval<N>) -> bool {
        self.overlaps_aux(other) || other.overlaps_aux(self)
    }
}

impl<'a, N> Interval<N>
where N: Ord + NumRef + 'a,
      &'a N: Add<usize, Output=N> {
    pub fn contains_range(&self, start: &'a N, size: usize) -> bool {
        let end = start + (size - 1);
        self.start() <= start && *self.end() >= end
    }
}
