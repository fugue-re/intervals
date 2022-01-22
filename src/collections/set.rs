use std::borrow::Borrow;
use std::iter::FromIterator;

use crate::interval::Interval;

use super::{DisjointIntervalTree, IntervalTree};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Entry<'a, N: Ord + Clone>(super::Entry<'a, N, ()>);

impl<'a, N: Ord + Clone + 'a> From<super::Entry<'a, N, ()>> for Entry<'a, N> {
    #[inline(always)]
    fn from(e: super::Entry<'a, N, ()>) -> Self {
        Self(e)
    }
}

impl<'a, N: Ord + Clone + 'a> Entry<'a, N> {
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.0.index()
    }

    #[inline(always)]
    pub fn interval(&self) -> &'a Interval<N> {
        self.0.interval()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DisjointIntervalSet<N: Ord + Clone>(DisjointIntervalTree<N, ()>);

impl<N: Ord + Clone> Default for DisjointIntervalSet<N> {
    #[inline(always)]
    fn default() -> Self {
        Self(DisjointIntervalTree::default())
    }
}

impl<N, V> FromIterator<V> for DisjointIntervalSet<N>
where
    V: Into<Interval<N>>,
    N: Ord + Clone,
{
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut set = Self::new();
        iter.into_iter()
            .for_each(|interval| set.insert(interval));
        set.index();
        set
    }
}

impl<N: Ord + Clone> DisjointIntervalSet<N> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn insert<K: Into<Interval<N>>>(&mut self, interval: K) {
        self.0.insert(interval, ());
    }

    #[inline(always)]
    pub fn extend<I, K>(&mut self, intervals: I)
    where K: Into<Interval<N>>,
          I: IntoIterator<Item=K> {
        self.0.extend(intervals.into_iter().map(|k| (k, ())))
    }

    #[inline(always)]
    fn index(&mut self) {
        self.0.index()
    }

    #[inline(always)]
    pub fn overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> bool {
        self.0.overlaps(interval)
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<Entry<N>> {
        self.0.get(index).map(Entry::from)
    }

    #[inline(always)]
    pub fn find<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Option<Entry<N>> {
        self.0.find(interval).map(Entry::from)
    }

    #[inline(always)]
    pub fn find_all<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Vec<Entry<N>> {
        self.0.find_all(interval).into_iter().map(Entry::from).collect()
    }

    #[inline(always)]
    pub fn remove_overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) {
        self.0.remove_overlaps(interval)
    }

    #[inline(always)]
    pub fn contains_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> bool {
        self.0.contains_point(point)
    }

    #[inline(always)]
    pub fn find_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> Option<Entry<N>> {
        self.0.find_point(point).map(Entry::from)
    }

    #[inline(always)]
    pub fn find_exact<M: Borrow<N>, I: Into<Interval<M>>>(
        &self,
        interval: I,
    ) -> Option<Entry<N>> {
        self.0.find_exact(interval).map(Entry::from)
    }

    #[inline(always)]
    pub fn remove_exact<M: Borrow<N>, I: Into<Interval<M>>>(
        &mut self,
        interval: I,
    ) {
        self.0.remove_exact(interval)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item=&Interval<N>> {
        self.0.iter().map(|(e, _)| e)
    }

    #[inline(always)]
    pub fn into_iter(self) -> impl Iterator<Item=Interval<N>> {
        self.0.into_iter().map(|(e, _)| e)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IntervalSet<N: Ord + Clone>(IntervalTree<N, ()>);

impl<N: Ord + Clone> Default for IntervalSet<N> {
    #[inline(always)]
    fn default() -> Self {
        Self(IntervalTree::default())
    }
}

impl<N, V> FromIterator<V> for IntervalSet<N>
where
    V: Into<Interval<N>>,
    N: Ord + Clone,
{
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut set = Self::new();
        iter.into_iter()
            .for_each(|interval| set.insert(interval));
        set.index();
        set
    }
}

impl<N: Ord + Clone> IntervalSet<N> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn insert<K: Into<Interval<N>>>(&mut self, interval: K) {
        self.0.insert(interval, ());
    }

    #[inline(always)]
    pub fn extend<I, K>(&mut self, intervals: I)
    where K: Into<Interval<N>>,
          I: IntoIterator<Item=K> {
        self.0.extend(intervals.into_iter().map(|k| (k, ())))
    }

    #[inline(always)]
    fn index(&mut self) {
        self.0.index()
    }

    #[inline(always)]
    pub fn overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> bool {
        self.0.overlaps(interval)
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<Entry<N>> {
        self.0.get(index).map(Entry::from)
    }

    #[inline(always)]
    pub fn find<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Option<Entry<N>> {
        self.0.find(interval).map(Entry::from)
    }

    #[inline(always)]
    pub fn find_all<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Vec<Entry<N>> {
        self.0.find_all(interval).into_iter().map(Entry::from).collect()
    }

    #[inline(always)]
    pub fn remove_overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) {
        self.0.remove_overlaps(interval)
    }

    #[inline(always)]
    pub fn contains_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> bool {
        self.0.contains_point(point)
    }

    #[inline(always)]
    pub fn find_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> Option<Entry<N>> {
        self.0.find_point(point).map(Entry::from)
    }

    #[inline(always)]
    pub fn find_exact<M: Borrow<N>, I: Into<Interval<M>>>(
        &self,
        interval: I,
    ) -> Option<Entry<N>> {
        self.0.find_exact(interval).map(Entry::from)
    }

    #[inline(always)]
    pub fn remove_exact<M: Borrow<N>, I: Into<Interval<M>>>(
        &mut self,
        interval: I,
    ) {
        self.0.remove_exact(interval)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item=&Interval<N>> {
        self.0.iter().map(|(e, _)| e)
    }

    #[inline(always)]
    pub fn into_iter(self) -> impl Iterator<Item=Interval<N>> {
        self.0.into_iter().map(|(e, _)| e)
    }
}
