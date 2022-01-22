use std::borrow::Borrow;
use std::cmp::{min, Ordering};
use std::hash::Hash;
use std::iter::FromIterator;

use num_traits::NumRef;

use crate::Interval;
use super::entry::*;

// TODO: refactor to use trait for common interval-tree functionality

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_derive", derive(serde::Deserialize, serde::Serialize))]
pub struct IntervalTree<N: Ord + Clone, D> {
    entries: Vec<InternalEntry<N, D>>,
    max_level: usize,
}

impl<N: Ord + Clone, D> Default for IntervalTree<N, D> {
    fn default() -> Self {
        IntervalTree {
            entries: vec![],
            max_level: 0,
        }
    }
}

impl<N, D, V> FromIterator<(V, D)> for IntervalTree<N, D>
where
    V: Into<Interval<N>>,
    N: Ord + Clone,
{
    fn from_iter<T: IntoIterator<Item = (V, D)>>(iter: T) -> Self {
        let mut tree = Self::new();
        iter.into_iter()
            .for_each(|(interval, data)| tree.insert(interval, data));
        tree.index();
        tree
    }
}

impl<N: Ord + Clone, D> IntervalTree<N, D> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert<K: Into<Interval<N>>, V: Into<D>>(&mut self, interval: K, data: V) {
        let interval = interval.into();
        let data = data.into();

        let max = interval.end().clone();
        self.entries.push(InternalEntry {
            interval,
            data,
            max,
        });
        self.index();
    }

    pub fn extend<I, K, V>(&mut self, intervals: I)
    where K: Into<Interval<N>>,
          V: Into<D>,
          I: IntoIterator<Item=(K, V)> {
        let it = intervals.into_iter();
        let (lb, ub) = it.size_hint();
        self.entries.reserve(ub.unwrap_or(lb));

        for (interval, data) in it.map(|(k, v)| (k.into(), v.into())) {
            let max = interval.end().clone();
            self.entries.push(InternalEntry {
                interval,
                data,
                max,
            });
        }
        self.index();
    }

    pub(crate) fn index(&mut self) {
        self.entries.sort_by(|l, r| l.interval.start().cmp(&r.interval.start()));
        self.index_core();
    }

    fn index_core(&mut self) {
        let a = &mut self.entries;
        if a.is_empty() {
            return;
        }

        let n = a.len();
        let mut last_i = 0;
        let mut last_value = a[0].max.clone();
        (0..n).step_by(2).for_each(|i| {
            last_i = i;
            a[i].max = a[i].interval.end().clone();
            last_value = a[i].max.clone();
        });
        let mut k = 1;
        while (1 << k) <= n {
            // process internal nodes in the bottom-up order
            let x = 1 << (k - 1);
            let i0 = (x << 1) - 1; // i0 is the first node
            let step = x << 2;
            for i in (i0..n).step_by(step) {
                // traverse all nodes at level k
                let end_left = a[i - x].max.clone(); // max value of the left child
                let end_right = if i + x < n { a[i + x].max.clone() } else { last_value.clone() }; // max value of the right child
                let end = max3(a[i].interval.end(), &end_left, &end_right).clone();
                a[i].max = end;
            }
            last_i = if (last_i >> k & 1) > 0 {
                last_i - x
            } else {
                last_i + x
            };
            if last_i < n && a[last_i].max > last_value {
                last_value = a[last_i].max.clone()
            }
            k += 1;
        }
        self.max_level = k - 1;
    }

    pub fn get(&self, index: usize) -> Option<Entry<N, D>> {
        self.entries.get(index)
            .map(|e| Entry { index, interval: &e.interval, data: &e.data })
    }

    pub fn get_mut(&mut self, index: usize) -> Option<EntryMut<N, D>> {
        self.entries.get_mut(index)
            .map(|e| EntryMut { index, interval: &e.interval, data: &mut e.data })
    }

    pub fn overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> bool {
        let mut found = false;
        self.find_aux(interval, &mut found);
        found
    }

    pub fn find<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Option<Entry<N, D>> {
        let mut first = None;
        self.find_aux(interval, &mut first);
        first
    }

    pub fn find_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> Option<Entry<'a, N, D>> {
        let point = point.borrow();
        self.find(point..=point)
    }

    pub fn find_point_mut<'b, 'a: 'b, M: Borrow<N>>(
        &'a mut self,
        point: M,
    ) -> Option<EntryMut<'a, N, D>> {
        let point = point.borrow();
        self.find_mut(point..=point)
    }

    pub fn find_exact<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Option<Entry<N, D>> {
        let mut first = None;
        self.find_exact_aux(interval, &mut first);
        first
    }

    pub fn find_all<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Vec<Entry<N, D>> {
        let mut buf = Vec::with_capacity(512);
        self.find_aux(interval, &mut buf);
        buf
    }

    pub fn find_mut<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) -> Option<EntryMut<N, D>> {
        let mut first = None;
        self.find_mut_aux(interval, &mut first);
        first
    }

    pub fn contains_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> bool {
        self.find_point(point).is_some()
    }

    pub fn find_all_mut<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) -> Vec<EntryMut<N, D>> {
        let mut buf = Vec::with_capacity(512);
        self.find_mut_aux(interval, &mut buf);
        buf
    }

    pub fn remove_exact<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) {
        let mut indices = RemovalIndices::new();
        self.find_exact_aux(interval, &mut indices);
        let positions = indices.into_vec();
        for pos in positions.into_iter().rev() {
            self.entries.remove(pos);
        }
        self.index();
    }

    pub fn remove_overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) {
        let mut indices = RemovalIndices::new();
        self.find_aux(interval, &mut indices);
        let positions = indices.into_vec();
        if positions.len() > 0 {
            for pos in positions.into_iter().rev() {
                self.entries.remove(pos);
            }
            self.index();
        }
    }

    pub fn take_overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) -> Vec<(Interval<N>, D)> {
        let mut indices = RemovalIndices::new();
        self.find_aux(interval, &mut indices);

        let positions = indices.into_vec();
        if positions.len() > 0 {
            let mut removed = Vec::with_capacity(positions.len());

            for pos in positions.into_iter().rev() {
                let entry = self.entries.remove(pos);
                removed.push((entry.interval, entry.data));
            }
            if removed.len() > 0 {
                self.index();
            }
            removed
        } else {
            Vec::new()
        }
    }

    fn find_exact_aux<'b, 'a: 'b, M: Borrow<N>, I: Into<Interval<M>>, C>(
        &'a self,
        interval: I,
        results: &'b mut C,
    ) where C: EntryContainer<'a, N, D> {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());
        let n = self.entries.len() as usize;
        let a = &self.entries;
        let mut stack = [StackCell::default(); 64];
        // push the root; this is a top down traversal
        stack[0].k = self.max_level;
        stack[0].x = (1 << self.max_level) - 1;
        stack[0].w = false;
        let mut t = 1;
        while t > 0 {
            t -= 1;
            let StackCell { k, x, w } = stack[t];
            if k <= 3 {
                // we are in a small subtree; traverse every node in this subtree
                let i0 = x >> k << k;
                let i1 = min(i0 + (1 << (k + 1)) - 1, n);
                for (i, node) in a.iter().enumerate().take(i1).skip(i0) {
                    if node.interval.start() > end {
                        break;
                    }
                    if start == node.interval.start() && end == node.interval.end() {
                        // if overlap, append to `results`
                        if results.push_entry(Entry {
                            index: i,
                            interval: &self.entries[i].interval,
                            data: &self.entries[i].data,
                        }) {
                            return
                        }
                    }
                }
            } else if !w {
                // if left child not processed
                let y = x - (1 << (k - 1)); // the left child of x; NB: y may be out of range (i.e. y>=n)
                stack[t].k = k;
                stack[t].x = x;
                stack[t].w = true; // re-add node x, but mark the left child having been processed
                t += 1;
                if y >= n || a[y].max >= *start {
                    // push the left child if y is out of range or may overlap with the query
                    stack[t].k = k - 1;
                    stack[t].x = y;
                    stack[t].w = false;
                    t += 1;
                }
            } else if x < n && a[x].interval.start() <= end {
                // need to push the right child
                if start == a[x].interval.start() && end == a[x].interval.end() {
                    if results.push_entry(Entry {
                        index: x,
                        interval: &self.entries[x].interval,
                        data: &self.entries[x].data,
                    }) {
                        return
                    }
                }
                stack[t].k = k - 1;
                stack[t].x = x + (1 << (k - 1));
                stack[t].w = false;
                t += 1;
            }
        }
    }

    #[inline(always)]
    fn find_aux<'b, 'a: 'b, M: Borrow<N>, I: Into<Interval<M>>, C>(
        &'a self,
        interval: I,
        results: &'b mut C,
    ) where C: EntryContainer<'a, N, D> {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());
        let n = self.entries.len() as usize;
        let a = &self.entries;
        let mut stack = [StackCell::default(); 64];
        // push the root; this is a top down traversal
        stack[0].k = self.max_level;
        stack[0].x = (1 << self.max_level) - 1;
        stack[0].w = false;
        let mut t = 1;
        while t > 0 {
            t -= 1;
            let StackCell { k, x, w } = stack[t];
            if k <= 3 {
                // we are in a small subtree; traverse every node in this subtree
                let i0 = x >> k << k;
                let i1 = min(i0 + (1 << (k + 1)) - 1, n);
                for (i, node) in a.iter().enumerate().take(i1).skip(i0) {
                    if node.interval.start() > end {
                        break;
                    }
                    if start <= node.interval.end() {
                        // if overlap, append to `results`
                        if results.push_entry(Entry {
                            index: i,
                            interval: &self.entries[i].interval,
                            data: &self.entries[i].data,
                        }) {
                            return
                        }
                    }
                }
            } else if !w {
                // if left child not processed
                let y = x - (1 << (k - 1)); // the left child of x; NB: y may be out of range (i.e. y>=n)
                stack[t].k = k;
                stack[t].x = x;
                stack[t].w = true; // re-add node x, but mark the left child having been processed
                t += 1;
                if y >= n || a[y].max >= *start {
                    // push the left child if y is out of range or may overlap with the query
                    stack[t].k = k - 1;
                    stack[t].x = y;
                    stack[t].w = false;
                    t += 1;
                }
            } else if x < n && a[x].interval.start() <= end {
                // need to push the right child
                if start <= a[x].interval.end() {
                    if results.push_entry(Entry {
                        index: x,
                        interval: &self.entries[x].interval,
                        data: &self.entries[x].data,
                    }) {
                        return
                    }
                }
                stack[t].k = k - 1;
                stack[t].x = x + (1 << (k - 1));
                stack[t].w = false;
                t += 1;
            }
        }
    }

    fn find_mut_aux<'b, 'a: 'b, M: Borrow<N>, I: Into<Interval<M>>, C>(
        &'a mut self,
        interval: I,
        results: &'b mut C,
    ) where C: EntryMutContainer<'a, N, D> {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());
        let n = self.entries.len() as usize;
        let a = &self.entries;

        let mut stack = [StackCell::default(); 64];
        // push the root; this is a top down traversal
        stack[0].k = self.max_level;
        stack[0].x = (1 << self.max_level) - 1;
        stack[0].w = false;

        let mut t = 1;

        while t > 0 {
            t -= 1;
            let StackCell { k, x, w } = stack[t];
            if k <= 3 {
                // we are in a small subtree; traverse every node in this subtree
                let i0 = x >> k << k;
                let i1 = min(i0 + (1 << (k + 1)) - 1, n);
                for (i, node) in a.iter().enumerate().take(i1).skip(i0) {
                    if node.interval.start() > end {
                        break;
                    }
                    if start <= node.interval.end() {
                        // if overlap, append to `results`
                        if unsafe {
                            let entries = self.entries.as_ptr();
                            let entry = entries.add(i);

                            results.push_entry_mut(EntryMut {
                                index: i,
                                interval: &(*entry).interval,
                                data: &mut (*(entry as *mut InternalEntry<N, D>)).data,
                            })
                        } {
                            return;
                        }
                    }
                }
            } else if !w {
                // if left child not processed
                let y = x - (1 << (k - 1)); // the left child of x; NB: y may be out of range (i.e. y>=n)
                stack[t].k = k;
                stack[t].x = x;
                stack[t].w = true; // re-add node x, but mark the left child having been processed
                t += 1;
                if y >= n || a[y].max >= *start {
                    // push the left child if y is out of range or may overlap with the query
                    stack[t].k = k - 1;
                    stack[t].x = y;
                    stack[t].w = false;
                    t += 1;
                }
            } else if x < n && a[x].interval.start() <= end {
                // need to push the right child
                if start <= a[x].interval.end() {
                    if unsafe {
                        let entries = self.entries.as_ptr();
                        let entry = entries.add(x);

                        results.push_entry_mut(EntryMut {
                            index: x,
                            interval: &(*entry).interval,
                            data: &mut (*(entry as *mut InternalEntry<N, D>)).data,
                        })
                    } {
                        return
                    }
                }
                stack[t].k = k - 1;
                stack[t].x = x + (1 << (k - 1));
                stack[t].w = false;
                t += 1;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=(&Interval<N>, &D)> {
        self.entries.iter().map(|e| (&e.interval, &e.data))
    }

    pub fn into_iter(self) -> impl Iterator<Item=(Interval<N>, D)> {
        self.entries.into_iter().map(|e| (e.interval, e.data))
    }

    pub fn intervals(&self) -> impl Iterator<Item=&Interval<N>> {
        self.entries.iter().map(|e| &e.interval)
    }

    pub fn values(&self) -> impl Iterator<Item=&D> {
        self.entries.iter().map(|e| &e.data)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item=&mut D> {
        self.entries.iter_mut().map(|e| &mut e.data)
    }
}

impl<N: std::fmt::Debug + Ord + Clone + NumRef, D> IntervalTree<N, D> {
    pub fn gaps<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Vec<Interval<N>> {
        let interval = interval.into();

        let mut start = interval.start().borrow().clone();
        let end = interval.end().borrow().clone();

        let mut overlaps = self.find_all(&start..=&end)
            .into_iter()
            .map(|e| e.interval.clone())
            .collect::<Vec<_>>();

        // 1) sort by start point
        overlaps.sort_by(|e1, e2| e1.start().cmp(e2.start()));

        // 2) merge overlaps
        let mut si = 0;
        let mut ei = 1;
        let count = overlaps.len();

        while ei < count {
            let (xs, ys) = overlaps.split_at_mut(si + 1);
            let yei = ei - xs.len();
            if N::one().add(xs[si].end()) >= *ys[yei].start() {
                let ns = xs[si].start().clone().min(ys[yei].start().clone());
                let ne = xs[si].end().clone().max(ys[yei].end().clone());
                xs[si] = Interval::from(ns..=ne);
            } else {
                si += 1;
            }
            ei += 1;
        }

        overlaps.truncate(si + 1);

        // 3) use overlaps as mask for interval
        if overlaps.is_empty() {
            vec![Interval::from(start..=end)]
        } else {
            let mut gaps = Vec::new();
            for iv in overlaps.into_iter() {
                let (istart, iend) = iv.into_inner();

                if istart > start {
                    gaps.push(Interval::from(start..=istart.sub(N::one())));
                }

                start = N::one().add(iend);

                if start > end {
                    break
                }
            }

            if start <= end {
                gaps.push(Interval::from(start..=end));
            }

            gaps
        }
    }
}

#[inline(always)]
fn max3<T: Ord>(a: T, b: T, c: T) -> T {
    a.max(b.max(c))
}

#[derive(Clone, Copy, Default)]
struct StackCell {
    // node
    x: usize,
    // level
    k: usize,
    // false if left child hasn't been processed
    w: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_derive", derive(serde::Deserialize, serde::Serialize))]
pub struct DisjointIntervalTree<N: Ord + Clone, D> {
    entries: Vec<InternalEntry<N, D>>,
    max_level: usize,
}

impl<N: Ord + Clone, D> Default for DisjointIntervalTree<N, D> {
    fn default() -> Self {
        DisjointIntervalTree {
            entries: vec![],
            max_level: 0,
        }
    }
}

impl<N, D, V> FromIterator<(V, D)> for DisjointIntervalTree<N, D>
where
    V: Into<Interval<N>>,
    N: Ord + Clone,
{
    fn from_iter<T: IntoIterator<Item = (V, D)>>(iter: T) -> Self {
        let mut tree = Self::new();
        iter.into_iter()
            .for_each(|(interval, data)| tree.insert(interval, data));
        tree.index();
        tree
    }
}

impl<N: Ord + Clone, D> DisjointIntervalTree<N, D> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert<K: Into<Interval<N>>, V: Into<D>>(&mut self, interval: K, data: V) {
        let interval = interval.into();
        let data = data.into();

        let max = interval.end().clone();
        self.entries.push(InternalEntry {
            interval,
            data,
            max,
        });
        self.index();
    }

    pub fn extend<I, K, V>(&mut self, intervals: I)
    where K: Into<Interval<N>>,
          V: Into<D>,
          I: IntoIterator<Item=(K, V)> {
        let it = intervals.into_iter();
        let (lb, ub) = it.size_hint();
        self.entries.reserve(ub.unwrap_or(lb));

        for (interval, data) in it.map(|(k, v)| (k.into(), v.into())) {
            let max = interval.end().clone();
            self.entries.push(InternalEntry {
                interval,
                data,
                max,
            });
        }
        self.index();
    }

    pub(crate) fn index(&mut self) {
        self.entries.sort_by(|l, r| l.interval.start().cmp(&r.interval.start()));
    }

    pub fn get(&self, index: usize) -> Option<Entry<N, D>> {
        self.entries.get(index)
            .map(|e| Entry { index, interval: &e.interval, data: &e.data })
    }

    pub fn get_mut(&mut self, index: usize) -> Option<EntryMut<N, D>> {
        self.entries.get_mut(index)
            .map(|e| EntryMut { index, interval: &e.interval, data: &mut e.data })
    }

    pub fn overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> bool {
        let mut found = false;
        self.find_aux(interval, &mut found);
        found
    }

    pub fn find<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Option<Entry<N, D>> {
        let mut first = None;
        self.find_aux(interval, &mut first);
        first
    }

    pub fn find_all<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Vec<Entry<N, D>> {
        let mut buf = Vec::with_capacity(32);
        self.find_aux(interval, &mut buf);
        buf
    }

    pub fn find_mut<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) -> Option<EntryMut<N, D>> {
        let mut first = None;
        self.find_mut_aux(interval, &mut first);
        first
    }

    pub fn find_all_mut<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) -> Vec<EntryMut<N, D>> {
        let mut buf = Vec::with_capacity(32);
        self.find_mut_aux(interval, &mut buf);
        buf
    }

    pub fn remove_overlaps<M: Borrow<N>, K: Into<Interval<M>>>(&mut self, interval: K) {
        let mut indices = RemovalIndices::new();
        self.find_aux(interval, &mut indices);
        let positions = indices.into_vec();
        for pos in positions.into_iter().rev() {
            self.entries.remove(pos);
        }
        self.index();
    }

    pub fn contains_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> bool {
        let start = point.borrow();
        let a = &self.entries;

        a.binary_search_by(|v| if v.interval.contains_point(&start) {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }).is_ok()
    }

    pub fn find_point<'b, 'a: 'b, M: Borrow<N>>(
        &'a self,
        point: M,
    ) -> Option<Entry<N, D>> {
        let start = point.borrow();
        let a = &self.entries;

        if let Ok(index) = a.binary_search_by(|v| if v.interval.contains_point(&start) {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }) {
            let v = &a[index];
            Some(Entry {
                index,
                interval: &v.interval,
                data: &v.data,
            })
        } else {
            None
        }
    }

    pub fn find_point_mut<M: Borrow<N>>(
        &mut self,
        point: M,
    ) -> Option<EntryMut<N, D>> {
        let start = point.borrow();
        let a = &mut self.entries;

        if let Ok(index) = a.binary_search_by(|v| if v.interval.contains_point(&start) {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }) {
            let v = &mut a[index];
            Some(EntryMut {
                index,
                interval: &v.interval,
                data: &mut v.data,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn find_exact<M: Borrow<N>, I: Into<Interval<M>>>(
        &self,
        interval: I,
    ) -> Option<Entry<N, D>> {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());

        let a = &self.entries;

        if let Ok(pos) = a.binary_search_by(|v| if v.interval.start() == start && v.interval.end() == end {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }) {
            Some(Entry {
                index: pos,
                interval: &a[pos].interval,
                data: &a[pos].data,
            })
        } else {
            None
        }
    }

    pub fn find_exact_mut<M: Borrow<N>, I: Into<Interval<M>>>(
        &mut self,
        interval: I,
    ) -> Option<EntryMut<N, D>> {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());

        let a = &mut self.entries;

        if let Ok(pos) = a.binary_search_by(|v| if v.interval.start() == start && v.interval.end() == end {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }) {
            let v = &mut a[pos];
            Some(EntryMut {
                index: pos,
                interval: &v.interval,
                data: &mut v.data,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn remove_exact<M: Borrow<N>, I: Into<Interval<M>>>(
        &mut self,
        interval: I,
    ) {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());

        let a = &mut self.entries;

        if let Ok(pos) = a.binary_search_by(|v| if v.interval.start() == start && v.interval.end() == end {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }) {
            a.remove(pos);
        }
    }

    #[inline(always)]
    fn find_aux<'b, 'a: 'b, M: Borrow<N>, I: Into<Interval<M>>, C>(
        &'a self,
        interval: I,
        results: &'b mut C,
    ) where C: EntryContainer<'a, N, D> {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());

        let a = &self.entries;

        if let Ok(pos) = a.binary_search_by(|v| if v.interval.contains_point(&start) {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }) {
            if results.push_entry(Entry {
                index: pos,
                interval: &a[pos].interval,
                data: &a[pos].data,
            }) {
                return
            }

            for (i, v) in a[pos+1..].iter().enumerate().take_while(|(_, v)| v.interval.contains_point(end) || end > v.interval.end()) {
                if results.push_entry(Entry {
                    index: i + pos + 1,
                    interval: &v.interval,
                    data: &v.data,
                }) {
                    return
                }
            }
        }
    }

    #[inline(always)]
    fn find_mut_aux<'b, 'a: 'b, M: Borrow<N>, I: Into<Interval<M>>, C>(
        &'a mut self,
        interval: I,
        results: &'b mut C,
    ) where C: EntryMutContainer<'a, N, D> {
        let interval = interval.into();
        let (start, end) = (interval.start().borrow(), interval.end().borrow());

        let a = &mut self.entries;

        if let Ok(pos) = a.binary_search_by(|v| if v.interval.contains_point(&start) {
            Ordering::Equal
        } else if start < v.interval.start() {
            Ordering::Greater
        } else {
            Ordering::Less
        }) {
            let (a, b) = a.split_at_mut(pos + 1);

            if results.push_entry_mut(EntryMut {
                index: pos,
                interval: &a[pos].interval,
                data: &mut a[pos].data,
            }) {
                return
            }

            for (i, v) in b.iter_mut().enumerate().take_while(|(_, v)| v.interval.contains_point(end) || end > v.interval.end()) {
                if results.push_entry_mut(EntryMut {
                    index: i + pos + 1,
                    interval: &v.interval,
                    data: &mut v.data,
                }) {
                    return
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=(&Interval<N>, &D)> {
        self.entries.iter().map(|e| (&e.interval, &e.data))
    }

    pub fn into_iter(self) -> impl Iterator<Item=(Interval<N>, D)> {
        self.entries.into_iter().map(|e| (e.interval, e.data))
    }

    pub fn values(&self) -> impl Iterator<Item=&D> {
        self.entries.iter().map(|e| &e.data)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item=&mut D> {
        self.entries.iter_mut().map(|e| &mut e.data)
    }
}

impl<N: std::fmt::Debug + Ord + Clone + NumRef, D> DisjointIntervalTree<N, D> {
    pub fn gaps<M: Borrow<N>, K: Into<Interval<M>>>(&self, interval: K) -> Vec<Interval<N>> {
        let interval = interval.into();

        let mut start = interval.start().borrow().clone();
        let end = interval.end().borrow().clone();

        let mut overlaps = self.find_all(&start..=&end)
            .into_iter()
            .map(|e| e.interval.clone())
            .collect::<Vec<_>>();

        // 1) sort by start point
        overlaps.sort_by(|e1, e2| e1.start().cmp(e2.start()));

        // 2) merge overlaps
        let mut si = 0;
        let mut ei = 1;
        let count = overlaps.len();

        while ei < count {
            let (xs, ys) = overlaps.split_at_mut(si + 1);
            let yei = ei - xs.len();
            if N::one().add(xs[si].end()) >= *ys[yei].start() {
                let ns = xs[si].start().clone().min(ys[yei].start().clone());
                let ne = xs[si].end().clone().max(ys[yei].end().clone());
                xs[si] = Interval::from(ns..=ne);
            } else {
                si += 1;
            }
            ei += 1;
        }

        overlaps.truncate(si + 1);

        // 3) use overlaps as mask for interval
        if overlaps.is_empty() {
            vec![Interval::from(start..=end)]
        } else {
            let mut gaps = Vec::new();
            for iv in overlaps.into_iter() {
                let (istart, iend) = iv.into_inner();

                if istart > start {
                    gaps.push(Interval::from(start..=istart.sub(N::one())));
                }

                start = N::one().add(iend);

                if start > end {
                    break
                }
            }

            if start <= end {
                gaps.push(Interval::from(start..=end));
            }

            gaps
        }
    }
}
