use crate::interval::Interval;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_derive", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct InternalEntry<N: Ord + Clone, D> {
    pub(crate) data: D,
    pub(crate) interval: Interval<N>,
    pub(crate) max: N,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Entry<'a, N: Ord + Clone, D> {
    pub(crate) index: usize,
    pub(crate) data: &'a D,
    pub(crate) interval: &'a Interval<N>,
}

impl<'a, N: Ord + Clone + 'a, D: 'a> Entry<'a, N, D> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn value(&self) -> &'a D {
        self.data
    }

    pub fn interval(&self) -> &'a Interval<N> {
        self.interval
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct EntryMut<'a, N: Ord + Clone, D> {
    pub(crate) index: usize,
    pub(crate) data: &'a mut D,
    pub(crate) interval: &'a Interval<N>,
}

impl<'a, N: Ord + Clone + 'a, D: 'a> EntryMut<'a, N, D> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn value(&'a mut self) -> &'a mut D {
        self.data
    }

    pub fn into_value(self) -> &'a mut D {
        self.data
    }

    pub fn interval(&self) -> &'a Interval<N> {
        self.interval
    }
}

pub(crate) struct RemovalIndices(pub(crate) Vec<usize>);

impl RemovalIndices {
    pub fn new() -> Self {
        Self(Vec::with_capacity(32))
    }

    pub fn into_vec(mut self) -> Vec<usize> {
        self.0.sort();
        self.0
    }
}

pub(crate) trait EntryContainer<'a, N: Ord + Clone, D> {
    fn push_entry(&mut self, entry: Entry<'a, N, D>) -> bool;
}

impl<'a, N: Ord + Clone, D> EntryContainer<'a, N, D> for bool {
    fn push_entry(&mut self, _entry: Entry<'a, N, D>) -> bool {
        *self = true;
        true
    }
}

impl<'a, N: Ord + Clone, D> EntryContainer<'a, N, D> for RemovalIndices {
    fn push_entry(&mut self, entry: Entry<'a, N, D>) -> bool {
        self.0.push(entry.index);
        false
    }
}

impl<'a, N: Ord + Clone, D> EntryContainer<'a, N, D> for Option<Entry<'a, N, D>> {
    fn push_entry(&mut self, entry: Entry<'a, N, D>) -> bool {
        *self = Some(entry);
        true
    }
}

impl<'a, N: Ord + Clone, D> EntryContainer<'a, N, D> for Vec<Entry<'a, N, D>> {
    fn push_entry(&mut self, entry: Entry<'a, N, D>) -> bool {
        self.push(entry);
        false
    }
}

pub(crate) trait EntryMutContainer<'a, N: Ord + Clone, D> {
    fn push_entry_mut(&mut self, entry: EntryMut<'a, N, D>) -> bool;
}

impl<'a, N: Ord + Clone, D> EntryMutContainer<'a, N, D> for Option<EntryMut<'a, N, D>> {
    fn push_entry_mut(&mut self, entry: EntryMut<'a, N, D>) -> bool {
        *self = Some(entry);
        true
    }
}

impl<'a, N: Ord + Clone, D> EntryMutContainer<'a, N, D> for Vec<EntryMut<'a, N, D>> {
    fn push_entry_mut(&mut self, entry: EntryMut<'a, N, D>) -> bool {
        self.push(entry);
        false
    }
}
