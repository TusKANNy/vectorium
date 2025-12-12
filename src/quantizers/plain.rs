use serde::{Deserialize, Serialize};

use std::marker::PhantomData;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlainDenseQuantizer<T> {
    d: usize,
    // distance: DistanceType,
    _phantom: PhantomData<T>,
}
