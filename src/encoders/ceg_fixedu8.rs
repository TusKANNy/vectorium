use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::{FixedU8Q, SpaceUsage};
use bytemuck::cast_slice;
use dsi_bitstream::impls::BufBitReader;
use dsi_bitstream::prelude::*;
const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CegFixedU8Encoder {
    dim: usize,
    all_refs: Vec<u16>,
    max_ref_size: usize,
    num_clusters: usize,
    component_mapping: Option<Box<[u16]>>,
}

impl PartialEq for CegFixedU8Encoder {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.all_refs == other.all_refs
            && self.max_ref_size == other.max_ref_size
            && self.component_mapping == other.component_mapping
            && self.num_clusters == other.num_clusters
    }
}

impl sealed::Sealed for CegFixedU8Encoder {}

impl CegFixedU8Encoder {

    pub fn new_with_references(input_dim: usize, reference_lists: Vec<Vec<u16>>, max_ref_size: usize) -> Self {
        let num_clusters = reference_lists.len();
        let mut all_refs = Vec::with_capacity(reference_lists.len() * max_ref_size);
        for mut list in reference_lists {
            list.sort_unstable();
            list.truncate(max_ref_size);
            while list.len() < max_ref_size {
                list.push(u16::MAX);
            }
            all_refs.extend_from_slice(&list);
        }
        Self {
            dim: input_dim,
            all_refs,
            max_ref_size,
            num_clusters,
            component_mapping: None,
        }
    }

    pub fn train<'a, V>(
        &mut self,
        training_data: impl Iterator<Item = SparseVectorView<'a, u16, V>>,
    ) where
        V: crate::ValueType,
    {
        let components_iter = training_data.map(|v| v.components());
        let permutation =
            crate::utils::permute_components_with_bisection(self.input_dim(), components_iter);
        let component_mapping: Vec<u16> = permutation.iter().map(|i| *i as u16).collect();
        self.component_mapping = Some(component_mapping.into_boxed_slice());
        self.remap_all_refs();
    }

    fn remap_all_refs(&mut self) {
        if let Some(mapping) = &self.component_mapping {
            for ref_id in self.all_refs.iter_mut() {
                if *ref_id != u16::MAX {
                    *ref_id = mapping[*ref_id as usize];
                }
            }
            (0..self.num_clusters).for_each(|i| {
                let s = i * self.max_ref_size;
                let range = s..s + self.max_ref_size;
                self.all_refs[range].sort_unstable();
            });
        }
    }

    fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u16; dim];
        for (old, &new) in component_mapping.iter().enumerate() {
            inverse[new as usize] = old as u16;
        }
        inverse
    }

    #[inline]
    fn get_ref_slice(&self, start: u32) -> &[u16] {
        unsafe {
            self.all_refs
                .get_unchecked(start as usize..(start as usize + self.max_ref_size))
        }
    }

    fn find_best_k_rice(gaps: &[u32]) -> u8 {
        (0..16)
            .map(|k| {
                let total_len: usize = gaps.iter().map(|&g| len_exp_golomb(g as u64, k)).sum();
                (k, total_len as f64 / gaps.len() as f64)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(k, _)| k as u8)
            .unwrap_or(0) // Default a 0 se l'array è vuoto
    }

    pub fn push_vector<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, FixedU8Q>,
        cluster_id: u16,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<u8> = input.values().iter().map(|v| v.to_bits()).collect();
        let mut q_components: Vec<u16> = if let Some(mapping) = &self.component_mapping {
            input
                .components()
                .iter()
                .map(|&c| mapping[c as usize])
                .collect()
        } else {
            input.components().to_vec()
        };

        if self.component_mapping.is_some() {
            let permutation = rusty_perm::PermD::from_sort(q_components.as_slice());
            permutation.apply(q_values.as_mut_slice()).unwrap();
            permutation.apply(q_components.as_mut_slice()).unwrap();
        }

        let ref_start = cluster_id as u32 * self.max_ref_size as u32;
        let ref_indices = self.get_ref_slice(ref_start);
        let mut mapped_positions = Vec::new();
        let mut mapped_values = Vec::new();
        let mut residual_indices = Vec::new();
        let mut residual_values = Vec::new();

        for (&comp, &val) in q_components.iter().zip(q_values.iter()) {
            match ref_indices.binary_search(&comp) {
                Ok(pos) => {
                    mapped_positions.push(pos as u32);
                    mapped_values.push(val);
                }
                Err(_) => {
                    residual_indices.push(comp as u32);
                    residual_values.push(val);
                }
            }
        }

        let map_n = mapped_values.len();
        let res_n = residual_values.len();
        let gap_mapped = mapped_positions
            .iter()
            .scan(0u32, |state, &pos| {
                let gap = pos - *state;
                *state = pos;
                Some(gap)
            })
            .collect::<Vec<u32>>();
        let gap_residual = residual_indices
            .iter()
            .scan(0u32, |state, &comp| {
                let gap = comp - *state;
                *state = comp;
                Some(gap)
            })
            .collect::<Vec<u32>>();
        let best_k_mapped = Self::find_best_k_rice(&gap_mapped) as usize;

        let best_k_residual = Self::find_best_k_rice(&gap_residual) as usize;
        let mut bit_writer = <BufBitWriter<LE, _>>::new(MemWordWriterVec::<u8, _>::new(Vec::new()));
        // 1. Header
        bit_writer.write_bits(cluster_id as u64, 16).unwrap();
        bit_writer.write_vbyte_le(map_n as u64).unwrap();
        bit_writer.write_vbyte_le(res_n as u64).unwrap();
        bit_writer.write_bits(best_k_mapped as u64, 4).unwrap();
        bit_writer.write_bits(best_k_residual as u64, 4).unwrap();

        // 2. Mapped & Residual Gaps (Exp-Golomb)
        for &gap in &gap_mapped {
            bit_writer
                .write_exp_golomb(gap as u64, best_k_mapped)
                .unwrap();
        }

        for &gap in &gap_residual {
            bit_writer
                .write_exp_golomb(gap as u64, best_k_residual)
                .unwrap();
        }

        let mut payload = Vec::new();
        payload.extend_from_slice(bit_writer.into_inner().unwrap().into_inner().as_slice());

        // 3. Padding to ensure values are aligned at the end of the payload on an 8-byte boundary.
        let values_len = map_n + res_n;
        let current_len = payload.len();
        let padding_needed = (8 - (current_len + values_len) % 8) % 8;
        for _ in 0..padding_needed {
            payload.push(0);
        }

        // 4. All Values M & R (u8) aligned at the end
        payload.extend_from_slice(&mapped_values);
        payload.extend_from_slice(&residual_values);

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        // Layout of the encoded vector in u64 words:
        // [ cluster_id (16 bits) | map_n (vbyte) | res_n (vbyte) | k_map (4 bits) | k_res (4 bits) ]
        // [ Mapped Gaps (Exp-Golomb) ]
        // [ Residual Gaps (Exp-Golomb) ]
        // [ Padding (to align the end of values to an 8-byte boundary) ]
        // [ Mapped Values (u8) ]
        // [ Residual Values (u8) ]
        output.extend(data);
    }
}

impl SparseDataEncoder for CegFixedU8Encoder {
    type InputComponentType = u16;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u16;
    type OutputValueType = FixedU8Q;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let bytes = cast_slice::<u64, u8>(encoded.data());
        let slices = cast_slice::<u64, u32>(encoded.data());

        let mut reader = BufBitReader::<LE, _>::new(MemWordReader::new_inf(slices));
        let cluster_id = reader.read_bits(16).unwrap() as u16;
        let map_n = reader.read_vbyte_le().unwrap() as usize;
        let res_n = reader.read_vbyte_le().unwrap() as usize;
        let k_map = reader.read_bits(4).unwrap() as usize;
        let k_res = reader.read_bits(4).unwrap() as usize;

        let total_n = map_n + res_n;
        let values_m = &bytes[bytes.len() - total_n..bytes.len() - res_n];
        let values_r = &bytes[bytes.len() - res_n..];

        let ref_start = cluster_id as u32 * self.max_ref_size as u32;
        let ref_indices = self.get_ref_slice(ref_start);

        let mut mapped_comps = Vec::with_capacity(map_n);
        let mut mapped_vals = Vec::with_capacity(map_n);
        let mut residual_comps = Vec::with_capacity(res_n);
        let mut residual_vals = Vec::with_capacity(res_n);

        let mut last_map = 0u32;
        for &value in values_m {
            let gap = reader.read_exp_golomb(k_map).expect("Failed to read EG mapped") as u32;
            last_map += gap;
            let comp = unsafe { *ref_indices.get_unchecked(last_map as usize) };
            mapped_comps.push(comp);
            mapped_vals.push(value as f32 * FIXED_U8_SCALE);
        }

        let mut last_res = 0u32;
        for &value in values_r {
            let gap = reader.read_exp_golomb(k_res).expect("Failed to read EG residual") as u32;
            last_res += gap;
            residual_comps.push(last_res as u16);
            residual_vals.push(value as f32 * FIXED_U8_SCALE);
        }

        if let Some(component_mapping) = &self.component_mapping {
            let mut decoded_comp = mapped_comps;
            let mut values = mapped_vals;
            decoded_comp.extend_from_slice(&residual_comps);
            values.extend_from_slice(&residual_vals);

            let inverse = Self::compute_inverse_mapping(component_mapping);
            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }
            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            SparseVectorOwned::new(decoded_comp, values)
        } else {
            let n_total = mapped_comps.len() + residual_comps.len();
            let mut decoded_comp = Vec::with_capacity(n_total);
            let mut values = Vec::with_capacity(n_total);

            let (mut i, mut j) = (0, 0);
            while i < mapped_comps.len() && j < residual_comps.len() {
                if mapped_comps[i] < residual_comps[j] {
                    decoded_comp.push(mapped_comps[i]);
                    values.push(mapped_vals[i]);
                    i += 1;
                } else {
                    decoded_comp.push(residual_comps[j]);
                    values.push(residual_vals[j]);
                    j += 1;
                }
            }
            decoded_comp.extend_from_slice(&mapped_comps[i..]);
            values.extend_from_slice(&mapped_vals[i..]);
            decoded_comp.extend_from_slice(&residual_comps[j..]);
            values.extend_from_slice(&residual_vals[j..]);

            SparseVectorOwned::new(decoded_comp, values)
        }
    }
}

impl PackedSparseVectorEncoder for CegFixedU8Encoder {
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, FixedU8Q>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        self.push_vector(input, 0, output);
    }

}

impl VectorEncoder for CegFixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = CegFixedU8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        CegFixedU8QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        CegFixedU8QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct CegFixedU8QueryEvaluator<'a> {
    encoder: &'a CegFixedU8Encoder,
    dense_query: Vec<f32>,
}

impl<'a, 'v> CegFixedU8QueryEvaluator<'a> {
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &'a CegFixedU8Encoder) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = v;
        }
        Self {
            encoder,
            dense_query,
        }
    }

    pub fn plain_compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let bytes = cast_slice::<u64, u8>(vector.data());
        let slices = cast_slice::<u64, u32>(vector.data());
        
        let mut reader = BufBitReader::<LE, _>::new(MemWordReader::new_inf(slices));
        let cluster_id = reader.read_bits(16).unwrap() as u16;
        let map_n = reader.read_vbyte_le().unwrap() as u16;
        let res_n = reader.read_vbyte_le().unwrap() as u16;
        let k_map = reader.read_bits(4).unwrap() as usize;
        let k_res = reader.read_bits(4).unwrap() as usize;
        
        let total_n = map_n as usize + res_n as usize;
        let all_values_m = &bytes[bytes.len() - total_n..bytes.len() - res_n as usize];
        let all_values_r = &bytes[bytes.len() - res_n as usize..];
        
        let ref_start = cluster_id as u32 * self.encoder.max_ref_size as u32;
        let ref_indices = self.encoder.get_ref_slice(ref_start);
        let query = self.dense_query.as_slice();
    
        let mapped_sum = all_values_m.iter().fold((0.0f32, 0u32), |(acc, last_map), &value| {
            let gap = reader.read_exp_golomb(k_map).expect("Failed to read EG mapped") as u32;
            let last_map = last_map + gap;
            let comp = unsafe { *ref_indices.get_unchecked(last_map as usize) };
            (acc + unsafe { *query.get_unchecked(comp as usize) } * value as f32, last_map)
        }).0;
        let total_sum = all_values_r.iter().fold((mapped_sum, 0u32), |(acc, last_res), &value| {
             let gap = reader.read_exp_golomb(k_res).expect("Failed to read EG residual") as u32;
             let last_res = last_res + gap;
             let comp = unsafe { *query.get_unchecked(last_res as usize) };
             (acc + comp * value as f32, last_res)
        }).0;

        DotProduct(total_sum * FIXED_U8_SCALE)
    }

}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for CegFixedU8QueryEvaluator<'a> {
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        self.plain_compute_distance(vector)
    }
}


impl SpaceUsage for CegFixedU8Encoder {
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        self.all_refs.len() * 2 + size_of_mapping + self.dim.space_usage_bytes() + self.num_clusters.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FromF32;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use num_traits::ToPrimitive;

    fn fixed(val: f32) -> FixedU8Q {
        FixedU8Q::from_f32_saturating(val)
    }

    #[test]
    fn cef_compute_distance_with_only_mapped_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CegFixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(1.5),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
        ];
        let input = SparseVectorView::new(&[0, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        println!("Distance: {:?}", dist);
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5), (36, 1.5), (48, 2.0), (53, 1.0), (90, 2.0)
        // query = (2, 0.5), (4, 1.5), (6, 2.5), (8, 1.0), (24, 2.0), (70, 1.0), (90, 2.0)
        let expected = 1.5 * 3.0 + 2.0 * 1.0 + 2.0 * 3.5 + 2.0 * 2.0;
        println!("Expected: {}", expected);
        assert!((dist.distance() - expected).abs() < 1e-5);

        let dist_plain = evaluator.plain_compute_distance(PackedVectorView::new(&buffer));
        println!("Plain Distance: {:?}", dist_plain);
        assert!((dist_plain.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn cef_compute_distance_with_mapped_both() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CegFixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(1.5),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
            fixed(3.0),
            fixed(2.5),
        ];
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5), (28, 1.5), (36, 2.0), (48, 1.0), (53, 2.0), (70, 3.0), (90, 2.5)
        let input = SparseVectorView::new(&[0, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        // query = (2, 0.5), (4, 1.5), (8, 2.5), (24, 1.0), (28, 2.0), (70, 1.0), (90, 2.0)
        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        println!("Distance: {:?}", dist);
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5), (28, 1.5), (36, 2.0), (48, 1.0), (53, 2.0), (70, 3.0), (90, 2.5)
        // query = (2, 0.5), (4, 1.5), (8, 2.5), (24, 1.0), (28, 2.0), (70, 1.0), (90, 2.0)
        let expected = 3.0 * 1.5 + 2.0 * 2.5 + 3.5 * 1.0 + 1.5 * 2.0 + 3.0 * 1.0 + 2.5 * 2.0;
        println!("Expected: {}", expected);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn cef_compute_distance_with_only_mapped_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CegFixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [fixed(1.0), fixed(3.0), fixed(2.0), fixed(3.5)];
        let input = SparseVectorView::new(&[0, 4, 8, 24], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[2, 4, 6, 8], &[0.5, 1.5, 2.5, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        println!("Distance: {:?}", dist);
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5)
        // query = (2, 0.5), (4, 1.5), (6, 2.5), (8, 1.0), (70, 1.0), (90, 2.0)
        let expected = 1.5 * 3.0 + 2.0 * 1.0;
        println!("Expected: {}", expected);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn cef_compute_distance_no_mapped_only_res_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CegFixedU8Encoder::new_with_references(25, reference_lists, 512);
        let binding = [fixed(1.0), fixed(3.0)];
        let input = SparseVectorView::new(&[5, 12], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12], &[0.5f32, 1.5, 2.5]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        println!("Distance: {:?}", dist);

        let expected = 1.0 * 0.5 + 3.0 * 2.5;
        println!(
            "Difference from expected: {}",
            (dist.distance() - expected).abs()
        );
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn cef_compute_distance_no_mapped_only_res_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CegFixedU8Encoder::new_with_references(70, reference_lists, 512);
        let binding = [
            fixed(1.0),
            fixed(2.5),
            fixed(2.0),
            fixed(3.5),
            fixed(1.0),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
        ];
        let input = SparseVectorView::new(&[5, 12, 15, 18, 21, 31, 35, 60], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12, 21, 60], &[0.5f32, 1.5, 2.5, 1.0, 2.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        println!("Distance: {:?}", dist);

        let expected = 1.0 * 0.5 + 2.5 * 2.5 + 1.0 * 1.0 + 2.0 * 2.0;
        println!("Expected: {}", expected);
        println!(
            "Difference from expected: {}",
            (dist.distance() - expected).abs()
        );
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    fn verify_cef_eg(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut refs = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            assert!(
                curr <= u16::MAX as u32,
                "Component index too large: {}",
                curr
            );
            refs.push(curr as u16);
        }

        let reference_lists = vec![refs.clone()];
        let encoder = CegFixedU8Encoder::new_with_references(
            u16::MAX as usize + 1,
            reference_lists,
            512
        );

        let values: Vec<_> = (0..num_vals)
            .map(|i| fixed(1.0 + (i % 7) as f32 / 10.0))
            .collect();
        let input = SparseVectorView::new(&refs, &values);

        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
        let query = SparseVectorView::new(&refs, &query_vals);

        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let mut expected = 0.0f32;
        for i in 0..num_vals {
            expected += values[i].to_f32().unwrap() * query_vals[i];
        }

        assert!(
            (dist.distance() - expected).abs() < 1e-3,
            "Failed: dist={}, expected={}, diff={}, gaps_len={}",
            dist.distance(),
            expected,
            (dist.distance() - expected).abs(),
            num_vals
        );
    }

    #[test]
    fn test_eg_dist_0() {
        let mut gaps = Vec::new();
        for _ in 0..2 {
            for _ in 0..28 {
                gaps.push(1);
            }
        } // 56 elements (multiple of 8 and 28)
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_1() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            // 21 * 8 = 168 (multiple of 8 and 21)
            for _ in 0..7 {
                gaps.push(3);
            }
            for _ in 0..14 {
                gaps.push(1);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_2() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            for _ in 0..7 {
                gaps.push(1);
            }
            for _ in 0..7 {
                gaps.push(3);
            }
            for _ in 0..7 {
                gaps.push(1);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_3() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            for _ in 0..14 {
                gaps.push(1);
            }
            for _ in 0..7 {
                gaps.push(3);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_4() {
        let mut gaps = Vec::new();
        for _ in 0..4 {
            for _ in 0..14 {
                gaps.push(3);
            }
        } // 56 elements
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_5() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            // 9 * 8 = 72
            gaps.push(15);
            for _ in 0..8 {
                gaps.push(7);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_6() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            // 8 * 8 = 64
            gaps.push(7);
            for _ in 0..4 {
                gaps.push(15);
            }
            for _ in 0..3 {
                gaps.push(7);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_7() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            for _ in 0..7 {
                gaps.push(15);
            }
        } // 56 elements
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_8() {
        let mut gaps = Vec::new();
        for _ in 0..4 {
            // 6 * 4 = 24
            for _ in 0..4 {
                gaps.push(31);
            }
            for _ in 0..2 {
                gaps.push(15);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_9() {
        let mut gaps = Vec::new();
        for _ in 0..4 {
            for _ in 0..2 {
                gaps.push(15);
            }
            for _ in 0..4 {
                gaps.push(31);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_10() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            // 5 * 8 = 40
            for _ in 0..3 {
                gaps.push(63);
            }
            for _ in 0..2 {
                gaps.push(31);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_11() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            for _ in 0..2 {
                gaps.push(31);
            }
            for _ in 0..3 {
                gaps.push(63);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_12() {
        let mut gaps = Vec::new();
        for _ in 0..2 {
            for _ in 0..4 {
                gaps.push(127);
            }
        } // 8 elements
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_13() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            // 3 * 8 = 24
            gaps.push(1023);
            for _ in 0..2 {
                gaps.push(511);
            }
        }
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_14() {
        let mut gaps = Vec::new();
        for _ in 0..4 {
            for _ in 0..2 {
                gaps.push(8000);
            }
        } // 8 elements
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_eg_dist_15() {
        let mut gaps = Vec::new();
        for _ in 0..8 {
            gaps.push(1023);
        } // 1023 * 8 < 65535, so it fits in u16 refs
        verify_cef_eg(&gaps);
    }

    #[test]
    fn test_cdotdp_decode_roundtrip() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CegFixedU8Encoder::new_with_references(100, reference_lists, 512);

        let components = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(1.5),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
            fixed(3.0),
            fixed(2.5),
        ];
        let input = SparseVectorView::new(&components, &binding);

        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &components);
        for (v1, v2) in decoded.values().iter().zip(binding.iter()) {
            assert!((v1 - v2.to_f32().unwrap()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cdotdp_decode_train_roundtrip() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CegFixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(2.5),
            fixed(1.5),
        ];
        let input_components = [0, 4, 8, 24, 60, 87];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [fixed(3.0), fixed(2.0), fixed(3.5), fixed(2.5), fixed(1.5)];
        let input_components1 = [4, 8, 24, 50, 55];
        let input1 = SparseVectorView::new(&input_components1, &binding1);

        encoder.train([input0.clone(), input1.clone()].into_iter());

        let mut buffer0 = Vec::new();
        encoder.push_encoded(input0, &mut buffer0);

        let mut buffer1 = Vec::new();
        encoder.push_encoded(input1, &mut buffer1);

        let decoded0 = encoder.decode_vector(PackedVectorView::new(&buffer0));
        let decoded1 = encoder.decode_vector(PackedVectorView::new(&buffer1));

        assert_eq!(decoded0.components(), &input_components);
        let decoded_vals0 = decoded0.values();
        assert_eq!(decoded_vals0.len(), 6);
        assert!((decoded_vals0[0] - 1.0).abs() < 1e-6);
        assert!((decoded_vals0[1] - 3.0).abs() < 1e-6);
        assert!((decoded_vals0[2] - 2.0).abs() < 1e-6);
        assert!((decoded_vals0[3] - 3.5).abs() < 1e-6);
        assert!((decoded_vals0[4] - 2.5).abs() < 1e-6);
        assert!((decoded_vals0[5] - 1.5).abs() < 1e-6);

        assert_eq!(decoded1.components(), &input_components1);
        let decoded_vals1 = decoded1.values();
        assert_eq!(decoded_vals1.len(), 5);
        assert!((decoded_vals1[0] - 3.0).abs() < 1e-6);
        assert!((decoded_vals1[1] - 2.0).abs() < 1e-6);
        assert!((decoded_vals1[2] - 3.5).abs() < 1e-6);
        assert!((decoded_vals1[3] - 2.5).abs() < 1e-6);
        assert!((decoded_vals1[4] - 1.5).abs() < 1e-6);
    }
}
