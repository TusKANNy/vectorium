use crate::SpaceUsage;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::eg::common::find_best_k_rice;
use crate::encoders::eg::quantizer::EgQuantizer;
use bytemuck::cast_slice;
use dsi_bitstream::impls::BufBitReader;
use dsi_bitstream::prelude::*;
use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CegEncoder<Q: EgQuantizer> {
    pub dim: usize,
    pub quantizer: Q,
    pub all_refs: Vec<u16>,
    pub max_ref_size: usize,
    pub num_clusters: usize,
    pub component_mapping: Option<Box<[u16]>>,
}

impl<Q: EgQuantizer> PartialEq for CegEncoder<Q>
where
    Q: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.quantizer == other.quantizer
            && self.all_refs == other.all_refs
            && self.max_ref_size == other.max_ref_size
            && self.component_mapping == other.component_mapping
            && self.num_clusters == other.num_clusters
    }
}

impl<Q> CegEncoder<Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    pub fn new(
        input_dim: usize,
        quantizer: Q,
        reference_lists: Vec<Vec<u16>>,
        max_ref_size: usize,
    ) -> Self {
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
            quantizer,
            all_refs,
            max_ref_size,
            num_clusters,
            component_mapping: None,
        }
    }

    pub fn train_components<'a, V>(
        &mut self,
        training_data: impl Iterator<Item = SparseVectorView<'a, u16, V>>,
    ) where
        V: crate::ValueType,
    {
        let components_iter = training_data.map(|v| v.components());
        let permutation =
            crate::utils::permute_components_with_bisection(self.dim, components_iter);
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

    pub fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
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

    pub fn push_vector<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, Q::InputValue>,
        cluster_id: u16,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values().iter())
            .map(|(&c, &v)| self.quantizer.encode_value(c, v))
            .collect();

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
        let best_k_mapped = find_best_k_rice(&gap_mapped) as usize;
        let best_k_residual = find_best_k_rice(&gap_residual) as usize;

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

        output.extend(data);
    }
}

impl<Q> SparseDataEncoder for CegEncoder<Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type InputComponentType = u16;
    type InputValueType = Q::InputValue;
    type OutputComponentType = u16;
    type OutputValueType = Q::InputValue;

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
        let mut mapped_vals_raw: Vec<u8> = Vec::with_capacity(map_n);
        let mut residual_comps = Vec::with_capacity(res_n);
        let mut residual_vals_raw: Vec<u8> = Vec::with_capacity(res_n);

        let mut last_map = 0u32;
        for &value in values_m {
            let gap = reader
                .read_exp_golomb(k_map)
                .expect("Failed to read EG mapped") as u32;
            last_map += gap;
            let comp = unsafe { *ref_indices.get_unchecked(last_map as usize) };
            mapped_comps.push(comp);
            mapped_vals_raw.push(value);
        }

        let mut last_res = 0u32;
        for &value in values_r {
            let gap = reader
                .read_exp_golomb(k_res)
                .expect("Failed to read EG residual") as u32;
            last_res += gap;
            residual_comps.push(last_res as u16);
            residual_vals_raw.push(value);
        }

        // Combine raw values and decode after applying inverse mapping/permutation.
        // For per-component quantizers (e.g., ScalarU8Quantizer), we must decode AFTER
        // inverse mapping and permutation to ensure each raw byte is decoded with its
        // original component's scale. Permuting raw bytes along with their components
        // ensures proper alignment.
        if let Some(component_mapping) = &self.component_mapping {
            let mut decoded_comp = mapped_comps;
            let mut values_raw = mapped_vals_raw;
            decoded_comp.extend_from_slice(&residual_comps);
            values_raw.extend_from_slice(&residual_vals_raw);

            let inverse = Self::compute_inverse_mapping(component_mapping);
            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }
            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values_raw.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();

            SparseVectorOwned::new(decoded_comp, values)
        } else {
            // Merge mapped and residual lists into sorted order, preserving corresponding raw values,
            // then decode values using original component indices (no remapping needed).
            let n_total = mapped_comps.len() + residual_comps.len();
            let mut decoded_comp = Vec::with_capacity(n_total);
            let mut values_raw = Vec::with_capacity(n_total);

            let (mut i, mut j) = (0usize, 0usize);
            while i < mapped_comps.len() && j < residual_comps.len() {
                if mapped_comps[i] < residual_comps[j] {
                    decoded_comp.push(mapped_comps[i]);
                    values_raw.push(mapped_vals_raw[i]);
                    i += 1;
                } else {
                    decoded_comp.push(residual_comps[j]);
                    values_raw.push(residual_vals_raw[j]);
                    j += 1;
                }
            }
            while i < mapped_comps.len() {
                decoded_comp.push(mapped_comps[i]);
                values_raw.push(mapped_vals_raw[i]);
                i += 1;
            }
            while j < residual_comps.len() {
                decoded_comp.push(residual_comps[j]);
                values_raw.push(residual_vals_raw[j]);
                j += 1;
            }

            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();

            SparseVectorOwned::new(decoded_comp, values)
        }
    }
}

impl<Q> PackedSparseVectorEncoder for CegEncoder<Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, Q::InputValue>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        self.push_vector(input, 0, output);
    }
}

impl<Q: EgQuantizer> VectorEncoder for CegEncoder<Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, Q::InputValue>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = CegQueryEvaluator<'e, Q>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        CegQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        CegQueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct CegQueryEvaluator<'a, Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    pub encoder: &'a CegEncoder<Q>,
    pub dense_query: Vec<f32>,
}

impl<'a, Q> CegQueryEvaluator<'a, Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &'a CegEncoder<Q>) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = encoder.quantizer.query_value(c, v);
        }
        Self {
            encoder,
            dense_query,
        }
    }

    pub fn plain_compute_distance(&self, vector: PackedVectorView<'_, u64>) -> DotProduct {
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

        let mapped_sum = all_values_m
            .iter()
            .fold((0.0f32, 0u32), |(acc, last_map), &value| {
                let gap = reader
                    .read_exp_golomb(k_map)
                    .expect("Failed to read EG mapped") as u32;
                let last_map = last_map + gap;
                let comp = unsafe { *ref_indices.get_unchecked(last_map as usize) };
                (
                    acc + unsafe { *query.get_unchecked(comp as usize) } * value as f32,
                    last_map,
                )
            })
            .0;
        let total_sum = all_values_r
            .iter()
            .fold((mapped_sum, 0u32), |(acc, last_res), &value| {
                let gap = reader
                    .read_exp_golomb(k_res)
                    .expect("Failed to read EG residual") as u32;
                let last_res = last_res + gap;
                let comp = unsafe { *query.get_unchecked(last_res as usize) };
                (acc + comp * value as f32, last_res)
            })
            .0;

        DotProduct(total_sum * self.encoder.quantizer.scale())
    }
}

impl<'a, 'v, Q> QueryEvaluator<PackedVectorView<'v, u64>> for CegQueryEvaluator<'a, Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        self.plain_compute_distance(vector)
    }
}

impl<Q: EgQuantizer> SpaceUsage for CegEncoder<Q> {
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        self.all_refs.len() * 2
            + size_of_mapping
            + self.dim.space_usage_bytes()
            + self.num_clusters.space_usage_bytes()
            + self.quantizer.space_usage_bytes()
    }
}
