use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::distances::{
    Distance, DotProduct, EuclideanDistance, dot_product_dense, euclidean_distance_dense,
};
use crate::quantizers::{Quantizer, QueryEvaluator};
use crate::{DenseComponent, DenseVector1D, FromF32, MutableVector1D, ValueType, Vector1D};

/// A plain dense quantizer that converts values from one float type to another.
///
/// - `Q`: Query value type
/// - `In`: Input value type (to be encoded)
/// - `Out`: Output value type (quantized)
/// - `D`: Distance type (EuclideanDistance or DotProduct)
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlainDenseQuantizer<Q, In, Out, D> {
    d: usize,
    _phantom: PhantomData<(Q, In, Out, D)>,
}

impl<Q, In, Out, D> PlainDenseQuantizer<Q, In, Out, D> {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            _phantom: PhantomData,
        }
    }
}

// Macro to implement Plain Quantizer for each existing distance type
macro_rules! impl_quantizer_for_distance {
    ($distance:ty) => {
        impl<Q, In, Out> Quantizer for PlainDenseQuantizer<Q, In, Out, $distance>
        where
            Q: ValueType,
            In: ValueType,
            Out: ValueType + FromF32,
        {
            type QueryValueType = Q;
            type QueryComponentType = DenseComponent;
            type InputValueType = In;
            type InputComponentType = DenseComponent;
            type OutputValueType = Out;
            type OutputComponentType = DenseComponent;
            type Evaluator = PlainDenseQueryEvaluator<Q, In, Out, $distance>;

            fn encode<InputVector, OutputVector>(
                &self,
                input_vector: InputVector,
                output_vector: &mut OutputVector,
            ) where
                InputVector: Vector1D<
                        ValueType = Self::InputValueType,
                        ComponentType = Self::InputComponentType,
                    >,
                OutputVector: MutableVector1D<
                        ValueType = Self::OutputValueType,
                        ComponentType = Self::OutputComponentType,
                    >,
            {
                let input = input_vector.values_as_slice();
                let output = output_vector.values_as_mut_slice();

                for (out_val, in_val) in output.iter_mut().zip(input.iter()) {
                    let f32_val = in_val.to_f32().unwrap();
                    *out_val = Out::from_f32_saturating(f32_val);
                }
            }

            fn get_query_evaluator<QueryVector>(&self, query: QueryVector) -> Self::Evaluator
            where
                QueryVector: Vector1D<
                        ValueType = Self::QueryValueType,
                        ComponentType = Self::QueryComponentType,
                    >,
            {
                PlainDenseQueryEvaluator::from_query(query)
            }

            #[inline]
            fn m(&self) -> usize {
                self.d
            }

            #[inline]
            fn dim(&self) -> usize {
                self.d
            }
        }
    };
}

impl_quantizer_for_distance!(EuclideanDistance);
impl_quantizer_for_distance!(DotProduct);

/// Query evaluator for PlainDenseQuantizer.
/// Stores the query and computes distances against encoded vectors.
pub struct PlainDenseQueryEvaluator<Q, In, Out, D>
where
    Q: ValueType,
    In: ValueType,
    Out: ValueType,
    D: Distance,
{
    query: Vec<Q>,
    _phantom: PhantomData<(In, Out, D)>,
}

impl<Q, In, Out, D> PlainDenseQueryEvaluator<Q, In, Out, D>
where
    Q: ValueType,
    In: ValueType,
    Out: ValueType,
    D: Distance,
{
    pub fn from_query<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<ValueType = Q, ComponentType = DenseComponent>,
    {
        Self {
            query: query.values_as_slice().to_vec(),
            _phantom: PhantomData,
        }
    }
}

impl<Q, In, Out> QueryEvaluator for PlainDenseQueryEvaluator<Q, In, Out, EuclideanDistance>
where
    Q: ValueType,
    In: ValueType,
    Out: ValueType + FromF32,
{
    type QuantizerType = PlainDenseQuantizer<Q, In, Out, EuclideanDistance>;
    type DistanceType = EuclideanDistance;

    fn new<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<
                ValueType = <Self::QuantizerType as Quantizer>::QueryValueType,
                ComponentType = <Self::QuantizerType as Quantizer>::QueryComponentType,
            >,
    {
        Self::from_query(query)
    }

    fn compute_distance<EncodedVector>(&self, vector: EncodedVector) -> Self::DistanceType
    where
        EncodedVector: Vector1D<
                ValueType = <Self::QuantizerType as Quantizer>::OutputValueType,
                ComponentType = <Self::QuantizerType as Quantizer>::OutputComponentType,
            >,
    {
        let query = DenseVector1D::new(self.query.as_slice());
        let vec = DenseVector1D::new(vector.values_as_slice());
        euclidean_distance_dense(query, vec)
    }
}

impl<Q, In, Out> QueryEvaluator for PlainDenseQueryEvaluator<Q, In, Out, DotProduct>
where
    Q: ValueType,
    In: ValueType,
    Out: ValueType + FromF32,
{
    type QuantizerType = PlainDenseQuantizer<Q, In, Out, DotProduct>;
    type DistanceType = DotProduct;

    fn new<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<
                ValueType = <Self::QuantizerType as Quantizer>::QueryValueType,
                ComponentType = <Self::QuantizerType as Quantizer>::QueryComponentType,
            >,
    {
        Self::from_query(query)
    }

    fn compute_distance<EncodedVector>(&self, vector: EncodedVector) -> Self::DistanceType
    where
        EncodedVector: Vector1D<
                ValueType = <Self::QuantizerType as Quantizer>::OutputValueType,
                ComponentType = <Self::QuantizerType as Quantizer>::OutputComponentType,
            >,
    {
        let query = DenseVector1D::new(self.query.as_slice());
        let vec = DenseVector1D::new(vector.values_as_slice());
        dot_product_dense(query, vec)
    }
}
