pub mod group_by_ext;
pub mod pivot_stable_seq;
pub mod transpose_seq;

use polars::prelude::*;
use rayon::iter::Either;

pub trait GroupByExt {
    fn keys_sliced_seq(&self, slice: Option<(i64, usize)>, by: Vec<Series>) -> Vec<Series>;
    fn keys_seq(&self, by: Vec<Series>) -> Vec<Series>;
    fn prepare_agg_seq(
        &self,
        by: Vec<Series>,
        selected_agg: Option<Vec<String>>,
    ) -> PolarsResult<(Vec<Series>, Vec<Series>)>;
    fn sum_seq(
        &self,
        by: Vec<Series>,
        selected_agg: Option<Vec<String>>,
    ) -> PolarsResult<DataFrame>;
}

pub trait DataframeSeqExt: Sized {
    fn transpose_seq(
        &mut self,
        keep_name_as: Option<&str>,
        new_col_names: Option<Either<String, Vec<String>>>,
    ) -> PolarsResult<DataFrame>;
    fn transpose_from_dtype(
        &self,
        dtype: &DataType,
        keep_names_as: Option<&str>,
        names_out: &[String],
    ) -> PolarsResult<DataFrame>;
    fn drop_nulls_seq<S: AsRef<str>>(&self, subset: Option<&[S]>) -> PolarsResult<Self>;
    fn fill_null_seq(&self, strategy: FillNullStrategy) -> PolarsResult<Self>;
}
