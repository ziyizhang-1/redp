#![allow(dead_code)]
use polars::frame::group_by::expr::PhysicalAggExpr;
use polars::prelude::*;
use polars_arrow::legacy::trusted_len::TrustedLenPush;
use polars_arrow::legacy::utils::CustomIterTools;
use polars_expr::planner::{create_physical_expr, ExpressionConversionState};
use polars_expr::prelude::*;
use polars_plan::logical_plan::Context;
use polars_utils::arena::Arena;
use polars_utils::sync::SyncPtr;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use core::hash::Hash;
use std::borrow::Cow;

const HASHMAP_INIT_SIZE: usize = 512;

/// Apply a macro on the Downcasted ChunkedArray's of DataTypes that are logical numerics.
/// So no logical.
macro_rules! downcast_as_macro_arg_physical {
    ($self:expr, $macro:ident $(, $opt_args:expr)*) => {{
        match $self.dtype() {
            DataType::UInt32 => $macro!($self.u32().unwrap() $(, $opt_args)*),
            DataType::UInt64 => $macro!($self.u64().unwrap() $(, $opt_args)*),
            DataType::Int32 => $macro!($self.i32().unwrap() $(, $opt_args)*),
            DataType::Int64 => $macro!($self.i64().unwrap() $(, $opt_args)*),
            DataType::Float32 => $macro!($self.f32().unwrap() $(, $opt_args)*),
            DataType::Float64 => $macro!($self.f64().unwrap() $(, $opt_args)*),
            dt => panic!("not implemented for {:?}", dt),
        }
    }};
}

#[derive(Clone)]
enum PivotAgg {
    First,
    Sum,
    Min,
    Max,
    Mean,
    Median,
    Count,
    Last,
    Expr(Arc<dyn PhysicalAggExpr + Send + Sync>),
}

fn restore_logical_type(s: &Series, logical_type: &DataType) -> Series {
    // restore logical type
    match (logical_type, s.dtype()) {
        (DataType::Float32, DataType::UInt32) => {
            let ca = s.u32().unwrap();
            ca._reinterpret_float().into_series()
        }
        (DataType::Float64, DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca._reinterpret_float().into_series()
        }
        (DataType::Int32, DataType::UInt32) => {
            let ca = s.u32().unwrap();
            ca.reinterpret_signed()
        }
        (DataType::Int64, DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca.reinterpret_signed()
        }
        _ => unsafe { s.cast_unchecked(logical_type).unwrap() },
    }
}

fn prepare_expression_for_context(
    name: &str,
    expr: &Expr,
    dtype: &DataType,
    ctxt: Context,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    let mut lp_arena = Arena::with_capacity(8);
    let mut expr_arena = Arena::with_capacity(10);

    // create a dummy lazyframe and run a very simple optimization run so that
    // type coercion and simplify expression optimizations run.
    let column = Series::full_null(name, 0, dtype);
    let lf = column
        .into_frame()
        .lazy()
        .without_optimizations()
        .with_simplify_expr(true)
        .select_seq([expr.clone()]);
    let optimized = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    let lp = lp_arena.get(optimized);
    let aexpr = lp.get_exprs().pop().unwrap();

    create_physical_expr(
        &aexpr,
        ctxt,
        &expr_arena,
        None,
        &mut ExpressionConversionState::new(true, 0),
    )
}

struct PivotExpr(Expr);

impl PhysicalAggExpr for PivotExpr {
    fn evaluate(&self, df: &DataFrame, groups: &GroupsProxy) -> PolarsResult<Series> {
        let state = ExecutionState::new();
        let dtype = df.get_columns()[0].dtype();
        let phys_expr = prepare_expression_for_context("", &self.0, dtype, Context::Aggregation)?;
        phys_expr
            .evaluate_on_groups(df, groups, &state)
            .map(|mut ac| ac.aggregated())
    }

    fn root_name(&self) -> PolarsResult<&str> {
        Ok("")
    }
}

fn prepare_eval_expr(expr: Expr) -> Expr {
    expr.map_expr(|e| match e {
        Expr::Column(_) => Expr::Column(Arc::from("")),
        Expr::Nth(_) => Expr::Column(Arc::from("")),
        e => e,
    })
}

pub fn pivot_stable_seq<I0, I1, I2, S0, S1, S2>(
    df: &DataFrame,
    index: I0,
    columns: I1,
    values: Option<I2>,
    sort_columns: bool,
    agg_expr: Option<Expr>,
    // used as separator/delimiter in generated column names.
    separator: Option<&str>,
) -> PolarsResult<DataFrame>
where
    I0: IntoIterator<Item = S0>,
    I1: IntoIterator<Item = S1>,
    I2: IntoIterator<Item = S2>,
    S0: AsRef<str>,
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    // make sure that the root column is replaced
    let agg_expr = agg_expr.map(|agg_expr| {
        let expr = prepare_eval_expr(agg_expr);
        PivotAgg::Expr(Arc::new(PivotExpr(expr)))
    });
    pivot_stable(
        df,
        index,
        columns,
        values,
        sort_columns,
        agg_expr,
        separator,
    )
}

/// Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.
///
/// # Note
/// Polars'/arrow memory is not ideal for transposing operations like pivots.
/// If you have a relatively large table, consider using a group_by over a pivot.
fn pivot_stable<I0, I1, I2, S0, S1, S2>(
    pivot_df: &DataFrame,
    index: I0,
    columns: I1,
    values: Option<I2>,
    sort_columns: bool,
    agg_fn: Option<PivotAgg>,
    separator: Option<&str>,
) -> PolarsResult<DataFrame>
where
    I0: IntoIterator<Item = S0>,
    I1: IntoIterator<Item = S1>,
    I2: IntoIterator<Item = S2>,
    S0: AsRef<str>,
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    let index = index
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();
    let columns = columns
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();
    let values = get_values_columns(pivot_df, &index, &columns, values);
    pivot_impl(
        pivot_df,
        &index,
        &columns,
        &values,
        agg_fn,
        sort_columns,
        true,
        separator,
    )
}

/// Determine `values` columns, which is optional in `pivot` calls.
///
/// If not specified (i.e. is `None`), use all remaining columns in the
/// `DataFrame` after `index` and `columns` have been excluded.
fn get_values_columns<I, S>(
    df: &DataFrame,
    index: &[String],
    columns: &[String],
    values: Option<I>,
) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    match values {
        Some(v) => v.into_iter().map(|s| s.as_ref().to_string()).collect(),
        None => df
            .get_column_names()
            .into_iter()
            .map(|c| c.to_string())
            .filter(|c| !(index.contains(c) | columns.contains(c)))
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn pivot_impl(
    pivot_df: &DataFrame,
    // keys of the first group_by operation
    index: &[String],
    // these columns will be used for a nested group_by
    // the rows of this nested group_by will be pivoted as header column values
    columns: &[String],
    // these columns will be aggregated in the nested group_by
    values: &[String],
    // aggregation function
    agg_fn: Option<PivotAgg>,
    sort_columns: bool,
    stable: bool,
    // used as separator/delimiter in generated column names.
    separator: Option<&str>,
) -> PolarsResult<DataFrame> {
    polars_ensure!(!index.is_empty(), ComputeError: "index cannot be zero length");
    polars_ensure!(!columns.is_empty(), ComputeError: "columns cannot be zero length");
    if !stable {
        println!("unstable pivot not yet supported, using stable pivot");
    };
    if columns.len() > 1 {
        let schema = Arc::new(pivot_df.schema());
        let binding = pivot_df.select_with_schema(columns, &schema)?;
        let fields = binding.get_columns();
        let column = format!("{{\"{}\"}}", columns.join("\",\""));
        if schema.contains(column.as_str()) {
            polars_bail!(ComputeError: "cannot use column name {column} that \
            already exists in the DataFrame. Please rename it prior to calling `pivot`.")
        }
        let columns_struct = StructChunked::new(&column, fields).unwrap().into_series();
        let mut binding = pivot_df.clone();
        let pivot_df = unsafe { binding.with_column_unchecked(columns_struct) };
        pivot_impl_single_column(
            pivot_df,
            index,
            &column,
            values,
            agg_fn,
            sort_columns,
            separator,
        )
    } else {
        pivot_impl_single_column(
            pivot_df,
            index,
            unsafe { columns.get_unchecked(0) },
            values,
            agg_fn,
            sort_columns,
            separator,
        )
    }
}

fn pivot_impl_single_column(
    pivot_df: &DataFrame,
    index: &[String],
    column: &str,
    values: &[String],
    agg_fn: Option<PivotAgg>,
    sort_columns: bool,
    separator: Option<&str>,
) -> PolarsResult<DataFrame> {
    let sep = separator.unwrap_or("_");
    let mut final_cols = vec![];
    let mut count = 0;
    let out: PolarsResult<()> = {
        let mut group_by = index.to_vec();
        group_by.push(column.to_string());

        let groups = pivot_df
            .group_by_with_series(pivot_df.select_series(group_by)?, false, true)?
            .take_groups();

        let col = compute_col_idx(pivot_df, column, &groups);
        let row = compute_row_idx(pivot_df, index, &groups, count);
        let (col_locations, column_agg) = col?;
        let (row_locations, n_rows, mut row_index) = row?;

        for value_col_name in values {
            let value_col = pivot_df.column(value_col_name)?;

            use PivotAgg::*;
            let value_agg = unsafe {
                match &agg_fn {
                    None => match value_col.len() > groups.len() {
                        true => {
                            polars_bail!(ComputeError: "found multiple elements in the same group, please specify an aggregation function")
                        }
                        false => value_col.agg_first(&groups),
                    },
                    Some(agg_fn) => match agg_fn {
                        Sum => value_col.agg_sum(&groups),
                        Min => value_col.agg_min(&groups),
                        Max => value_col.agg_max(&groups),
                        Last => value_col.agg_last(&groups),
                        First => value_col.agg_first(&groups),
                        Mean => value_col.agg_mean(&groups),
                        Median => value_col.agg_median(&groups),
                        Count => groups.group_count().into_series(),
                        Expr(ref expr) => {
                            let name = expr.root_name()?;
                            let mut value_col = value_col.clone();
                            value_col.rename(name);
                            let tmp_df = value_col.into_frame();
                            let mut aggregated = expr.evaluate(&tmp_df, &groups)?;
                            aggregated.rename(value_col_name);
                            aggregated
                        }
                    },
                }
            };

            let headers = column_agg.unique_stable()?.cast(&DataType::String)?;
            let mut headers = headers.str().unwrap().clone();
            if values.len() > 1 {
                // TODO! MILESTONE 1.0: change to `format!("{value_col_name}{sep}{v}")`
                headers = headers
                    .apply_values(|v| Cow::from(format!("{value_col_name}{sep}{column}{sep}{v}")))
            }

            let n_cols = headers.len();
            let value_agg_phys = value_agg.to_physical_repr();
            let logical_type = value_agg.dtype();

            debug_assert_eq!(row_locations.len(), col_locations.len());
            debug_assert_eq!(value_agg_phys.len(), row_locations.len());

            let mut cols = if value_agg_phys.dtype().is_numeric() {
                macro_rules! dispatch {
                    ($ca:expr) => {{
                        position_aggregates_numeric(
                            n_rows,
                            n_cols,
                            &row_locations,
                            &col_locations,
                            $ca,
                            logical_type,
                            &headers,
                        )
                    }};
                }
                downcast_as_macro_arg_physical!(value_agg_phys, dispatch)
            } else {
                position_aggregates(
                    n_rows,
                    n_cols,
                    &row_locations,
                    &col_locations,
                    &value_agg_phys,
                    logical_type,
                    &headers,
                )
            };

            if sort_columns {
                cols.sort_unstable_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
            }

            let cols = if count == 0 {
                let mut final_cols = row_index.take().unwrap();
                final_cols.extend(cols);
                final_cols
            } else {
                cols
            };
            count += 1;
            final_cols.extend_from_slice(&cols);
        }
        Ok(())
    };
    out?;

    // SAFETY: length has already been checked.
    unsafe { DataFrame::new_no_length_checks(final_cols) }
}

fn _split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect_trusted()
    }
}

fn position_aggregates(
    n_rows: usize,
    n_cols: usize,
    row_locations: &[IdxSize],
    col_locations: &[IdxSize],
    value_agg_phys: &Series,
    logical_type: &DataType,
    headers: &StringChunked,
) -> Vec<Series> {
    let mut buf = vec![AnyValue::Null; n_rows * n_cols];
    let start_ptr = buf.as_mut_ptr() as usize;

    let n_threads = 1;
    let split = _split_offsets(row_locations.len(), n_threads);

    // ensure the slice series are not dropped
    // so the AnyValues are referencing correct data, if they reference arrays (struct)
    let n_splits = split.len();
    let mut arrays: Vec<Series> = Vec::with_capacity(n_splits);

    // every thread will only write to their partition
    let array_ptr = unsafe { SyncPtr::new(arrays.as_mut_ptr()) };

    split
        .into_iter()
        .enumerate()
        .for_each(|(i, (offset, len))| {
            let start_ptr = start_ptr as *mut AnyValue;
            let row_locations = &row_locations[offset..offset + len];
            let col_locations = &col_locations[offset..offset + len];
            let value_agg_phys = value_agg_phys.slice(offset as i64, len);

            for ((row_idx, col_idx), val) in row_locations
                .iter()
                .zip(col_locations)
                .zip(value_agg_phys.phys_iter())
            {
                // SAFETY:
                // in bounds
                unsafe {
                    let idx = *row_idx as usize + *col_idx as usize * n_rows;
                    debug_assert!(idx < buf.len());
                    let pos = start_ptr.add(idx);
                    std::ptr::write(pos, val)
                }
            }
            // ensure the `values_agg_phys` stays alive
            let array_ptr = array_ptr.clone().get();
            unsafe { std::ptr::write(array_ptr.add(i), value_agg_phys) }
        });
    // ensure the content of the arrays are dropped
    unsafe {
        arrays.set_len(n_splits);
    }

    let headers_iter = headers.iter();
    let phys_type = logical_type.to_physical();

    (0..n_cols)
        .zip(headers_iter)
        .map(|(i, opt_name)| {
            let offset = i * n_rows;
            let avs = &buf[offset..offset + n_rows];
            let name = opt_name.unwrap_or("null");
            let out = Series::from_any_values_and_dtype(name, avs, &phys_type, false).unwrap();
            unsafe { out.cast_unchecked(logical_type).unwrap() }
        })
        .collect::<Vec<_>>()
}

fn position_aggregates_numeric<T>(
    n_rows: usize,
    n_cols: usize,
    row_locations: &[IdxSize],
    col_locations: &[IdxSize],
    value_agg_phys: &ChunkedArray<T>,
    logical_type: &DataType,
    headers: &StringChunked,
) -> Vec<Series>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let mut buf = vec![None; n_rows * n_cols];
    let start_ptr = buf.as_mut_ptr() as usize;

    let n_threads = 1;

    let split = _split_offsets(row_locations.len(), n_threads);
    let n_splits = split.len();
    // ensure the arrays are not dropped
    // so the AnyValues are referencing correct data, if they reference arrays (struct)
    let mut arrays: Vec<ChunkedArray<T>> = Vec::with_capacity(n_splits);

    // every thread will only write to their partition
    let array_ptr = unsafe { SyncPtr::new(arrays.as_mut_ptr()) };

    split
        .into_iter()
        .enumerate()
        .for_each(|(i, (offset, len))| {
            let start_ptr = start_ptr as *mut Option<T::Native>;
            let row_locations = &row_locations[offset..offset + len];
            let col_locations = &col_locations[offset..offset + len];
            let value_agg_phys = value_agg_phys.slice(offset as i64, len);

            // todo! remove lint silencing
            #[allow(clippy::useless_conversion)]
            for ((row_idx, col_idx), val) in row_locations
                .iter()
                .zip(col_locations)
                .zip(value_agg_phys.into_iter())
            {
                // SAFETY:
                // in bounds
                unsafe {
                    let idx = *row_idx as usize + *col_idx as usize * n_rows;
                    debug_assert!(idx < buf.len());
                    let pos = start_ptr.add(idx);
                    std::ptr::write(pos, val)
                }
            }
            // ensure the `values_agg_phys` stays alive
            let array_ptr = array_ptr.clone().get();
            unsafe { std::ptr::write(array_ptr.add(i), value_agg_phys) }
        });
    // ensure the content of the arrays are dropped
    unsafe {
        arrays.set_len(n_splits);
    }
    let headers_iter = headers.iter();

    (0..n_cols)
        .zip(headers_iter)
        .map(|(i, opt_name)| {
            let offset = i * n_rows;
            let opt_values = &buf[offset..offset + n_rows];
            let name = opt_name.unwrap_or("null");
            let out = ChunkedArray::<T>::from_slice_options(name, opt_values).into_series();
            unsafe { out.cast_unchecked(logical_type).unwrap() }
        })
        .collect::<Vec<_>>()
}

fn compute_row_idx(
    pivot_df: &DataFrame,
    index: &[String],
    groups: &GroupsProxy,
    count: usize,
) -> PolarsResult<(Vec<IdxSize>, usize, Option<Vec<Series>>)> {
    let (row_locations, n_rows, row_index) = if index.len() == 1 {
        let index_s = pivot_df.column(&index[0])?;
        let index_agg = unsafe { index_s.agg_first(groups) };
        let index_agg_physical = index_agg.to_physical_repr();

        use DataType::*;
        match index_agg_physical.dtype() {
            Int32 | UInt32 => {
                let ca = index_agg_physical.bit_repr_small();
                compute_row_index(index, &ca, count, index_s.dtype())
            }
            Int64 | UInt64 => {
                let ca = index_agg_physical.bit_repr_large();
                compute_row_index(index, &ca, count, index_s.dtype())
            }
            Float64 => {
                let ca: &ChunkedArray<Float64Type> = index_agg_physical.as_ref().as_ref().as_ref();
                compute_row_index(index, ca, count, index_s.dtype())
            }
            Float32 => {
                let ca: &ChunkedArray<Float32Type> = index_agg_physical.as_ref().as_ref().as_ref();
                compute_row_index(index, ca, count, index_s.dtype())
            }
            Boolean => {
                let ca = index_agg_physical.bool().unwrap();
                compute_row_index(index, ca, count, index_s.dtype())
            }
            Struct(_) => {
                let ca = index_agg_physical.struct_().unwrap();
                let ca = ca.rows_encode()?;
                compute_row_index_struct(index, &index_agg, &ca, count)
            }
            String => {
                let ca = index_agg_physical.str().unwrap();
                compute_row_index(index, ca, count, index_s.dtype())
            }
            _ => {
                let mut row_to_idx =
                    PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
                let mut idx = 0 as IdxSize;
                let row_locations = index_agg_physical
                    .phys_iter()
                    .map(|v| {
                        let idx = *row_to_idx.entry(v).or_insert_with(|| {
                            let old_idx = idx;
                            idx += 1;
                            old_idx
                        });
                        idx
                    })
                    .collect::<Vec<_>>();

                let row_index = match count {
                    0 => {
                        let s = Series::new(
                            &index[0],
                            row_to_idx.into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
                        );
                        let s = restore_logical_type(&s, index_s.dtype());
                        Some(vec![s])
                    }
                    _ => None,
                };

                (row_locations, idx as usize, row_index)
            }
        }
    } else {
        let binding = pivot_df.select(index)?;
        let fields = binding.get_columns();
        let index_struct_series = StructChunked::new("placeholder", fields)?.into_series();
        let index_agg = unsafe { index_struct_series.agg_first(groups) };
        let index_agg_physical = index_agg.to_physical_repr();
        let ca = index_agg_physical.struct_()?;
        let ca = ca.rows_encode()?;
        let (row_locations, n_rows, row_index) =
            compute_row_index_struct(index, &index_agg, &ca, count);
        let row_index = row_index.map(|x| {
            unsafe { x.get_unchecked(0) }
                .struct_()
                .unwrap()
                .fields()
                .to_vec()
        });
        (row_locations, n_rows, row_index)
    };

    Ok((row_locations, n_rows, row_index))
}

fn compute_row_index<'a, T>(
    index: &[String],
    index_agg_physical: &'a ChunkedArray<T>,
    count: usize,
    logical_type: &DataType,
) -> (Vec<IdxSize>, usize, Option<Vec<Series>>)
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + Copy + ToTotalOrd,
    <Option<T::Physical<'a>> as ToTotalOrd>::TotalOrdItem: Hash + Eq,
    ChunkedArray<T>: FromIterator<Option<T::Physical<'a>>>,
    ChunkedArray<T>: IntoSeries,
{
    let mut row_to_idx =
        PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
    let mut idx = 0 as IdxSize;

    let mut row_locations = Vec::with_capacity(index_agg_physical.len());
    for opt_v in index_agg_physical.iter() {
        let opt_v = opt_v.to_total_ord();
        let idx = *row_to_idx.entry(opt_v).or_insert_with(|| {
            let old_idx = idx;
            idx += 1;
            old_idx
        });

        // SAFETY:
        // we pre-allocated
        unsafe {
            row_locations.push_unchecked(idx);
        }
    }
    let row_index = match count {
        0 => {
            let mut s = row_to_idx
                .into_iter()
                .map(|(k, _)| Option::<T::Physical<'a>>::peel_total_ord(k))
                .collect::<ChunkedArray<T>>()
                .into_series();
            s.rename(&index[0]);
            let s = restore_logical_type(&s, logical_type);
            Some(vec![s])
        }
        _ => None,
    };

    (row_locations, idx as usize, row_index)
}

fn compute_row_index_struct(
    index: &[String],
    index_agg: &Series,
    index_agg_physical: &BinaryOffsetChunked,
    count: usize,
) -> (Vec<IdxSize>, usize, Option<Vec<Series>>) {
    let mut row_to_idx =
        PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
    let mut idx = 0 as IdxSize;

    let mut row_locations = Vec::with_capacity(index_agg_physical.len());
    let mut unique_indices = Vec::with_capacity(index_agg_physical.len());
    let mut row_number: IdxSize = 0;
    for arr in index_agg_physical.downcast_iter() {
        for opt_v in arr.iter() {
            let idx = *row_to_idx.entry(opt_v).or_insert_with(|| {
                // SAFETY: we pre-allocated
                unsafe { unique_indices.push_unchecked(row_number) };
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            row_number += 1;

            // SAFETY:
            // we pre-allocated
            unsafe {
                row_locations.push_unchecked(idx);
            }
        }
    }
    let row_index = match count {
        0 => {
            // SAFETY: `unique_indices` is filled with elements between
            // 0 and `index_agg.len() - 1`.
            let mut s = unsafe { index_agg.take_slice_unchecked(&unique_indices) };
            s.rename(&index[0]);
            Some(vec![s])
        }
        _ => None,
    };

    (row_locations, idx as usize, row_index)
}

fn compute_col_idx(
    pivot_df: &DataFrame,
    column: &str,
    groups: &GroupsProxy,
) -> PolarsResult<(Vec<IdxSize>, Series)> {
    let column_s = pivot_df.column(column)?;
    let column_agg = unsafe { column_s.agg_first(groups) };
    let column_agg_physical = column_agg.to_physical_repr();

    use DataType::*;
    let col_locations = match column_agg_physical.dtype() {
        Int32 | UInt32 => {
            let ca = column_agg_physical.bit_repr_small();
            compute_col_idx_numeric(&ca)
        }
        Int64 | UInt64 => {
            let ca = column_agg_physical.bit_repr_large();
            compute_col_idx_numeric(&ca)
        }
        Float64 => {
            let ca: &ChunkedArray<Float64Type> = column_agg_physical.as_ref().as_ref().as_ref();
            compute_col_idx_numeric(ca)
        }
        Float32 => {
            let ca: &ChunkedArray<Float32Type> = column_agg_physical.as_ref().as_ref().as_ref();
            compute_col_idx_numeric(ca)
        }
        Struct(_) => {
            let ca = column_agg_physical.struct_().unwrap();
            let ca = ca.rows_encode()?;
            compute_col_idx_gen(&ca)
        }
        String => {
            let ca = column_agg_physical.str().unwrap();
            let ca = ca.as_binary();
            compute_col_idx_gen(&ca)
        }
        Binary => {
            let ca = column_agg_physical.binary().unwrap();
            compute_col_idx_gen(ca)
        }
        Boolean => {
            let ca = column_agg_physical.bool().unwrap();
            compute_col_idx_gen(ca)
        }
        _ => {
            let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
            let mut idx = 0 as IdxSize;
            column_agg_physical
                .phys_iter()
                .map(|v| {
                    let idx = *col_to_idx.entry(v).or_insert_with(|| {
                        let old_idx = idx;
                        idx += 1;
                        old_idx
                    });
                    idx
                })
                .collect()
        }
    };

    Ok((col_locations, column_agg))
}

fn compute_col_idx_numeric<T>(column_agg_physical: &ChunkedArray<T>) -> Vec<IdxSize>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
    let mut idx = 0 as IdxSize;
    let mut out = Vec::with_capacity(column_agg_physical.len());

    for opt_v in column_agg_physical.iter() {
        let opt_v = opt_v.to_total_ord();
        let idx = *col_to_idx.entry(opt_v).or_insert_with(|| {
            let old_idx = idx;
            idx += 1;
            old_idx
        });
        // SAFETY:
        // we pre-allocated
        unsafe { out.push_unchecked(idx) };
    }
    out
}

fn compute_col_idx_gen<'a, T>(column_agg_physical: &'a ChunkedArray<T>) -> Vec<IdxSize>
where
    T: PolarsDataType,
    &'a T::Array: IntoIterator<Item = Option<T::Physical<'a>>>,
    T::Physical<'a>: Hash + Eq,
{
    let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
    let mut idx = 0 as IdxSize;
    let mut out = Vec::with_capacity(column_agg_physical.len());

    for arr in column_agg_physical.downcast_iter() {
        for opt_v in arr.into_iter() {
            let idx = *col_to_idx.entry(opt_v).or_insert_with(|| {
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            // SAFETY:
            // we pre-allocated
            unsafe { out.push_unchecked(idx) };
        }
    }
    out
}
