use super::Arithmetic;
use crate::redp::core::types::MetricDefinition;
use crate::SymbolTable;
use polars::lazy::dsl::{col, cols, lit, Expr};
use polars::prelude::*;
use std::collections::HashMap;

pub trait MetricCompute {
    /// Compute metrics on the input dataframe
    /// :param df: input dataframe containing event counts.
    /// `df` is expected to have the following structure:
    /// - Columns: event names, where each column represents a single event
    /// - Index: If `calculate_block_level` is True, `df` must have a multi-index where the first two levels
    ///          represent timestamp and event group id.
    ///          `df` can have additional levels, e.g. for socket, core, thread...
    /// - Rows: event counts for each event
    ///
    /// :param constant_values: an optional dictionary that maps constant expressions, used in metric formulas,
    ///                         to their values (e.g. 'system.socket_count')
    ///
    /// :param calculate_block_level: whether to calculate block level metrics in addition to sample level metrics
    ///
    /// :param group_index_name: index level name in `df` to use for determining event group id.
    ///                          Used only when `calculate_block_level` is True.
    ///
    /// :param timestamp_index_name: index level name in `df` to use for determining the timestamp.
    ///                              Used only when `calculate_block_level` is True.
    ///
    /// :return: a new dataframe where each column is a metric. The index of the result dataframe is identical to `df`
    fn compute_metric(
        &self,
        event_df: LazyFrame,
        constant_values: Option<&HashMap<String, f32>>,
        calculate_block_level: bool,
        index: Option<&[&str]>,
    ) -> LazyFrame;

    /// Special logic to extract the value out of 'per_socket' metric constants so these can be set to 1 for uncore
    /// views. Any special logic to change the symbol_table that requires the metric_constants should go here.
    /// This only needs to be done if there is a hardcoded metric constant in the metric file that needs to be updated
    /// later for a specific view.
    /// i.e. {'channels_populated_per_socket': 8} becomes {'channels_populated_per_socket':
    /// 'system.channels_populated_per_socket', 'system.channels_populated_per_socket': 8}
    /// @param constants: a dictionary of a metric constant name and corresponding value
    /// @return: An updated metric constant dictionary for (previously) hardcoded metric constants
    #[allow(dead_code)]
    fn update_symbol_table(&self, symbol_table: SymbolTable) -> SymbolTable;
}

#[allow(deprecated)]
impl MetricCompute for Vec<MetricDefinition> {
    fn compute_metric(
        &self,
        event_df: LazyFrame,
        constant_values: Option<&HashMap<String, f32>>,
        calculate_block_level: bool,
        index: Option<&[&str]>,
    ) -> LazyFrame {
        let index = index.unwrap_or(&[]);

        let df = match constant_values {
            Some(v) => event_df
                .with_columns_seq(
                    v.iter()
                        .map(|(name, value)| lit(*value).alias(name).strict_cast(DataType::Float32))
                        .collect::<Vec<Expr>>(),
                )
                .collect()
                .unwrap(),
            None => event_df.collect().unwrap(),
        };
        // Compute sample-level metrics for the input dataframe (in lazy mode)
        let sample_level_metrics_df = LazyFrame::default()
            .with_columns_seq(
                index
                    .iter()
                    .map(|x| lit(df.column(x).unwrap().to_owned()))
                    .collect::<Vec<Expr>>(),
            )
            .with_columns_seq(
                self.iter()
                    .filter(|x| x.all_metric_references_are_available(&df))
                    .map(|x| x.arithmetic(&df).strict_cast(DataType::Float32))
                    .collect::<Vec<Expr>>(),
            );
        // Compute block-level metrics for the input dataframe (in lazy mode)
        let block_level_metrics_df = match calculate_block_level {
            true => {
                assert!(index.len() >= 2);
                let mut column_names = df
                    .fields()
                    .into_iter()
                    .filter_map(|x| match x.data_type() {
                        &DataType::Float32 => Some(x.name.to_string()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                column_names.insert(0, index[1].to_owned());

                let mut block_avg_df = df
                    .group_by_with_series(
                        df.select_series(index.iter().filter(|x| *x != &index[1]).copied())
                            .unwrap(),
                        false,
                        true,
                    )
                    .unwrap()
                    .select(column_names)
                    .agg_list()
                    .unwrap();

                block_avg_df
                    .rename(&(index[1].to_owned() + "_agg_list"), index[1])
                    .unwrap();
                block_avg_df
                    .apply(index[1], |s| {
                        s.list()
                            .unwrap()
                            .into_iter()
                            .map(|x| x.and_then(|t| t.max::<i64>().unwrap()))
                            .collect::<Int64Chunked>()
                            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                            .unwrap()
                    })
                    .unwrap();

                df.fields().into_iter().for_each(|x| {
                    if let &DataType::Float32 = x.data_type() {
                        block_avg_df
                            .rename(&(x.name.clone() + "_agg_list"), &x.name)
                            .unwrap();
                        block_avg_df
                            .apply(&x.name, |s| {
                                s.list()
                                    .unwrap()
                                    .into_iter()
                                    .map(|x| x.and_then(|f| f.mean()))
                                    .collect::<Float64Chunked>()
                                    .cast(&DataType::Float32)
                                    .unwrap()
                                    .with_name(&x.name)
                            })
                            .unwrap();
                    }
                });

                drop(df);

                let is_null = sample_level_metrics_df
                    .clone()
                    .select_seq([col("*").is_null().all(false)])
                    .collect()
                    .unwrap();

                block_avg_df
                    .clone()
                    .lazy()
                    .select_seq(&[cols(index)])
                    .with_columns_seq(
                        self.iter()
                            .filter(|x| x.all_metric_references_are_available(&block_avg_df))
                            .filter_map(|x| {
                                if is_null
                                    .column(&x.name)
                                    .unwrap()
                                    .bool()
                                    .unwrap()
                                    .get(0)
                                    .unwrap()
                                {
                                    Some(x.arithmetic(&block_avg_df).strict_cast(DataType::Float32))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<Expr>>(),
                    )
            }
            false => {
                return sample_level_metrics_df;
            }
        };
        let block_level_metrics_cols = block_level_metrics_df.schema().unwrap();
        let merged_cols: Vec<String> = sample_level_metrics_df
            .schema()
            .unwrap()
            .get_names()
            .into_iter()
            .map(|x| {
                if index.contains(&x) {
                    x.to_owned()
                } else if block_level_metrics_cols.get_names().contains(&x) {
                    x.to_owned() + "_right"
                } else {
                    x.to_owned()
                }
            })
            .collect();

        sample_level_metrics_df
            .join_builder()
            .with(block_level_metrics_df)
            .how(JoinType::Left)
            .left_on(index.iter().map(|x| col(x)).collect::<Vec<Expr>>())
            .right_on(index.iter().map(|x| col(x)).collect::<Vec<Expr>>())
            .allow_parallel(false)
            .suffix("_right")
            .finish()
            .select_seq(
                merged_cols
                    .iter()
                    .map(|x| col(x).alias(&x.replace("_right", "")))
                    .collect::<Vec<Expr>>(),
            )
    }

    #[allow(unused_variables)]
    fn update_symbol_table(&self, symbol_table: SymbolTable) -> SymbolTable {
        unimplemented!()
    }
}
