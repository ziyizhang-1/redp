#![allow(non_upper_case_globals)]

use super::types::{RawEmonDataFrame, RawEmonDataFrameColumns};
use polars::lazy::dsl::{col, lit, when};
use polars::prelude::*;

static redc: RawEmonDataFrameColumns = RawEmonDataFrameColumns::new();

pub trait Normalizer {
    type Output;
    /// Computes normalized event count
    /// @param df: data frame containing data to normalize
    /// @param event_axis: axis where event names exist, must be either 'columns' or 'index'
    /// @return a copy of df where the "value" column is updated to contain normalized values
    fn normalize(self, ref_tsc: f32, event_axis: &str) -> Self::Output;
}

impl Normalizer for RawEmonDataFrame {
    type Output = RawEmonDataFrame;
    fn normalize(self, ref_tsc: f32, event_axis: &str) -> Self::Output {
        let events_to_exclude: Vec<&str> = ["$samplingTime", "$processed_samples"].into();

        assert!(
            ["columns", "index"].contains(&event_axis),
            "'event_axis' argument must be either 'columns' or 'index'"
        );

        if !self.schema().unwrap().get_names().contains(&redc.NAME) {
            self.with_column(col(redc.VALUE) / col(redc.TSC) * lit(ref_tsc).alias(redc.VALUE))
        } else if event_axis == "columns" {
            self.with_column(
                when(col(redc.NAME).is_in(lit(Series::from_iter(events_to_exclude))))
                    .then(col(redc.VALUE))
                    .otherwise(col(redc.VALUE) / col(redc.TSC) * lit(ref_tsc)),
            )
        } else {
            self.with_column(col(redc.VALUE) / col(redc.TSC) * lit(ref_tsc).alias(redc.VALUE))
        }
    }
}
