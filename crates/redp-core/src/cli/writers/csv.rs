#![allow(non_upper_case_globals)]
use crate::redp::core::types::RawEmonDataFrameColumns;
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

static redc: RawEmonDataFrameColumns = RawEmonDataFrameColumns::new();

pub trait CsvWrite {
    fn to_csv(self, path: &Path, mem: bool) -> PolarsResult<()>;
}

impl CsvWrite for DataFrame {
    fn to_csv(self, path: &Path, mem: bool) -> PolarsResult<()> {
        let file = File::create(path)?;
        let mut df = self;

        if !mem {
            let cols = df.get_column_names_owned();

            if cols.starts_with(&[
                redc.GROUP.into(),
                redc.TIMESTAMP.into(),
                redc.SOCKET.into(),
                redc.CORE.into(),
            ]) {
                df.sort_in_place(&cols[..4], SortMultipleOptions::default())
                    .unwrap();
            } else if cols.starts_with(&[
                redc.GROUP.into(),
                redc.TIMESTAMP.into(),
                redc.SOCKET.into(),
            ]) || cols.starts_with(&[
                redc.GROUP.into(),
                redc.TIMESTAMP.into(),
                redc.UNIT.into(),
            ]) {
                df.sort_in_place(&cols[..3], SortMultipleOptions::default())
                    .unwrap();
            } else if cols.starts_with(&[redc.GROUP.into(), redc.TIMESTAMP.into()]) {
                df.sort_in_place(&cols[..2], SortMultipleOptions::default())
                    .unwrap();
            } else {
                unreachable!();
            }
        }

        CsvWriter::new(file).include_header(true).finish(&mut df)?;

        Ok(())
    }
}
