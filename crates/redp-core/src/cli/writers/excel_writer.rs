#![allow(non_upper_case_globals)]
use crate::redp::core::types::{MetricDefinition, RawEmonDataFrameColumns};
use crate::redp::core::views::{ViewAggregationLevel, ViewData};
use crate::utils::DataframeSeqExt;
use polars::export::arrow::temporal_conversions::*;
use polars::prelude::*;
use rust_xlsxwriter::{Format, Table, TableColumn, Worksheet, XlsxError};

static redc: RawEmonDataFrameColumns = RawEmonDataFrameColumns::new();

pub trait ExcelWrite {
    fn write_view(
        &mut self,
        view: ViewData<'_, ViewAggregationLevel, Vec<MetricDefinition>>,
    ) -> Result<(), XlsxError>;

    #[allow(dead_code)]
    fn write_dataframe(&mut self, df: DataFrame) -> Result<(), XlsxError>;
}

impl ExcelWrite for Worksheet {
    fn write_view(
        &mut self,
        view: ViewData<'_, ViewAggregationLevel, Vec<MetricDefinition>>,
    ) -> Result<(), XlsxError> {
        self.set_tab_color(rust_xlsxwriter::Color::Green);

        let mut df: DataFrame = view.data.unwrap().collect().unwrap();

        let cols: Option<rayon::iter::Either<String, Vec<String>>> =
            match df.get_columns()[0].dtype() {
                &DataType::String => Some(rayon::iter::Either::Left(
                    df.get_column_names()[0].to_string(),
                )),
                _ => None,
            };

        df = df.transpose_seq(Some("metrics + events"), cols).unwrap();

        match self.name().as_str() {
            "__edp_system_view_summary" => {
                self.set_name("system view")?;
                self.set_active(true);
            }
            "__edp_socket_view_summary" => {
                self.set_name("socket view")?;
            }
            "__edp_core_view_summary" => {
                self.set_name("core view")?;
            }
            "__edp_thread_view_summary" => {
                self.set_name("thread view")?;
            }
            _ => {
                if self.name().contains("uncore_view_summary") {
                    let new_name = self
                        .name()
                        .strip_prefix("__edp_")
                        .unwrap_or(&self.name())
                        .strip_suffix("_uncore_view_summary")
                        .expect("The view name is not ended with 'uncore_view_summary'")
                        .to_owned();
                    self.set_name(new_name + " uncore view")?;
                }
            }
        }
        // Create a new Excel file object.
        let mut headers: Vec<String> = Vec::new();
        // Create some formats for the dataframe.
        let datetime_format = Format::new().set_num_format("yyyy\\-mm\\-dd\\ hh:mm:ss");
        let number_format = Format::new().set_num_format("#,##0.0000");

        // Iterate through the dataframe column by column.

        for (col_num, column) in df.get_columns().iter().enumerate() {
            let col_num = col_num as u16;

            // Store the column names for use as table headers.
            match self.name().as_str() {
                "system view" => headers.push(column.name().to_string()),
                "socket view" => {
                    let res = column.name().parse::<char>();
                    if let Ok(socket) = res {
                        headers.push("Socket ".to_string() + socket.to_string().as_str());
                    } else {
                        headers.push(column.name().to_string());
                    }
                }
                "core view" => {
                    let res = column.name().split_once('_');
                    if let Some((socket, core)) = res {
                        if let Some((module, core)) = core.split_once('_') {
                            headers.push(
                                "Socket ".to_string()
                                    + socket
                                    + " Module "
                                    + module
                                    + " Core "
                                    + core,
                            );
                        } else {
                            headers.push("Socket ".to_string() + socket + " Core " + core);
                        }
                    } else {
                        headers.push(column.name().to_string());
                    }
                }
                "thread view" => {
                    let res = column.name().parse::<u32>();
                    if let Ok(core) = res {
                        headers.push("Core ".to_string() + core.to_string().as_str());
                    } else {
                        headers.push(column.name().to_string());
                    }
                }
                _ => {
                    if self.name().contains("uncore view") {
                        let res = column.name().split_once('_');
                        if let Some((socket, unit)) = res {
                            headers.push("Socket ".to_string() + socket + " Unit " + unit);
                        } else {
                            headers.push(column.name().to_string());
                        }
                    }
                }
            }

            // Write the row data for each column/type.
            for (row_num, data) in column.iter().enumerate() {
                let row_num = 1 + row_num as u32;

                // Map the Polars Series AnyValue types to Excel/rust_xlsxwriter
                // types.
                match data {
                    AnyValue::Int8(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::UInt8(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::Int16(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::UInt16(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::Int32(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::UInt32(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::Float32(value) => {
                        if value.is_infinite() || value.is_nan() {
                            self.write_blank(row_num, col_num, &Format::default())?;
                        } else {
                            self.write_number_with_format(row_num, col_num, value, &number_format)?;
                        }
                    }
                    AnyValue::Float64(value) => {
                        if value.is_infinite() || value.is_nan() {
                            self.write_blank(row_num, col_num, &Format::default())?;
                        } else {
                            self.write_number_with_format(row_num, col_num, value, &number_format)?;
                        }
                    }
                    AnyValue::String(value) => {
                        self.write_string(row_num, col_num, value)?;
                    }
                    AnyValue::Boolean(value) => {
                        self.write_boolean(row_num, col_num, value)?;
                    }
                    AnyValue::Null => {
                        // Treat Null as blank for now.
                    }
                    AnyValue::Datetime(value, time_units, _) => {
                        let datetime = match time_units {
                            TimeUnit::Nanoseconds => timestamp_ns_to_datetime(value),
                            TimeUnit::Microseconds => timestamp_us_to_datetime(value),
                            TimeUnit::Milliseconds => timestamp_ms_to_datetime(value),
                        };
                        self.write_datetime_with_format(
                            row_num,
                            col_num,
                            datetime,
                            &datetime_format,
                        )?;
                        self.set_column_width(col_num, 18)?;
                    }
                    _ => {
                        println!(
                            "WARNING: AnyValue data type '{}' is not supported",
                            data.dtype()
                        );
                        break;
                    }
                }
            }
        }

        // Create a table for the dataframe range.
        let (max_row, max_col) = df.shape();
        let mut table = Table::new();
        let columns: Vec<TableColumn> = headers
            .into_iter()
            .map(|x| TableColumn::new().set_header(x))
            .collect();
        table = table.set_columns(&columns);

        // Add the table to the worksheet.
        self.add_table(0, 0, max_row as u32, max_col as u16 - 1, &table)?;

        // Autofit the columns.
        self.autofit();

        Ok(())
    }

    fn write_dataframe(&mut self, df: DataFrame) -> Result<(), XlsxError> {
        self.set_tab_color(rust_xlsxwriter::Color::Brown);
        let mut df = df;
        let cols = df.get_column_names_owned();

        if cols.starts_with(&[
            redc.GROUP.into(),
            redc.TIMESTAMP.into(),
            redc.SOCKET.into(),
            redc.CORE.into(),
        ]) {
            df = df
                .lazy()
                .group_by_stable([polars::lazy::dsl::cols(&cols[2..4])])
                .agg([polars::lazy::dsl::cols(&cols[4..])])
                .collect()
                .unwrap();
            df.sort_in_place([redc.SOCKET, redc.CORE], SortMultipleOptions::default())
                .unwrap();
        } else if cols.starts_with(&[redc.GROUP.into(), redc.TIMESTAMP.into(), redc.SOCKET.into()])
        {
            df = df
                .lazy()
                .group_by_stable([polars::lazy::dsl::cols(&cols[2..3])])
                .agg([polars::lazy::dsl::cols(&cols[3..])])
                .collect()
                .unwrap();
            df.sort_in_place([redc.SOCKET], SortMultipleOptions::default())
                .unwrap();
        } else if cols.starts_with(&[redc.GROUP.into(), redc.TIMESTAMP.into(), redc.UNIT.into()]) {
            df = df
                .lazy()
                .group_by_stable([polars::lazy::dsl::cols(&cols[2..3])])
                .agg([polars::lazy::dsl::cols(&cols[3..])])
                .collect()
                .unwrap();
            df.sort_in_place([redc.UNIT], SortMultipleOptions::default())
                .unwrap();
        } else if cols.starts_with(&[redc.GROUP.into(), redc.TIMESTAMP.into()]) {
            df.sort_in_place(&cols[..2], SortMultipleOptions::default())
                .unwrap();
            df = df.select(&cols[1..]).unwrap();
        } else {
            unreachable!();
        }

        let mut headers: Vec<String> = Vec::new();
        let datetime_format = Format::new().set_num_format("yyyy\\-mm\\-dd\\ hh:mm:ss");
        let number_format = Format::new().set_num_format("#,##0.0000");

        for (col_num, column) in df.get_columns().iter().enumerate() {
            let col_num = col_num as u16;

            // Store the column names for use as table headers.
            headers.push(column.name().to_string());

            // Write the row data for each column/type.
            for (row_num, data) in column.iter().enumerate() {
                let row_num = 1 + row_num as u32;

                // Map the Polars Series AnyValue types to Excel/rust_xlsxwriter
                // types.
                match data {
                    AnyValue::Int8(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::UInt8(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::Int16(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::UInt16(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::Int32(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::UInt32(value) => {
                        self.write_number(row_num, col_num, value)?;
                    }
                    AnyValue::Float32(value) => {
                        if value.is_infinite() || value.is_nan() {
                            self.write_blank(row_num, col_num, &Format::default())?;
                        } else {
                            self.write_number_with_format(row_num, col_num, value, &number_format)?;
                        }
                    }
                    AnyValue::Float64(value) => {
                        if value.is_infinite() || value.is_nan() {
                            self.write_blank(row_num, col_num, &Format::default())?;
                        } else {
                            self.write_number_with_format(row_num, col_num, value, &number_format)?;
                        }
                    }
                    AnyValue::String(value) => {
                        self.write_string(row_num, col_num, value)?;
                    }
                    AnyValue::Boolean(value) => {
                        self.write_boolean(row_num, col_num, value)?;
                    }
                    AnyValue::Null => {
                        // Treat Null as blank for now.
                    }
                    AnyValue::Datetime(value, time_units, _) => {
                        let datetime = match time_units {
                            TimeUnit::Nanoseconds => timestamp_ns_to_datetime(value),
                            TimeUnit::Microseconds => timestamp_us_to_datetime(value),
                            TimeUnit::Milliseconds => timestamp_ms_to_datetime(value),
                        };
                        self.write_datetime_with_format(
                            row_num,
                            col_num,
                            datetime,
                            &datetime_format,
                        )?;
                        self.set_column_width(col_num, 18)?;
                    }
                    AnyValue::List(value) => {
                        let numbers: Vec<String> = value
                            .f32()
                            .unwrap()
                            .to_vec()
                            .into_iter()
                            .map(|x| {
                                if let Some(v) = x {
                                    v.to_string()
                                } else {
                                    "null".to_owned()
                                }
                            })
                            .collect();
                        self.write_string(row_num, col_num, format!("{:?}", numbers))?;
                    }
                    _ => {
                        println!(
                            "WARNING: AnyValue data type '{}' is not supported",
                            data.dtype()
                        );
                        break;
                    }
                }
            }
        }

        self.write_row(0, 0, headers)?;

        // Autofit the columns.
        self.autofit();

        Ok(())
    }
}
