#![allow(dead_code)]
#![allow(non_snake_case)]

use super::metric_computer::MetricCompute;
use polars::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct DeviceType<'a> {
    CORE: &'a str,
    UNCORE: &'a str,
    SYSTEM: &'a str,
}

impl DeviceType<'static> {
    pub const fn new() -> Self {
        let CORE = "core";
        let UNCORE = "UNCORE";
        let SYSTEM = "SYSTEM";
        Self {
            CORE,
            UNCORE,
            SYSTEM,
        }
    }
}

impl Default for DeviceType<'static> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Device<'a, T: Clone, S: MetricCompute + Clone> {
    // Device handling class all supported devices ('core', 'bigcore', 'smallcore', 'cha', etc.)
    // note: label is blank for non-hybrid cores (ex: 'core') because no core type specifier is
    //       added to report and filenames for non-hybrid cores.
    // @param type_name: gets the type of device, used for filtering an emon df
    pub type_name: &'a str,
    label: Option<String>,
    pub aggregation_levels: Option<Vec<T>>,
    pub metric_computer: Option<S>,
    valid_device_names: Vec<&'a str>,
    pub exclusions: Vec<&'a str>,
}

impl<'a, T: Clone, S: MetricCompute + Clone> Device<'a, T, S> {
    pub fn new(
        type_name: &'a str,
        unique_devices: Option<Vec<&'a str>>,
        aggregation_levels: Option<Vec<T>>,
        metric_computer: Option<S>,
    ) -> Self {
        let valid_device_names: Vec<&'a str> = unique_devices.unwrap_or_default();
        let device_type = DeviceType::new();
        let label: Option<String>;
        if type_name.trim_matches('"') == device_type.CORE {
            label = None;
        } else if type_name.contains("UNC_") {
            label = Some(type_name.replace("UNC_", "").to_lowercase());
        } else {
            label = Some(type_name.to_owned());
        }
        let _valid_device_names = valid_device_names.clone();
        let exclusions = _valid_device_names
            .into_iter()
            .filter(|x| *x != type_name)
            .collect();
        Self {
            type_name,
            label,
            aggregation_levels,
            metric_computer,
            valid_device_names,
            exclusions,
        }
    }

    pub fn decorate_label(&self, prefix: &str, postfix: &str) -> Option<String> {
        // method for prefixing/postfixing the label with specified characters
        // when preparing the label for use in a filename, chart name, stdout statement, etc
        // :param prefix: a string to add to the beginning of the label (ex: ' ', '_'
        // :param postfix: a string to add to the end of the label (ex: ' ', '_')
        // :return: a copy of the decorated label
        self.label
            .as_ref()
            .map(|label| prefix.to_owned() + label.as_str() + postfix)
    }
}

pub type RawEmonDataFrame = LazyFrame;

pub type SummaryViewDataFrame = LazyFrame;

pub type EventInfoDataFrame = LazyFrame;

#[derive(Debug)]
pub struct RawEmonDataFrameColumns<'a> {
    pub TIMESTAMP: &'a str,
    pub SOCKET: &'a str,
    pub DEVICE: &'a str,
    pub CORE: &'a str,
    pub THREAD: &'a str,
    pub UNIT: &'a str,
    pub MODULE: &'a str,
    pub TSC: &'a str,
    pub GROUP: &'a str,
    pub NAME: &'a str,
    pub VALUE: &'a str,
    pub COLUMNS: [&'a str; 11],
}

impl RawEmonDataFrameColumns<'static> {
    pub const fn new() -> Self {
        let TIMESTAMP = "timestamp";
        let SOCKET = "socket";
        let DEVICE = "device";
        let CORE = "core";
        let THREAD = "thread";
        let UNIT = "unit";
        let MODULE = "module";
        let TSC = "tsc";
        let GROUP = "group";
        let NAME = "name";
        let VALUE = "value";
        let COLUMNS = [
            TIMESTAMP, SOCKET, DEVICE, CORE, THREAD, UNIT, MODULE, TSC, GROUP, NAME, VALUE,
        ];
        Self {
            TIMESTAMP,
            SOCKET,
            DEVICE,
            CORE,
            THREAD,
            UNIT,
            MODULE,
            TSC,
            GROUP,
            NAME,
            VALUE,
            COLUMNS,
        }
    }
}

impl Default for RawEmonDataFrameColumns<'static> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct SummaryViewDataFrameColumns<'a> {
    AGGREGATED: &'a str,
    MIN: &'a str,
    MAX: &'a str,
    PERCENTILE: &'a str,
    VARIATION: &'a str,
    COLUMNS: [&'a str; 5],
}

impl SummaryViewDataFrameColumns<'static> {
    pub const fn new() -> Self {
        let AGGREGATED = "aggregated";
        let MIN = "min";
        let MAX = "max";
        let PERCENTILE = "95th percentile";
        let VARIATION = "variation (stdev/avg)";
        let COLUMNS = [AGGREGATED, MIN, MAX, PERCENTILE, VARIATION];
        Self {
            AGGREGATED,
            MIN,
            MAX,
            PERCENTILE,
            VARIATION,
            COLUMNS,
        }
    }
}

impl Default for SummaryViewDataFrameColumns<'static> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct EventInfoDataFrameColumns<'a> {
    pub NAME: &'a str,
    pub DEVICE: &'a str,
}

impl EventInfoDataFrameColumns<'static> {
    pub const fn new() -> Self {
        let NAME = "name";
        let DEVICE = "device";
        Self { NAME, DEVICE }
    }
}

impl Default for EventInfoDataFrameColumns<'static> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MetricDefinition {
    // Metric definition including all its attributes (name, description, formula, etc...).
    // "current" name of the metric, i.e. the EDP metric name
    pub name: String,

    // corresponding metric name for "per transaction" metrics
    pub throughput_metric_name: String,

    // metric description for documentation purposes
    pub description: String,

    // a human readable version of formula. For documentation purposes
    pub human_readable_expression: String,

    // the metric formula
    pub formula: String,

    // maps event aliases (e.g. "a") to their respective event names (e.g. "INST_RETIRED.ANY")
    pub event_aliases: HashMap<String, String>,

    // maps constant aliases (e.g. "a") to their value (e.g. "2", "system.sockets[0][0].size",
    // "$samplingTime", $processed_samples)
    pub constants: HashMap<String, String>,

    // maps retire latency aliases (e.g. "a") to their respective latency names
    pub retire_latencies: HashMap<String, String>,

    // the "standard" name of the metric
    pub canonical_name: Option<String>,
}

impl MetricDefinition {
    pub fn new(
        name: String,
        throughput_metric_name: String,
        description: String,
        human_readable_expression: String,
        formula: String,
        event_aliases: HashMap<String, String>,
        constants: HashMap<String, String>,
        retire_latencies: HashMap<String, String>,
        canonical_name: Option<String>,
    ) -> Self {
        Self {
            name,
            throughput_metric_name,
            description,
            human_readable_expression,
            formula,
            event_aliases,
            constants,
            retire_latencies,
            canonical_name,
        }
    }

    #[inline]
    pub fn all_metric_references_are_available(&self, df: &DataFrame) -> bool {
        for event in self.event_aliases.values() {
            if !df.get_column_names().contains(&event.as_str()) {
                return false;
            }
        }

        for constant in self.constants.values() {
            if constant.parse::<f32>().is_err()
                && !df.get_column_names().contains(&constant.as_str())
            {
                return false;
            }
        }

        for retire_latency in self.retire_latencies.values() {
            if !df.get_column_names().contains(&retire_latency.as_str()) {
                return false;
            }
        }

        true
    }
}

#[derive(Debug)]
pub struct StatisticsDataFrameColumns<'a> {
    pub MIN: &'a str,
    pub MAX: &'a str,
    pub COUNT: &'a str,
    pub SUM: &'a str,
    pub PERCENTILE: &'a str,
    pub VARIATION: &'a str,
    pub COLUMNS: [&'a str; 6],
}

impl StatisticsDataFrameColumns<'static> {
    pub const fn new() -> Self {
        let MIN = "min";
        let MAX = "max";
        let COUNT = "count";
        let SUM = "sum";
        let PERCENTILE = "95th percentile";
        let VARIATION = "variation (stdev/avg)";
        let COLUMNS = [MIN, MAX, PERCENTILE, COUNT, SUM, VARIATION];
        Self {
            MIN,
            MAX,
            COUNT,
            SUM,
            PERCENTILE,
            VARIATION,
            COLUMNS,
        }
    }
}

impl Default for StatisticsDataFrameColumns<'static> {
    fn default() -> Self {
        Self::new()
    }
}
