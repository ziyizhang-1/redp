#![allow(non_upper_case_globals)]

use super::chunk_reader::Chunk;
use super::emon_system_information::EmonSystemInformationParser;
use crate::redp::core::types::MetricDefinition;
use crate::redp::core::types::{EventInfoDataFrame, RawEmonDataFrame, RawEmonDataFrameColumns};
use crate::redp::core::views::{DataAccumulator, ViewAggregationLevel, ViewGenerator};
use crate::redp::core::SliceExt;
use crate::utils::DataframeSeqExt;
use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};
use lazy_static::lazy_static;
use ndarray::{Array, Array1, ArrayBase};
use polars::prelude::*;
use rayon::iter::plumbing::Producer;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use regex::bytes::Regex;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::marker::PhantomData;
use std::path::Path;
use std::str::from_utf8_unchecked;
use std::sync::Mutex;

lazy_static! {
    static ref UNCORE_UNIT_RE: Regex = Regex::new(r"UNC_(.*?)_").unwrap();
    static ref FREERUN_RE: Regex = Regex::new(r"FREERUN_(.*?)_").unwrap();
    static ref FREERUN_SCOPED_RE: Regex = Regex::new(r"FREERUN:.*scope=(.*)").unwrap();
    static ref DATE_PATTERN_RE: Regex =
        Regex::new(r"(\d{2})/(\d{2})/(\d{4})\s(\d{2}):(\d{2}):(\d{2}).(\d{3})").unwrap();
}

static redc: RawEmonDataFrameColumns = RawEmonDataFrameColumns::new();
static SAMPLE_SEPARATOR: &[u8; 10] = b"----------";
static BLOCK_SEPARATOR: &[u8; 10] = b"==========";
static EMON_DATE_FORMAT: &[u8; 20] = b"%m/%d/%Y %H:%M:%S.%f";

unsafe fn get_event_device(event_name: &[u8]) -> String {
    /*!
    * Return device name associated with an event
       :param event_name: the event
       :return: the device name associated with the event
    */
    let device_core = b"core";
    let device_uncore = b"UNCORE";
    let device_package = b"PACKAGE";

    let mut captures = UNCORE_UNIT_RE.captures(event_name);
    match captures {
        Some(c) => "UNC_".to_owned() + from_utf8_unchecked(c.get(1).unwrap().as_bytes()),
        None => {
            captures = FREERUN_RE.captures(event_name);
            match captures {
                Some(c) => from_utf8_unchecked(c.get(1).unwrap().as_bytes()).to_owned(),
                None => {
                    captures = FREERUN_SCOPED_RE.captures(event_name);
                    if let Some(c) = captures {
                        let scope = c.get(1).unwrap().as_bytes();
                        if scope == device_package {
                            return from_utf8_unchecked(scope).to_owned();
                        }
                    }
                    if event_name.to_ascii_uppercase().starts_with(b"UNC_") {
                        return from_utf8_unchecked(device_uncore).to_owned();
                    }
                    from_utf8_unchecked(device_core).to_owned()
                }
            }
        }
    }
}

trait DatetimeExt {
    #[allow(dead_code)]
    fn to_string(&self) -> Option<String>;

    fn to_timestamp(&self) -> Option<i64>;
}

impl<T: TimeZone> DatetimeExt for Option<Datetime<T>> {
    fn to_string(&self) -> Option<String> {
        match self {
            Some(ref t) => match t {
                Datetime::DateTime(ref v) => Some(v.with_timezone(&Local).to_string()),
                Datetime::NaiveDateTime(ref v) => Some(v.to_string()),
            },
            None => None,
        }
    }

    fn to_timestamp(&self) -> Option<i64> {
        match self {
            Some(ref t) => match t {
                Datetime::DateTime(ref v) => v.with_timezone(&Local).timestamp_nanos_opt(),
                Datetime::NaiveDateTime(ref v) => v.and_utc().timestamp_nanos_opt(),
            },
            None => None,
        }
    }
}

trait SliceExt2 {
    fn is_separator(&self) -> bool;
    fn is_block_separator(&self) -> bool;
    fn is_timestamp(&self) -> bool;
    unsafe fn convert_to_datetime<T: TimeZone>(&self, timezone: Option<&T>) -> Datetime<T>;
}

impl SliceExt2 for [u8] {
    fn is_separator(&self) -> bool {
        self == SAMPLE_SEPARATOR || self == BLOCK_SEPARATOR
    }

    fn is_block_separator(&self) -> bool {
        self == BLOCK_SEPARATOR
    }

    fn is_timestamp(&self) -> bool {
        DATE_PATTERN_RE.captures(self).is_some()
    }

    unsafe fn convert_to_datetime<T: TimeZone>(&self, timezone: Option<&T>) -> Datetime<T> {
        match timezone {
            Some(timezone) => {
                let _s = from_utf8_unchecked(self);
                let fmt = from_utf8_unchecked(EMON_DATE_FORMAT);
                let native_datetime = NaiveDateTime::parse_from_str(_s, fmt).unwrap();
                Datetime::DateTime(timezone.from_local_datetime(&native_datetime).unwrap())
            }
            None => {
                let _s = from_utf8_unchecked(self);
                let fmt = from_utf8_unchecked(EMON_DATE_FORMAT);
                Datetime::NaiveDateTime(NaiveDateTime::parse_from_str(_s, fmt).unwrap())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Datetime<T: TimeZone> {
    DateTime(DateTime<T>),
    NaiveDateTime(NaiveDateTime),
}

impl<T: TimeZone> PartialEq for Datetime<T> {
    fn eq(&self, other: &Self) -> bool {
        let t1 = match self {
            Datetime::DateTime(t) => t.timestamp(),
            Datetime::NaiveDateTime(t) => t.and_utc().timestamp(),
        };
        let t2 = match other {
            Datetime::DateTime(t) => t.timestamp(),
            Datetime::NaiveDateTime(t) => t.and_utc().timestamp(),
        };
        t1 == t2
    }
}

impl<T: TimeZone> PartialOrd for Datetime<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let t1 = match self {
            Datetime::DateTime(t) => t.timestamp(),
            Datetime::NaiveDateTime(t) => t.and_utc().timestamp(),
        };
        let t2 = match other {
            Datetime::DateTime(t) => t.timestamp(),
            Datetime::NaiveDateTime(t) => t.and_utc().timestamp(),
        };
        Some(t1.cmp(&t2))
    }
}

pub struct EmonParser<T: TimeZone> {
    /* Parse EMON data file (emon.dat) */
    pub system_info: EmonSystemInformationParser,
    pub event_info: Option<EventInfoDataFrame>,
    timezone: Option<T>,
}

impl<'a, T: TimeZone + Send + Copy> EmonParser<T> {
    pub fn new(
        input_file: &'a Path,
        emon_v_file: Option<&'a Path>,
        timezone: Option<T>,
        ref_tsc_hz: f32,
    ) -> Self {
        let _input_file: &Path;

        if let Some(path) = emon_v_file {
            _input_file = path;
        } else {
            _input_file = input_file;
        }

        Self {
            system_info: EmonSystemInformationParser::new(_input_file, Some(ref_tsc_hz)),
            event_info: None,
            timezone,
        }
    }

    pub fn event_reader(
        &mut self,
        views: &'a ViewGenerator<ViewAggregationLevel, Vec<MetricDefinition>>,
        data_accumulator: *mut DataAccumulator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
        details: *const Mutex<*mut HashMap<String, DataFrame>>,
        from_timestamp: Option<Datetime<T>>,
        to_timestamp: Option<Datetime<T>>,
        from_sample: Option<u32>,
        to_sample: Option<u32>,
        chunks: Vec<Chunk<'a, u8>>,
        no_details: bool,
    ) -> ChunkIterator<'a, EventContentHandler<'a, T>, T> {
        /*!
            Parse EMON data and return a generator for EMON values.
            :param from_timestamp: include only samples with timestamp equal to or greater than the specified value.
            :param to_timestamp: include only samples with timestamps equal to or less than the specified value.
            :param from_sample: include only samples equal to or greater than the specified sample number
                                (first sample is 1).
            :param to_sample: include only samples equal to or less than the specified sample number.
            :param partition: include only samples from the specified partition. Cannot be combined with any of the
                                `from` and `to` arguments.
                                Use the `EmonParser.partition` function to generate partition objects.
            :param chunk_size: the maximum number of EMON blocks to include in each chunk returned.
                                An EMON block represents all event values collected in a single iteration of all
                                EMON event groups.
                                Setting this parameter to a high value may cause an out of memory error.
                                Setting this parameter to 0 will read the entire file into memory and may cause an
                                out of memory error.
            :return: a generator for EMON event values, represented as a pandas dataframe with the following structure:
                        rows: a single event value
                        columns:
                        TimeStamp: sample timestamp
                        socket: socket/package number (0 based)
                        device: event type/source, e.g. CORE, UNCORE, ...
                        core: core number (0 based)
                        thread: hyper-thread number within the core (0 based)
                        unit: device instance number, e.g. logical core number (0 based)
                        group: the EMON group number in which the event was collected (0 based)
                        name: event name
                        value: event value
        */
        assert_ne!(
            self.system_info.ref_tsc,
            0.0,
            "Unable to determine system frequency.\nPlease provide an EMON system information file (emon-v.dat) or\nspecify a value for the system frequency"
        );

        assert!(
            !self.system_info.socket_map.is_empty(),
            "Unable to determine processor mapping.\nPlease make sure the information is included in the EMON data file,\nor provide an EMON processor mapping file (emon-m.dat)"
        );

        let content_handler = EventContentHandler::new(
            self.system_info.clone(),
            self.timezone,
            views,
            data_accumulator,
            details,
            no_details,
        );
        // Need to set the parser's __very_first_sample to very first sample processed in first partition/chunk
        ChunkIterator::new(
            chunks,
            content_handler,
            from_timestamp,
            to_timestamp,
            from_sample,
            to_sample,
            self.timezone,
        )
    }
}

/// A content handler for splitting EMON event samples into chunks.
/// Chunks are returned as `RawEmonDataFrame` objects.
/// See the `_ContentHandler` class for additional information.
#[derive(Clone)]
pub struct EventContentHandler<'a, T: TimeZone + Send> {
    samples: Vec<Sample<T>>,
    event_buffer: Vec<Vec<u8>>,
    system_info: EmonSystemInformationParser,
    timezone: Option<T>,
    views: &'a ViewGenerator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
    data_accumulator: *mut DataAccumulator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
    details: *const Mutex<*mut HashMap<String, DataFrame>>,
    no_details: bool,
}

unsafe impl<'a, T: TimeZone + Send> Send for EventContentHandler<'a, T> {}

impl<'a, T: TimeZone + Send> EventContentHandler<'a, T> {
    /// Receive notification of the beginning of the EMON performance data
    /// (before reading the first event data in the EMON file).
    /// The parser will invoke this method once, before any other methods in this interface
    fn new(
        system_info: EmonSystemInformationParser,
        timezone: Option<T>,
        views: &'a ViewGenerator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
        data_accumulator: *mut DataAccumulator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
        details: *const Mutex<*mut HashMap<String, DataFrame>>,
        no_details: bool,
    ) -> Self {
        Self {
            samples: vec![],
            event_buffer: vec![],
            system_info,
            timezone,
            views,
            data_accumulator,
            details,
            no_details,
        }
    }
}

impl<'a, T: TimeZone + Send + Copy> ContentHandler for EventContentHandler<'a, T> {
    type Output = Block<'a, T>;

    fn end_sample(&mut self, _sample_number: u32, block_number: u32) {
        if !self.event_buffer.is_empty() {
            self.samples.push(Sample::new(
                self.system_info.ref_tsc,
                &self.event_buffer,
                block_number,
                self.timezone,
            ));
            self.event_buffer.clear();
        }
    }

    fn end_block(&mut self, _block_number: u32) {}

    fn end_event(&mut self, data: &[u8]) {
        self.event_buffer.push(data.to_owned());
    }

    fn end_file(&self) {}

    fn get_chunk_data(&mut self) -> Option<Self::Output> {
        /*!
         *      Returns a pandas DataFrame containing a row for every event, sample, and value within the raw EMON file.
         *      @param samples: List of Sample objects, containing each individual line from the raw EMON capture.
         *      @return DataFrame: Contains a row for every event, sample, and value within the raw EMON file.
         */
        if self.samples.is_empty() {
            return None;
        }
        let block = Block::new(
            self.samples.clone(),
            self.system_info.clone(),
            self.views,
            self.data_accumulator,
            self.details,
            self.no_details,
        );

        self.samples.clear();
        Some(block)
    }
}

#[derive(Debug, Clone, Copy)]
enum FilterMode {
    TimeStamp,
    Sample,
}

/// Store a line from the EMON file, extracting the relevant information as class attributes.
/// Extract attributes from a line in the EMON raw file.  Each line represents the data about a given event for
/// the sample duration.  Each line may have a breakdown of the event counts of the system across
/// sockets/cores/threads/channels/etc.
#[derive(Debug, Clone)]
struct Line {
    tsc_count: f32,
    name: String,
    device: String,
    values: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>,
}

impl Line {
    /// EMON lines are tab separated
    /// Captures the key attributes of an EMON event from a line in the EMON collection file.
    /// @param line: raw line from the EMON data file
    fn new(line: &[u8]) -> Option<Self> {
        let mut line_values: Vec<&str> = unsafe {
            line.split(|x| x.eq(&b'\t'))
                .map(|x| from_utf8_unchecked(x))
                .collect()
        };
        let name = line_values.remove(0);
        let device = unsafe { get_event_device(name.as_bytes()) };
        let tsc_count: f32 = if line_values.is_empty() {
            return None;
        } else {
            line_values.remove(0).replace(',', "").parse().unwrap()
        };
        let values: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>> = Array::from_vec(
            line_values
                .into_iter()
                .map(|x| x.replace(',', "").parse().unwrap_or(f32::NAN))
                .collect::<Vec<f32>>(),
        );
        Some(Self {
            tsc_count,
            name: name.to_owned(),
            device,
            values,
        })
    }
}

/// Store a sample from the EMON file, extracting the relevant information as class attributes.
/// Extract attributes from a sample in the EMON raw file.  Each line with the same timestamp represents a sample.
/// Within a sample there are various events collected.  These events are listed on their own lines and are stored
/// in a Line/Event object.
/// Each sample starts with a date line.
#[derive(Clone)]
struct Sample<T: TimeZone + Send> {
    events: HashMap<String, Line>,
    block_number: u32,
    tsc_count: f32,
    duration: f32,
    time_stamp: Option<Datetime<T>>,
}

unsafe impl<T: TimeZone + Send> Send for Sample<T> {}

impl<T: TimeZone + Send> Sample<T> {
    fn new(ref_tsc: f32, data: &Vec<Vec<u8>>, block_number: u32, timezone: Option<T>) -> Self {
        let normalized_topdown_events = [
            "PERF_METRICS.RETIRING",
            "PERF_METRICS.BAD_SPECULATION",
            "PERF_METRICS.FRONTEND_BOUND",
            "PERF_METRICS.BACKEND_BOUND",
        ];
        let mut topdown_events: HashSet<String> = HashSet::new();
        let mut topdown_events_original_values: HashMap<
            String,
            ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>,
        > = HashMap::new();
        let mut time_stamp: Option<Datetime<T>> = None;
        let mut events: HashMap<String, Line> = HashMap::new();
        let tsc_count: f32;
        let duration: f32;
        // If line starts with date format, we are at the beginning of the block
        for line in data {
            match DATE_PATTERN_RE.captures(line.as_slice()) {
                Some(_v) => unsafe {
                    time_stamp = Some(line.convert_to_datetime(timezone.as_ref()));
                },
                None => {
                    let event_wrapper = Line::new(line.as_slice());
                    if let Some(event) = event_wrapper {
                        let name = event.name.clone();
                        let values = event.values.clone();
                        let _name = event.name.clone();
                        events.insert(name, event);
                        if _name.to_lowercase().starts_with("perf_metrics")
                            || _name.starts_with("TOPDOWN.SLOTS")
                        {
                            topdown_events.insert(_name.clone());
                            topdown_events_original_values.insert(_name, values);
                        }
                    }
                }
            }
        }

        if !events.is_empty() {
            // If stats are empty, retrieve tsc_count for this block (same for all lines in a block)
            // Calculate the duration from the tsc_count
            tsc_count = events.values().next().unwrap().tsc_count;
            duration = tsc_count / ref_tsc;
        } else {
            tsc_count = 0.0;
            duration = 0.0;
        }

        // Adjust the values of the PERF_METRICS.* (TMA) events
        // TODO: refactor, clean-up
        // let _topdown_events_keys = topdown_events.clone().into_keys();
        let perf_metrics: Vec<String> = topdown_events
            .into_iter()
            .filter(|key| key.to_lowercase().starts_with("perf_metrics"))
            .collect();

        let normalized_events: Vec<String> = perf_metrics
            .clone()
            .into_iter()
            .filter(|x| normalized_topdown_events.contains(&x.as_str()))
            .collect();

        perf_metrics.into_iter().for_each(|x| {
            let mut divider = Array1::<f32>::zeros(
                topdown_events_original_values
                    .values()
                    .next()
                    .unwrap()
                    .shape()[0],
            );
            for event in normalized_events.iter() {
                divider += topdown_events_original_values.get(event).unwrap();
            }
            events.get_mut(&x).unwrap().values = (topdown_events_original_values.get(&x).unwrap()
                / divider)
                * topdown_events_original_values
                    .get("TOPDOWN.SLOTS:perf_metrics")
                    .unwrap();
        });

        Self {
            events,
            block_number,
            tsc_count,
            duration,
            time_stamp,
        }
    }
}

pub struct Block<'a, T: TimeZone + Send> {
    samples: Vec<Sample<T>>,
    system_info: EmonSystemInformationParser,
    view_generator: &'a ViewGenerator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
    data_accumulator: *mut DataAccumulator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
    details: *const Mutex<*mut HashMap<String, DataFrame>>,
    no_details: bool,
}

unsafe impl<'a, T: TimeZone + Send> Send for Block<'a, T> {}

impl<'a, T: TimeZone + Send> Block<'a, T> {
    fn new(
        samples: Vec<Sample<T>>,
        system_info: EmonSystemInformationParser,
        view_generator: &'a ViewGenerator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
        data_accumulator: *mut DataAccumulator<'a, ViewAggregationLevel, Vec<MetricDefinition>>,
        details: *const Mutex<*mut HashMap<String, DataFrame>>,
        no_details: bool,
    ) -> Self {
        Self {
            samples,
            system_info,
            view_generator,
            data_accumulator,
            details,
            no_details,
        }
    }

    pub fn update(self) {
        let mut dataframe_builder = EventDataFrameBuilder::new(&self.system_info);

        self.samples
            .iter()
            .filter(|x| !x.events.is_empty())
            .for_each(|sample| {
                sample
                    .events
                    .values()
                    .for_each(|event| dataframe_builder.append_event_values(sample, event));
                dataframe_builder.append_pseudo_event_values(sample);
            });

        let df = dataframe_builder.into_dataframe();
        let event_aggregates = self.view_generator.compute_aggregates(Some(df.clone()));
        let detail_views = self
            .view_generator
            .generate_detail_views(Some(df), self.no_details);

        unsafe {
            (*self.data_accumulator).update_statistics(Some(&detail_views));
            (*self.data_accumulator).update_aggregates(&event_aggregates);
        }

        if !self.no_details {
            let mut detail_dfs: HashMap<String, DataFrame> = HashMap::new();
            detail_views.into_iter().for_each(|(key, value)| {
                if let Some(lf) = value.data {
                    detail_dfs.insert(key, lf.collect().unwrap());
                }
            });
            unsafe {
                let details = (*self.details).lock().unwrap().as_mut().unwrap();
                if details.is_empty() {
                    *details = detail_dfs;
                } else {
                    details.iter_mut().for_each(|(name, df)| {
                        let mut _df = detail_dfs.remove(name).unwrap();
                        if df.width() > _df.width() {
                            let schema = df.schema();
                            let _schema = _df.schema();
                            let diff: Vec<&str> = schema
                                .get_names()
                                .into_iter()
                                .filter(|s| !_schema.get_names().contains(s))
                                .collect();
                            diff.into_iter().for_each(|x| {
                                _df.with_column_unchecked(Series::full_null(
                                    x,
                                    _df.height(),
                                    &DataType::Float32,
                                ));
                            });
                            _df = _df.select(schema.get_names()).unwrap();
                        }

                        if df.width() < _df.width() {
                            let schema = df.schema();
                            let _schema = _df.schema();
                            let diff: Vec<&str> = _schema
                                .get_names()
                                .into_iter()
                                .filter(|s| !schema.get_names().contains(s))
                                .collect();
                            diff.into_iter().for_each(|x| {
                                df.with_column_unchecked(Series::full_null(
                                    x,
                                    df.height(),
                                    &DataType::Float32,
                                ));
                            });
                            *df = df.select(_schema.get_names()).unwrap();
                        }

                        df.vstack_mut(&_df).unwrap().align_chunks();
                    });
                }
            }
        }
    }
}

/// Utility class to assist in tracking and filtering EMON samples
#[derive(Clone)]
struct SampleTracker<T: TimeZone> {
    current_sample_number: u32,
    current_sample_timestamp: Datetime<T>,
    first_sample_number_processed: u32,
    is_first_processed_sample_updated: bool,
    mode: FilterMode,
    from_timestamp: Option<Datetime<T>>,
    to_timestamp: Option<Datetime<T>>,
    from_sample: Option<u32>,
    to_sample: Option<u32>,
}

impl<T: TimeZone> SampleTracker<T> {
    fn new(
        _from_sample: Option<u32>,
        _to_sample: Option<u32>,
        _from_timestamp: Option<Datetime<T>>,
        _to_timestamp: Option<Datetime<T>>,
    ) -> Self {
        let mode: FilterMode;
        let from_timestamp: Option<Datetime<T>>;
        let to_timestamp: Option<Datetime<T>>;
        let from_sample: Option<u32>;
        let to_sample: Option<u32>;
        let current_sample_timestamp = Datetime::<T>::NaiveDateTime(NaiveDateTime::new(
            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
            NaiveTime::MIN,
        ));
        if _from_timestamp.is_some() || _to_timestamp.is_some() {
            mode = FilterMode::TimeStamp;
            from_sample = None;
            to_sample = None;
            from_timestamp = match _from_timestamp {
                Some(t) => Some(t),
                None => {
                    let d = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                    let t = NaiveTime::MIN;
                    Some(Datetime::NaiveDateTime(NaiveDateTime::new(d, t)))
                }
            };
            to_timestamp = match _to_timestamp {
                Some(t) => Some(t),
                None => {
                    let d = NaiveDate::MAX;
                    let t = NaiveTime::MIN;
                    Some(Datetime::NaiveDateTime(NaiveDateTime::new(d, t)))
                }
            };
        } else {
            mode = FilterMode::Sample;
            from_timestamp = None;
            to_timestamp = None;
            from_sample = _from_sample.or(Some(1));
            to_sample = _to_sample.or(Some(u32::MAX));
        }
        Self {
            current_sample_number: 0,
            current_sample_timestamp,
            first_sample_number_processed: 0,
            is_first_processed_sample_updated: false,
            mode,
            from_timestamp,
            to_timestamp,
            from_sample,
            to_sample,
        }
    }

    fn process(&mut self, sample_timestamp: Datetime<T>) {
        self.current_sample_number += 1;
        self.current_sample_timestamp = sample_timestamp;
        if !self.is_first_processed_sample_updated && self.is_current_sample_in_range() {
            self.first_sample_number_processed = self.current_sample_number;
            self.is_first_processed_sample_updated = true;
        }
    }

    fn last_sample_number_processed(&self) -> u32 {
        if self.is_current_sample_greater_than_range_max() {
            self.current_sample_number - 1
        } else {
            self.current_sample_number
        }
    }

    fn is_current_sample_in_range(&self) -> bool {
        match self.mode {
            FilterMode::Sample => {
                self.from_sample.unwrap() <= self.current_sample_number
                    && self.current_sample_number <= self.to_sample.unwrap()
            }
            FilterMode::TimeStamp => {
                *self.from_timestamp.as_ref().unwrap() <= self.current_sample_timestamp
                    && self.current_sample_timestamp <= *self.to_timestamp.as_ref().unwrap()
            }
        }
    }

    fn is_current_sample_greater_than_range_max(&self) -> bool {
        match self.mode {
            FilterMode::Sample => self.current_sample_number > self.to_sample.unwrap(),
            FilterMode::TimeStamp => {
                self.current_sample_timestamp > *self.to_timestamp.as_ref().unwrap()
            }
        }
    }
}

/// Callback interface for the EMON performance data parser.
/// The order of events in this interface mirrors the order of the information in the EMON file
pub trait ContentHandler {
    /// The output type can either be Partition or RawEmonDataFrame
    type Output: Send;

    /// Receive notification of the end of the EMON performance data
    /// (after reading the last line in the EMON data file).
    /// The parser will invoke this method once, and it will be the last method invoked during the parse.
    fn end_file(&self);

    /// Signals the end of an EMON sample.
    /// The parser will invoke this method each time it encounters the EMON sample separator (e.g. '----------').
    /// :param sample_number: the sample number. First sample is 1.
    /// :param block_number: the block number. First block is 1.
    fn end_sample(&mut self, sample_number: u32, block_number: u32);

    /// Signals the end of an EMON block.
    /// The parser will invoke this method each time it encounters the EMON block separator (e.g. '==========').
    /// :param block_number: the block number. First block is 1.
    fn end_block(&mut self, block_number: u32);

    /// Signals the end of a single EMON event data.
    /// The parser will invoke this method each time it encounters an EMON event data line.
    /// @param data: event data.
    fn end_event(&mut self, data: &[u8]);

    /// Signals the end of a chunk, which contains 1 or more blocks.
    /// The parser will invoke this method after parsing a number of EMON blocks that is less or equal to the
    /// specified chunk size. The content handler is expected to return the chunk data.
    /// :return: chunk data or None if there's no data to return.
    /// Return type depends on the implementation. Content handlers decide which data to return.
    fn get_chunk_data(&mut self) -> Option<Self::Output>;
}

/// Iterator that produces chunks of EMON events data. Chunks are aligned to EMON block boundaries.
pub struct ChunkIterator<'a, T: ContentHandler, U: TimeZone> {
    chunks: std::vec::IntoIter<Chunk<'a, u8>>,
    handler: T,
    sample_tracker: SampleTracker<U>,
    timezone: Option<U>,
}

impl<'a, T: ContentHandler, U: TimeZone> ChunkIterator<'a, T, U> {
    /// Initialize the chunk iterator
    /// :param emon_parser: the EMON parser object that owns this chunk iterator.
    /// :param file: an open file handle of the EMON data file to parse.
    /// :param handler: a content handler object that implements the EMON performance data parser interface.
    /// See `EmonParser.event_reader` for the description of all other arguments.
    fn new(
        chunks: Vec<Chunk<'a, u8>>,
        handler: T,
        from_timestamp: Option<Datetime<U>>,
        to_timestamp: Option<Datetime<U>>,
        from_sample: Option<u32>,
        to_sample: Option<u32>,
        timezone: Option<U>,
    ) -> Self {
        if let Some(v) = from_sample {
            assert_ne!(v, 0, "The 'from_sample' must greater than 0");

            assert!(
                from_timestamp.is_none(),
                "The 'from_sample' and 'from_timestamp' arguments are mutex"
            );

            assert!(
                to_timestamp.is_none(),
                "Cannot use both sample numbers and timestamps to specify a sample range"
            );

            if let Some(w) = to_sample {
                assert!(
                    v <= w,
                    "The specified 'from_sample' value must be less than or equal to the specified 'to_sample' value"
                );
            }
        }
        if let Some(v) = to_sample {
            assert_ne!(v, 0, "The 'to_sample' must greater than 0");

            assert!(
                to_timestamp.is_none(),
                "The 'to_sample' and 'to_timestamp' arguments are mutex"
            );

            assert!(
                from_timestamp.is_none(),
                "The 'to_sample' and 'from_timestamp' arguments are mutex"
            );
        }
        if let Some(ref v) = from_timestamp {
            if let Some(ref w) = to_timestamp {
                match v {
                    Datetime::DateTime(from_datetime) => match w {
                        Datetime::DateTime(to_datetime) => {
                            assert!(
                                    from_datetime <= to_datetime,
                                    "The specified 'from_timestamp' value must be less than or equal to the specified 'to_timestamp' value"
                                );
                        }
                        _ => unreachable!(),
                    },
                    Datetime::NaiveDateTime(from_datetime) => match w {
                        Datetime::NaiveDateTime(to_datetime) => {
                            assert!(
                                    from_datetime <= to_datetime,
                                    "The specified 'from_timestamp' value must be less than or equal to the specified 'to_timestamp' value"
                                );
                        }
                        _ => unreachable!(),
                    },
                }
            }
        }

        let chunks: std::vec::IntoIter<Chunk<u8>> = chunks.into_iter();

        Self {
            chunks,
            handler,
            sample_tracker: SampleTracker::new(
                from_sample,
                to_sample,
                from_timestamp,
                to_timestamp,
            ),
            timezone,
        }
    }
}

unsafe impl<'a, T: ContentHandler + Clone + Send, U: TimeZone + Send> Send
    for ChunkIterator<'a, T, U>
{
}

impl<'a, T: ContentHandler + Clone, U: TimeZone> ExactSizeIterator for ChunkIterator<'a, T, U> {
    fn len(&self) -> usize {
        self.chunks.len()
    }
}

impl<'a, T: ContentHandler + Clone, U: TimeZone> DoubleEndedIterator for ChunkIterator<'a, T, U> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let res = self.chunks.next_back();

        let mut chunk = res?;
        let mut handler = self.handler.clone();
        let mut next_block_number = chunk.start_idx as u32;
        let mut buf: Vec<u8> = vec![];

        while let Ok(len) = chunk.inner.read_until(0xa, &mut buf) {
            if len > 0 {
                let line = buf.trim();
                if line.is_empty() {
                    buf.clear();
                    continue;
                }
                if line.is_timestamp() {
                    let sample_timestamp =
                        unsafe { line.convert_to_datetime(self.timezone.as_ref()) };
                    self.sample_tracker.process(sample_timestamp);
                    // NOTE: If there is ever a need to notify content handlers on sample start,
                    //       insert the notification here:
                    //       `self._handler.start_sample(sample_timestamp)`
                }
                // Find the first sample to process
                if !self.sample_tracker.is_current_sample_in_range() {
                    if self
                        .sample_tracker
                        .is_current_sample_greater_than_range_max()
                    {
                        // Already passed the last sample to process, so terminate the iteration
                        // (same as reaching the end of the file)
                        break;
                    }
                    buf.clear();
                    continue;
                }

                if line.is_separator() {
                    handler.end_sample(
                        self.sample_tracker.last_sample_number_processed(),
                        next_block_number,
                    );
                    if line.is_block_separator() {
                        let current_block_number = next_block_number;
                        next_block_number += 1;
                        handler.end_block(current_block_number);
                    }
                } else {
                    handler.end_event(line);
                }
            } else {
                break;
            }
            buf.clear();
        }
        // We reach here when we're done processing the file
        handler.end_sample(
            self.sample_tracker.last_sample_number_processed(),
            next_block_number,
        );
        handler.end_block(next_block_number);
        let last_chunk = handler.get_chunk_data();
        handler.end_file();
        if last_chunk.is_some() {
            last_chunk
        } else {
            None
        }
    }
}

impl<'a, T: ContentHandler + Clone + Send, U: TimeZone + Send> Producer
    for ChunkIterator<'a, T, U>
{
    type Item = T::Output;
    type IntoIter = Self;

    fn into_iter(self) -> Self::IntoIter {
        self
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let chunks: Vec<Chunk<u8>> = self.chunks.collect();
        let (left, right) = chunks.split_at(index);
        (
            Self {
                chunks: Vec::from(left).into_iter(),
                handler: self.handler.clone(),
                sample_tracker: self.sample_tracker.clone(),
                timezone: self.timezone.clone(),
            },
            Self {
                chunks: Vec::from(right).into_iter(),
                handler: self.handler,
                sample_tracker: self.sample_tracker,
                timezone: self.timezone,
            },
        )
    }
}

impl<'a, T: ContentHandler + Clone + Send, U: TimeZone + Send> ParallelIterator
    for ChunkIterator<'a, T, U>
{
    type Item = T::Output;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.chunks.len())
    }
}

impl<'a, T: ContentHandler + Clone + Send, U: TimeZone + Send> IndexedParallelIterator
    for ChunkIterator<'a, T, U>
{
    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        callback.callback(self)
    }
}

impl<'a, T: ContentHandler + Clone, U: TimeZone> Iterator for ChunkIterator<'a, T, U> {
    type Item = T::Output;
    fn next(&mut self) -> Option<Self::Item> {
        let res = self.chunks.next();

        let mut chunk = res?;
        let mut handler = self.handler.clone();
        let mut next_block_number = chunk.start_idx as u32;
        let mut buf: Vec<u8> = vec![];

        while let Ok(len) = chunk.inner.read_until(0xa, &mut buf) {
            if len > 0 {
                let line = buf.trim();
                if line.is_empty() {
                    buf.clear();
                    continue;
                }
                if line.is_timestamp() {
                    let sample_timestamp =
                        unsafe { line.convert_to_datetime(self.timezone.as_ref()) };
                    self.sample_tracker.process(sample_timestamp);
                    // NOTE: If there is ever a need to notify content handlers on sample start,
                    //       insert the notification here:
                    //       `self._handler.start_sample(sample_timestamp)`
                }
                // Find the first sample to process
                if !self.sample_tracker.is_current_sample_in_range() {
                    if self
                        .sample_tracker
                        .is_current_sample_greater_than_range_max()
                    {
                        // Already passed the last sample to process, so terminate the iteration
                        // (same as reaching the end of the file)
                        break;
                    }
                    buf.clear();
                    continue;
                }

                if line.is_separator() {
                    handler.end_sample(
                        self.sample_tracker.last_sample_number_processed(),
                        next_block_number,
                    );
                    if line.is_block_separator() {
                        let current_block_number = next_block_number;
                        next_block_number += 1;
                        handler.end_block(current_block_number);
                    }
                } else {
                    handler.end_event(line);
                }
            } else {
                break;
            }
            buf.clear();
        }
        // We reach here when we're done processing the file
        handler.end_sample(
            self.sample_tracker.last_sample_number_processed(),
            next_block_number,
        );
        handler.end_block(next_block_number);
        let last_chunk = handler.get_chunk_data();
        handler.end_file();
        if last_chunk.is_some() {
            last_chunk
        } else {
            None
        }
    }
}

/// Utility class for building a data frame from event values
struct EventDataFrameBuilder<'a, T: TimeZone> {
    system_info: &'a EmonSystemInformationParser,
    data: Vec<Series>,

    phantom: PhantomData<T>,
}

impl<'a, T: TimeZone + Send> EventDataFrameBuilder<'a, T> {
    fn new(system_info: &'a EmonSystemInformationParser) -> Self {
        let dtypes = [
            DataType::Int64,
            DataType::UInt32,
            DataType::String,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::UInt32,
            DataType::Float32,
            DataType::UInt32,
            DataType::String,
            DataType::Float32,
        ];
        let initial_series: Vec<Series> = redc
            .COLUMNS
            .iter()
            .enumerate()
            .map(|(i, col)| Series::new_empty(col, &dtypes[i]))
            .collect();
        Self {
            system_info,
            data: initial_series,
            phantom: PhantomData,
        }
    }

    #[inline]
    fn append_event_values(&mut self, sample: &Sample<T>, event: &Line) {
        let unit_count = event.values.len() as u32;
        let mut core_devices: Vec<String> = self.system_info.unique_core_types.clone();
        core_devices.push("core".to_owned());
        if core_devices.contains(&event.device.to_lowercase()) {
            self.append_core_event(sample, event)
        } else {
            self.append_noncore_event(sample, event, unit_count);
        }
    }

    #[inline]
    fn append_core_event(&mut self, sample: &Sample<T>, event: &Line) {
        let _time_stamp: Option<i64> = sample.time_stamp.to_timestamp();
        self.data[0]
            .append(&Series::new(
                redc.COLUMNS[0],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| _time_stamp)
                    .collect::<Vec<Option<i64>>>(),
            ))
            .unwrap();
        self.data[1]
            .append(&Series::new(
                redc.COLUMNS[1],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| self.system_info.socket_map[processor] as u32)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[2]
            .append(&Series::new(
                redc.COLUMNS[2],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| self.system_info.core_type_map[processor].as_str())
                    .collect::<Vec<&str>>(),
            ))
            .unwrap();
        self.data[3]
            .append(&Series::new(
                redc.COLUMNS[3],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| Some(self.system_info.core_map[processor] as u32))
                    .collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[4]
            .append(&Series::new(
                redc.COLUMNS[4],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| Some(self.system_info.thread_map[processor] as u32))
                    .collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[5]
            .append(&Series::new(
                redc.COLUMNS[5],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| *processor as u32)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[6]
            .append(&Series::new(
                redc.COLUMNS[6],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| Some(self.system_info.module_map[processor] as u32))
                    .collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[7]
            .append(&Series::new(
                redc.COLUMNS[7],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| sample.tsc_count)
                    .collect::<Vec<f32>>(),
            ))
            .unwrap();
        self.data[8]
            .append(&Series::new(
                redc.COLUMNS[8],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| sample.block_number)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[9]
            .append(&Series::new(
                redc.COLUMNS[9],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| event.name.as_str())
                    .collect::<Vec<&str>>(),
            ))
            .unwrap();
        self.data[10]
            .append(&Series::new(
                redc.COLUMNS[10],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .enumerate()
                    .map(|(index, _)| *event.values.get(index).unwrap_or(&f32::NAN))
                    .collect::<Vec<f32>>(),
            ))
            .unwrap();
    }

    #[inline]
    fn append_noncore_event(&mut self, sample: &Sample<T>, event: &Line, unit_count: u32) {
        let _time_stamp: Option<i64> = sample.time_stamp.to_timestamp();
        let socket_count = self
            .system_info
            .socket_map
            .values()
            .copied()
            .collect::<HashSet<i32>>()
            .len() as u32;
        let units_per_socket = unit_count / socket_count;
        self.data[0]
            .append(&Series::new(
                redc.COLUMNS[0],
                (0..unit_count)
                    .map(|_| _time_stamp)
                    .collect::<Vec<Option<i64>>>(),
            ))
            .unwrap();
        self.data[1]
            .append(&Series::new(
                redc.COLUMNS[1],
                (0..unit_count)
                    .map(|index| index / units_per_socket)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[2]
            .append(&Series::new(
                redc.COLUMNS[2],
                (0..unit_count)
                    .map(|_| event.device.as_str())
                    .collect::<Vec<&str>>(),
            ))
            .unwrap();
        self.data[3]
            .append(&Series::new(
                redc.COLUMNS[3],
                (0..unit_count).map(|_| None).collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[4]
            .append(&Series::new(
                redc.COLUMNS[4],
                (0..unit_count).map(|_| None).collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[5]
            .append(&Series::new(
                redc.COLUMNS[5],
                (0..unit_count)
                    .map(|index| index % units_per_socket)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[6]
            .append(&Series::new(
                redc.COLUMNS[6],
                (0..unit_count).map(|_| None).collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[7]
            .append(&Series::new(
                redc.COLUMNS[7],
                (0..unit_count)
                    .map(|_| sample.tsc_count)
                    .collect::<Vec<f32>>(),
            ))
            .unwrap();
        self.data[8]
            .append(&Series::new(
                redc.COLUMNS[8],
                (0..unit_count)
                    .map(|_| sample.block_number)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[9]
            .append(&Series::new(
                redc.COLUMNS[9],
                (0..unit_count)
                    .map(|_| event.name.as_str())
                    .collect::<Vec<&str>>(),
            ))
            .unwrap();
        self.data[10]
            .append(&Series::new(
                redc.COLUMNS[10],
                (0..unit_count)
                    .map(|index| *event.values.get(index as usize).unwrap_or(&f32::NAN))
                    .collect::<Vec<f32>>(),
            ))
            .unwrap();
    }

    #[inline]
    fn append_tsc(&mut self, sample: &Sample<T>) {
        let _time_stamp: Option<i64> = sample.time_stamp.to_timestamp();
        self.data[0]
            .append(&Series::new(
                redc.COLUMNS[0],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| _time_stamp)
                    .collect::<Vec<Option<i64>>>(),
            ))
            .unwrap();
        self.data[1]
            .append(&Series::new(
                redc.COLUMNS[1],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| self.system_info.socket_map[processor] as u32)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[2]
            .append(&Series::new(
                redc.COLUMNS[2],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| self.system_info.core_type_map[processor].as_str())
                    .collect::<Vec<&str>>(),
            ))
            .unwrap();
        self.data[3]
            .append(&Series::new(
                redc.COLUMNS[3],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| Some(self.system_info.core_map[processor] as u32))
                    .collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[4]
            .append(&Series::new(
                redc.COLUMNS[4],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| Some(self.system_info.thread_map[processor] as u32))
                    .collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[5]
            .append(&Series::new(
                redc.COLUMNS[5],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .enumerate()
                    .map(|(index, _)| index as u32)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[6]
            .append(&Series::new(
                redc.COLUMNS[6],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|processor| Some(self.system_info.module_map[processor] as u32))
                    .collect::<Vec<Option<u32>>>(),
            ))
            .unwrap();
        self.data[7]
            .append(&Series::new(
                redc.COLUMNS[7],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| sample.tsc_count)
                    .collect::<Vec<f32>>(),
            ))
            .unwrap();
        self.data[8]
            .append(&Series::new(
                redc.COLUMNS[8],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| sample.block_number)
                    .collect::<Vec<u32>>(),
            ))
            .unwrap();
        self.data[9]
            .append(&Series::new(
                redc.COLUMNS[9],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| "TSC")
                    .collect::<Vec<&str>>(),
            ))
            .unwrap();
        self.data[10]
            .append(&Series::new(
                redc.COLUMNS[10],
                self.system_info
                    .unique_os_processors
                    .iter()
                    .map(|_| sample.tsc_count)
                    .collect::<Vec<f32>>(),
            ))
            .unwrap();
    }

    #[inline]
    fn append_sampling_time(&mut self, sample: &Sample<T>) {
        let _time_stamp: Option<i64> = sample.time_stamp.to_timestamp();
        self.data[0]
            .append(&Series::new(redc.COLUMNS[0], [_time_stamp]))
            .unwrap();
        self.data[1]
            .append(&Series::new(redc.COLUMNS[1], [0_u32]))
            .unwrap();
        self.data[2]
            .append(&Series::new(redc.COLUMNS[2], ["SYSTEM"]))
            .unwrap();
        self.data[3]
            .append(&Series::new(redc.COLUMNS[3], [None as Option<u32>]))
            .unwrap();
        self.data[4]
            .append(&Series::new(redc.COLUMNS[4], [None as Option<u32>]))
            .unwrap();
        self.data[5]
            .append(&Series::new(redc.COLUMNS[5], [0_u32]))
            .unwrap();
        self.data[6]
            .append(&Series::new(redc.COLUMNS[6], [None as Option<u32>]))
            .unwrap();
        self.data[7]
            .append(&Series::new(redc.COLUMNS[7], [sample.tsc_count]))
            .unwrap();
        self.data[8]
            .append(&Series::new(redc.COLUMNS[8], [sample.block_number]))
            .unwrap();
        self.data[9]
            .append(&Series::new(redc.COLUMNS[9], ["$samplingTime"]))
            .unwrap();
        self.data[10]
            .append(&Series::new(redc.COLUMNS[10], [sample.duration]))
            .unwrap();
    }

    #[inline]
    fn append_processed_samples(&mut self, sample: &Sample<T>) {
        let _time_stamp: Option<i64> = sample.time_stamp.to_timestamp();
        self.data[0]
            .append(&Series::new(redc.COLUMNS[0], [_time_stamp]))
            .unwrap();
        self.data[1]
            .append(&Series::new(redc.COLUMNS[1], [0_u32]))
            .unwrap();
        self.data[2]
            .append(&Series::new(redc.COLUMNS[2], ["SYSTEM"]))
            .unwrap();
        self.data[3]
            .append(&Series::new(redc.COLUMNS[3], [None as Option<u32>]))
            .unwrap();
        self.data[4]
            .append(&Series::new(redc.COLUMNS[4], [None as Option<u32>]))
            .unwrap();
        self.data[5]
            .append(&Series::new(redc.COLUMNS[5], [0_u32]))
            .unwrap();
        self.data[6]
            .append(&Series::new(redc.COLUMNS[6], [None as Option<u32>]))
            .unwrap();
        self.data[7]
            .append(&Series::new(redc.COLUMNS[7], [sample.tsc_count]))
            .unwrap();
        self.data[8]
            .append(&Series::new(redc.COLUMNS[8], [sample.block_number]))
            .unwrap();
        self.data[9]
            .append(&Series::new(redc.COLUMNS[9], ["$processed_samples"]))
            .unwrap();
        self.data[10]
            .append(&Series::new(redc.COLUMNS[10], [1.0_f32]))
            .unwrap();
    }

    #[inline]
    fn append_pseudo_event_values(&mut self, sample: &Sample<T>) {
        self.append_tsc(sample);
        self.append_sampling_time(sample);
        self.append_processed_samples(sample);
    }

    fn into_dataframe(self) -> RawEmonDataFrame {
        let mut df = DataFrame::new(self.data.into_iter().map(|s| s.rechunk()).collect()).unwrap();
        df.with_column(
            df.column("timestamp")
                .unwrap()
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap(),
        )
        .unwrap();
        df.drop_nulls_seq(Some(&[redc.VALUE])).unwrap().lazy()
    }
}
