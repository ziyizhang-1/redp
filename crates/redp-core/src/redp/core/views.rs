#![allow(non_upper_case_globals)]

use super::metric_computer::MetricCompute;
use super::normalizer::Normalizer;
use super::types::{
    Device, EventInfoDataFrame, EventInfoDataFrameColumns, RawEmonDataFrameColumns,
    StatisticsDataFrameColumns,
};
use crate::utils::pivot_stable_seq::pivot_stable_seq;
use crate::utils::{DataframeSeqExt, GroupByExt};

use polars::functions::concat_df_horizontal;
use polars::lazy::dsl::{col, cols, concat_str, dtype_col, when, Expr};
use polars::prelude::*;

use rayon::iter::Either;
use tdigest::TDigest;

use std::cmp::Ordering::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Mutex;

static redc: RawEmonDataFrameColumns = RawEmonDataFrameColumns::new();
static eidc: EventInfoDataFrameColumns = EventInfoDataFrameColumns::new();
static sdf: StatisticsDataFrameColumns = StatisticsDataFrameColumns::new();

pub trait DataframeExt<'a, T: Clone, S: MetricCompute + Clone> {
    fn filter_core_type_event(self, device: &'a Device<T, S>) -> LazyFrame;
}

impl<'a, T: Clone, S: MetricCompute + Clone> DataframeExt<'a, T, S> for EventInfoDataFrame {
    fn filter_core_type_event(self, device: &'a Device<T, S>) -> LazyFrame {
        // TODO: change function name? see if this applies to all devices
        if self.schema().unwrap().get_names().contains(&eidc.DEVICE) {
            let df = self.collect().unwrap();
            let mask = df
                .column(eidc.DEVICE)
                .unwrap()
                .iter()
                .map(|x| device.exclusions.contains(&x.get_str().unwrap()))
                .collect_ca("mask");
            df._filter_seq(&mask).unwrap().lazy()
        } else {
            self
        }
    }
}

/// Supported view types. Can be combined using bitwise OR, e.g. SUMMARY | DETAILS
#[derive(Debug, Clone, Copy)]
pub enum ViewType {
    SUMMARY,
    DETAILS,
}

/// Supported view aggregation levels
#[derive(Debug, Clone, Copy)]
pub enum ViewAggregationLevel {
    SYSTEM,
    SOCKET,
    CORE,
    THREAD,
    UNCORE,
}

impl From<&ViewAggregationLevel> for String {
    fn from(value: &ViewAggregationLevel) -> Self {
        match value {
            ViewAggregationLevel::CORE => "core".to_owned(),
            ViewAggregationLevel::SOCKET => "socket".to_owned(),
            ViewAggregationLevel::SYSTEM => "system".to_owned(),
            ViewAggregationLevel::THREAD => "thread".to_owned(),
            ViewAggregationLevel::UNCORE => "uncore".to_owned(),
        }
    }
}

/// View attributes
#[derive(Clone)]
pub struct ViewAttributes<'a, T: Clone, S: MetricCompute + Clone> {
    /// a name that uniquely identifies this view
    view_name: String,
    /// view type, e.g. Summary, Details, ...
    view_type: ViewType,
    /// data aggregation level, e.g. System, Socket, Core, ...
    aggregation_level: T,
    /// device for this view # TODO: this may be required (can't be None)
    device: Option<Device<'a, T, S>>,
    /// ref_tsc
    ref_tsc: f32,
    /// show module information for this view
    show_modules: bool,
    /// metric computer assigned to the view
    metric_computer: Option<S>,
    /// constant values from the symbol table
    constant_values: Option<HashMap<String, f32>>,
    /// column names that must appear in the view data
    _required_events: Option<EventInfoDataFrame>,
}

impl<'a, T: Clone, S: MetricCompute + Clone> ViewAttributes<'a, T, S> {
    fn _update(
        &self,
        view_name: Option<String>,
        view_type: Option<ViewType>,
        aggregation_level: Option<T>,
        device: Option<Device<'a, T, S>>,
        ref_tsc: Option<f32>,
        show_modules: Option<bool>,
        metric_computer: Option<S>,
        constant_values: Option<HashMap<String, f32>>,
        required_events: Option<EventInfoDataFrame>,
    ) -> Self {
        Self {
            view_name: view_name.unwrap_or(self.view_name.clone()),
            view_type: view_type.unwrap_or(self.view_type),
            aggregation_level: aggregation_level.unwrap_or(self.aggregation_level.clone()),
            device: device.or(self.device.clone()),
            ref_tsc: ref_tsc.unwrap_or(self.ref_tsc),
            show_modules: show_modules.unwrap_or(self.show_modules),
            metric_computer: metric_computer.or(self.metric_computer.clone()),
            constant_values: constant_values.or(self.constant_values.clone()),
            _required_events: required_events.or(self._required_events.clone()),
        }
    }

    fn new(
        view_name: String,
        view_type: ViewType,
        aggregation_level: T,
        device: Option<Device<'a, T, S>>,
        ref_tsc: f32,
        show_modules: bool,
        metric_computer: Option<S>,
        constant_values: Option<HashMap<String, f32>>,
        _required_events: Option<EventInfoDataFrame>,
    ) -> Self {
        Self {
            view_name,
            view_type,
            aggregation_level,
            device,
            ref_tsc,
            show_modules,
            metric_computer,
            constant_values,
            _required_events,
        }
    }
}

#[derive(Clone)]
pub struct ViewData<'a, T: Clone, S: MetricCompute + Clone> {
    attributes: ViewAttributes<'a, T, S>,
    pub data: Option<LazyFrame>,
}

impl<'a, T: Clone, S: MetricCompute + Clone> ViewData<'a, T, S> {
    fn new(attributes: ViewAttributes<'a, T, S>, data: Option<LazyFrame>) -> Self {
        Self { attributes, data }
    }
}

impl<'a, T: Clone, S: MetricCompute + Clone> Debug for ViewData<'a, T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ViewData")
            .field(&self.attributes.view_name)
            .field(&self.data.clone().unwrap_or_default().collect().unwrap())
            .finish()
    }
}

#[derive(Clone, Default)]
/// Stores information about the EDP views to generate
pub struct ViewCollection<'a, T: Clone, S: MetricCompute + Clone> {
    view_configurations: Vec<ViewAttributes<'a, T, S>>,
}

impl<'a, T: Clone, S: MetricCompute + Clone> ViewCollection<'a, T, S> {
    /// Initializes an empty view collection.
    pub fn new() -> Self {
        Self {
            view_configurations: vec![],
        }
    }

    fn views(self) -> Vec<ViewAttributes<'a, T, S>> {
        self.view_configurations
    }

    fn _append_views(&mut self, views: Vec<ViewAttributes<'a, T, S>>) {
        /*!
               Appends a list of ViewAttributes to the internal __view_configurations

               @param: views, A list of ViewAttributes to be appended
               @return: None, updates the internal __view_configuration list of ViewAttributes
        */
        let mut views = views;
        self.view_configurations.append(&mut views);
    }

    pub fn add_view(
        &mut self,
        view_name: String,
        view_type: ViewType,
        aggregation_level: T,
        device: Option<Device<'a, T, S>>,
        ref_tsc: f32,
        show_modules: bool,
        metric_computer: Option<S>,
        constant_values: Option<HashMap<String, f32>>,
        required_events: Option<EventInfoDataFrame>,
    ) {
        /*!
               Add a view configuration

               :param view_name: a name that uniquely identify this view
               :param view_type: the view type to add, e.g. Summary, Details. You can specify multiple types
                               (e.g. Summary and Details) by combining types using bitwise OR.
               :param aggregation_level: the level at which to aggregate data, e.g. System, Socket, Core...
               @param device: device to be filtered for this view
               :param show_modules: include module information if True
               :param metric_computer: the metric computer to use for the specified view
               :param normalizer: the normalizer to use for the specified view
               :param required_events: Events required to generate the specified view
        */
        let mut required_events = required_events;

        assert!(
            !self
                .view_configurations
                .iter()
                .map(|x| x.view_name.as_str())
                .collect::<Vec<&str>>()
                .contains(&view_name.as_str()),
            "a view with the name '{}' already exists. Duplicate view names not allowed",
            &view_name
        );

        if device.is_some() && required_events.is_some() {
            required_events = Some(
                required_events
                    .unwrap()
                    .filter_core_type_event(device.as_ref().unwrap()),
            );
        }

        let view_attr = ViewAttributes::new(
            view_name,
            view_type,
            aggregation_level,
            device,
            ref_tsc,
            show_modules,
            metric_computer,
            constant_values,
            required_events,
        );
        self.view_configurations.push(view_attr);
    }
}

/// Generate views with various levels of data aggregation (per system, socket, core, thread...)
pub struct ViewGenerator<'a, T: Clone, S: MetricCompute + Clone> {
    pub views: Vec<SpecificDataView<'a, T, S>>,
    phantom_data: PhantomData<S>,
}

impl<'a, S: MetricCompute + Clone + 'a> ViewGenerator<'a, ViewAggregationLevel, S> {
    /// Initialize view generator
    /// :param view_collection: views to generate
    pub fn new(
        view_collection: Option<ViewCollection<'a, ViewAggregationLevel, S>>,
        phantom_data: PhantomData<S>,
    ) -> Self {
        if view_collection.is_none()
            || view_collection
                .as_ref()
                .unwrap()
                .view_configurations
                .is_empty()
        {
            panic!("at least one view is required but none provided");
        }
        let views: Vec<SpecificDataView<'_, ViewAggregationLevel, S>> = view_collection
            .unwrap()
            .views()
            .into_iter()
            .map(|x| match x.aggregation_level {
                ViewAggregationLevel::CORE => SpecificDataView::Core(CoreDataView::new(x)),
                ViewAggregationLevel::SOCKET => SpecificDataView::Socket(SocketDataView::new(x)),
                ViewAggregationLevel::SYSTEM => SpecificDataView::System(SystemDataView::new(x)),
                ViewAggregationLevel::THREAD => SpecificDataView::Thread(ThreadDataView::new(x)),
                ViewAggregationLevel::UNCORE => SpecificDataView::Uncore(UncoreDataView::new(x)),
            })
            .collect();

        Self {
            views,
            phantom_data,
        }
    }

    /// Process the input dataframe and generate data for all Detail Views
    /// :param df: input data frame
    /// :return: A list of `ViewData` objects, one for each Details View specified in the view generator configuration.
    pub fn generate_detail_views(
        &self,
        df: Option<LazyFrame>,
        no_details: bool,
    ) -> HashMap<String, ViewData<'_, ViewAggregationLevel, S>> {
        let mut results: HashMap<String, ViewData<'_, ViewAggregationLevel, S>> =
            HashMap::default();

        if df.is_none() {
            return results;
        }

        self.views
            .iter()
            .filter(|view| match view.attributes().view_type {
                ViewType::SUMMARY => {
                    if no_details {
                        matches!(
                            view.attributes().aggregation_level,
                            ViewAggregationLevel::SYSTEM
                        )
                    } else {
                        false
                    }
                }
                _ => !no_details,
            })
            .for_each(|view| {
                let details_view = view.generate_details(df.clone());
                if details_view.data.is_some() {
                    results.insert(
                        view.attributes().view_name.replace("summary", "details"),
                        details_view,
                    );
                }
            });

        results
    }

    /// Computes aggregated sums of event values for a raw input dataframe and each view type specified in the view
    /// generator configuration
    /// :@param df: RawEmonDataFrame input
    /// :return: aggregated event values for each view type specified in the view generator configuration
    pub fn compute_aggregates(
        &self,
        df: Option<LazyFrame>,
    ) -> Vec<ViewData<'_, ViewAggregationLevel, S>> {
        if df.is_none() {
            return self
                .views
                .iter()
                .filter(|view| !matches!(view.attributes().view_type, ViewType::DETAILS))
                .map(|view| ViewData::new(view.attributes().clone(), None))
                .collect::<Vec<ViewData<'_, ViewAggregationLevel, S>>>();
        }

        self.views
            .iter()
            .filter(|view| !matches!(view.attributes().view_type, ViewType::DETAILS))
            .map(|view| view.compute_aggregate(df.clone()))
            .collect::<Vec<ViewData<'_, ViewAggregationLevel, S>>>()
    }
}

/// Accumulates event counts for summary views with various levels of data aggregation (per system, socket, core,
/// thread...)
/// Also updates statistics for these views (currently only the system summary view)
pub struct DataAccumulator<'a, T: Clone, S: MetricCompute + Clone> {
    views: HashMap<String, SpecificDataView<'a, T, S>>,
    event_views: HashMap<String, Mutex<ViewData<'a, T, S>>>,
    stats_views: HashMap<String, Mutex<ViewData<'a, T, S>>>,
}

unsafe impl<'a, T: Clone, S: MetricCompute + Clone> Sync for DataAccumulator<'a, T, S> {}

impl<'a, T: Clone, S: MetricCompute + Clone> DataAccumulator<'a, T, S> {
    pub fn new(view_generator: &'a ViewGenerator<'_, T, S>) -> Self {
        assert!(
            !view_generator.views.is_empty(),
            "view_collection must have at least one view"
        );

        let mut views: HashMap<String, SpecificDataView<'a, T, S>> = HashMap::default();
        let mut event_views: HashMap<String, Mutex<ViewData<'a, T, S>>> = HashMap::default();
        let mut stats_views: HashMap<String, Mutex<ViewData<'a, T, S>>> = HashMap::default();
        view_generator.views.iter().for_each(|view| {
            views.insert(view.attributes().view_name.clone(), view.clone());
            event_views.insert(
                view.attributes().view_name.clone(),
                Mutex::new(ViewData::new(view.attributes().clone(), None)),
            );
            stats_views.insert(
                view.attributes().view_name.clone(),
                Mutex::new(ViewData::new(view.attributes().clone(), None)),
            );
        });
        Self {
            views,
            event_views,
            stats_views,
        }
    }

    #[allow(clippy::mut_mutex_lock)]
    pub fn update_aggregates(&mut self, summary_computation: &[ViewData<'_, T, S>]) {
        summary_computation.iter().for_each(|new_view| {
            let data_view = self.views.get(&new_view.attributes.view_name).unwrap();
            let view = self
                .event_views
                .get_mut(&new_view.attributes.view_name)
                .unwrap();
            let mut view = view.lock().unwrap(); // here must use lock() instead of get_mut() to ensure safety
            view.data = data_view.update_summary(new_view.data.clone(), view.data.clone());
        });
    }

    #[allow(clippy::mut_mutex_lock)]
    pub fn update_statistics(
        &mut self,
        detail_views: Option<&HashMap<String, ViewData<'_, T, S>>>,
    ) {
        /*!
               Updates summary statistics for a list of detail views or a RawEmonDataFrame.
               Takes in either a list of detail views or a RawEmonDataFrame. Only pass in a RawEmonDataFrame if you do not
               wish to generate detail views, otherwise only pass in detail_views
               @param detail_views: a list of detail_views required to update summary statistics
               @param df: a raw Emon or perfmon dataframe
               @return: None, internal stats_views will be updated and persisted inside of DataAccumulator
        */
        assert!(
            detail_views.is_some(),
            "if no detail views are requested, then a dataframe must be passed into this method"
        );

        let detail_views = detail_views.unwrap();

        for (view_id, view) in self.stats_views.iter_mut() {
            let mut view = view.lock().unwrap(); // here must use lock() instead of get_mut() to ensure safety
            match view.attributes.view_type {
                ViewType::DETAILS => {
                    continue;
                }
                _ => {
                    let data_view = self.views.get_mut(view_id).unwrap();
                    let df = match detail_views
                        .get(&view_id.replace("summary", "details"))
                        .or(detail_views.get(&view_id.to_owned()))
                    {
                        Some(v) => v.data.clone(),
                        None => None,
                    };
                    view.data = data_view.update_statistics(df);
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_summary_views(self) -> HashMap<String, ViewData<'a, T, S>> {
        /*!
               Computes metrics for each view and updates statistics when needed.
               :return: summary view for each view type specified in the view generator configuration
        */
        let mut summary_data_views: HashMap<
            String,
            (ViewData<'_, T, S>, SpecificDataView<'_, T, S>),
        > = HashMap::new();
        let mut views = self.views;
        self.event_views.into_iter().for_each(|(key, view)| {
            summary_data_views.insert(
                key.clone(),
                (view.into_inner().unwrap(), views.remove(&key).unwrap()),
            );
        });
        let mut summary_views: HashMap<String, ViewData<'_, T, S>> = HashMap::new();
        summary_data_views
            .into_iter()
            .for_each(|(view_id, (mut summary_view, data_view))| {
                let stats_view = self
                    .stats_views
                    .get(&view_id)
                    .unwrap()
                    .lock()
                    .unwrap()
                    .clone();
                match summary_view.attributes.view_type {
                    ViewType::DETAILS => {}
                    _ => {
                        summary_view = ViewData::new(
                            summary_view.attributes.clone(),
                            data_view.generate_summary(summary_view.data).data,
                        );
                        if let Some(v) = stats_view.data {
                            let aggregated = summary_view.data.unwrap().collect().unwrap();
                            let others = v
                                .collect()
                                .unwrap()
                                .transpose_seq(Some("stat"), Some(Either::Left("name".to_owned())))
                                .unwrap();
                            summary_view.data = Some(
                                concat(
                                    &[aggregated.lazy(), others.lazy()],
                                    UnionArgs {
                                        parallel: false,
                                        rechunk: true,
                                        to_supertypes: false,
                                        diagonal: false,
                                        from_partitioned_ds: false,
                                    },
                                )
                                .unwrap(),
                            );
                        }

                        summary_views
                            .insert(summary_view.attributes.view_name.clone(), summary_view);
                    }
                }
            });
        summary_views
    }
}

/// Define the common parameters and logic for generating an EDP data view
pub trait DataView<T: Clone, S: MetricCompute + Clone> {
    type Input;
    type Output;
    /// Process the input dataframe and update summary statistics
    /// @param df: input data frame of new event values
    /// @param event_summary_values: persisted summary values from DataAccumulator
    fn update_summary(
        &self,
        df: Option<Self::Input>,
        event_summary_values: Option<Self::Input>,
    ) -> Option<Self::Output>;

    /// A method that subclasses can override to update statistics in summary views
    #[allow(unused_variables)]
    fn update_statistics(&mut self, details_view_df: Option<Self::Input>) -> Option<Self::Output> {
        None
    }

    /// Process the input dataframe and return details data
    /// :param df: input data frame
    /// :return: a `DetailsData` object. The `data` member contain the details view data, or an empty data frame if the
    ///          view is not configured to generate details data
    fn generate_details(&self, df: Option<Self::Input>) -> ViewData<'_, T, S>;

    fn generate_details_dataframe(
        &self,
        lf: Option<LazyFrame>,
        device_filter: Option<&[&str]>,
        filter_mode: &str,
        ref_tsc: f32,
        details_group_by: &[&str],
        details_index: &[&str],
        metric_computer: Option<&S>,
        constant_values: Option<&HashMap<String, f32>>,
        normalize: bool,
    ) -> Option<LazyFrame> {
        let lf = lf?;
        let mut df = lf.collect().unwrap();

        df = match device_filter {
            Some(v) => match filter_mode {
                "include" => {
                    let mask = df
                        .column(redc.DEVICE)
                        .unwrap()
                        .iter()
                        .map(|x| v.contains(&x.get_str().unwrap()))
                        .collect_ca("mask");
                    df._filter_seq(&mask).unwrap()
                }
                "exclude" => {
                    let mask = df
                        .column(redc.DEVICE)
                        .unwrap()
                        .iter()
                        .map(|x| !v.contains(&x.get_str().unwrap()))
                        .collect_ca("mask");
                    df._filter_seq(&mask).unwrap()
                }
                _ => panic!("device_filter_mode must be either 'include' or 'exclude'"),
            },
            None => df,
        };

        if df.height() == 0 {
            return None;
        }

        if normalize {
            df = df.lazy().normalize(ref_tsc, "columns").collect().unwrap();
        }

        df = df
            .group_by_with_series(df.select_series(details_group_by).unwrap(), false, true)
            .unwrap()
            .select([redc.VALUE])
            .sum_seq(
                df.select_series(details_group_by).unwrap(),
                Some(vec![String::from(redc.VALUE)]),
            )
            .unwrap();
        df.rename("value_sum", "value").unwrap();

        let mut events_df = pivot_stable_seq(
            &df,
            details_index,
            [redc.NAME],
            Some([redc.VALUE]),
            true,
            Some(
                dtype_col(&DataType::Float32)
                    .mean()
                    .strict_cast(DataType::Float32),
            ),
            None,
        )
        .unwrap()
        .lazy();

        let metrics_df = metric_computer.unwrap().compute_metric(
            events_df.clone(),
            constant_values,
            true,
            Some(details_index),
        );

        events_df = events_df.drop(details_index);

        Some(
            concat_df_horizontal(&[metrics_df.collect().unwrap(), events_df.collect().unwrap()])
                .unwrap()
                .lazy(),
        )
    }

    /// Return summary data
    /// :return: a dataframe with the summary view data
    fn compute_aggregate(&self, df: Option<Self::Input>) -> ViewData<'_, T, S>;

    fn generate_aggregate_dataframe(
        &self,
        lf: Option<LazyFrame>,
        device_filter: Option<&[&str]>,
        filter_mode: &str,
        aggregator_columns: &[&str],
        aggregator_group_by: &[&str],
    ) -> Option<LazyFrame> {
        let lf = lf?;
        let mut df = lf.collect().unwrap();
        df = match device_filter {
            Some(v) => match filter_mode {
                "include" => {
                    let mask = df
                        .column(redc.DEVICE)
                        .unwrap()
                        .iter()
                        .map(|x| v.contains(&x.get_str().unwrap()))
                        .collect_ca("mask");
                    df._filter_seq(&mask).unwrap()
                }
                "exclude" => {
                    let mask = df
                        .column(redc.DEVICE)
                        .unwrap()
                        .iter()
                        .map(|x| !v.contains(&x.get_str().unwrap()))
                        .collect_ca("mask");
                    df._filter_seq(&mask).unwrap()
                }
                _ => panic!("device_filter_mode must be either 'include' or 'exclude'"),
            },
            None => df,
        };
        df = df.select(aggregator_columns).unwrap();

        if df.height() == 0 {
            return None;
        }

        Some(
            df.group_by_with_series(df.select_series(aggregator_group_by).unwrap(), false, true)
                .unwrap()
                .select(
                    df.get_column_names()
                        .into_iter()
                        .filter(|x| !aggregator_group_by.contains(x)),
                )
                .sum_seq(
                    df.select_series(aggregator_group_by).unwrap(),
                    Some(
                        df.get_column_names()
                            .into_iter()
                            .filter_map(|x| match !aggregator_group_by.contains(&x) {
                                true => Some(x.to_owned()),
                                false => None,
                            })
                            .collect(),
                    ),
                )
                .unwrap()
                .lazy()
                .rename(["tsc_sum", "value_sum"], ["tsc", "value"]),
        )
    }

    /// Return summary data
    /// :return: a dataframe with the summary view data
    fn generate_summary(&self, event_df: Option<Self::Input>) -> ViewData<'_, T, S>;

    fn generate_summary_dataframe(
        &self,
        lf: Option<LazyFrame>,
        ref_tsc: f32,
        summary_group_by: &[&str],
        metric_computer: Option<&S>,
        constant_values: Option<&HashMap<String, f32>>,
        normalize: bool,
    ) -> Option<LazyFrame> {
        let mut lf = lf?;

        if normalize {
            lf = lf.normalize(ref_tsc, "columns");
        }
        let mut df = lf.collect().unwrap();

        df = df
            .group_by_with_series(df.select_series(summary_group_by).unwrap(), false, true)
            .unwrap()
            .select([redc.VALUE])
            .sum_seq(
                df.select_series(summary_group_by).unwrap(),
                Some(vec![String::from(redc.VALUE)]),
            )
            .unwrap()
            .sort(
                summary_group_by,
                SortMultipleOptions::default().with_multithreaded(false),
            )
            .unwrap();

        df.rename("value_sum", "value").unwrap();
        let cols: Vec<String> = df
            .get_column_names()
            .into_iter()
            .map(|x| x.to_owned())
            .collect();
        let _cols: Vec<&str> = cols.iter().map(|x| x.as_str()).collect();
        let mut lf_d = df.lazy();

        lf = match _cols.as_slice() {
            ["name", "value"] => lf_d
                .collect()
                .unwrap()
                .transpose_seq(None, Some(Either::Left(redc.NAME.to_owned())))
                .unwrap()
                .lazy(),
            ["name", "socket", "value"] => pivot_stable_seq(
                &lf_d.collect().unwrap(),
                [redc.NAME],
                [redc.SOCKET],
                Some([redc.VALUE]),
                false,
                None,
                None,
            )
            .unwrap()
            .transpose_seq(Some("Device"), Some(Either::Left(redc.NAME.to_owned())))
            .unwrap()
            .lazy(),
            ["name", "socket", "core", "value"] => {
                lf_d = lf_d.with_column(
                    concat_str([col(redc.SOCKET), col(redc.CORE)], "_", true).alias("CoreMap"),
                );
                pivot_stable_seq(
                    &lf_d.collect().unwrap(),
                    [redc.NAME],
                    ["CoreMap"],
                    Some([redc.VALUE]),
                    false,
                    None,
                    None,
                )
                .unwrap()
                .transpose_seq(Some("Device"), Some(Either::Left(redc.NAME.to_owned())))
                .unwrap()
                .lazy()
            }
            ["name", "socket", "module", "core", "value"] => {
                lf_d = lf_d.with_column(
                    concat_str(
                        [col(redc.SOCKET), col(redc.MODULE), col(redc.CORE)],
                        "_",
                        true,
                    )
                    .alias("CoreMap"),
                );
                pivot_stable_seq(
                    &lf_d.collect().unwrap(),
                    [redc.NAME],
                    ["CoreMap"],
                    Some([redc.VALUE]),
                    false,
                    None,
                    None,
                )
                .unwrap()
                .transpose_seq(Some("Device"), Some(Either::Left(redc.NAME.to_owned())))
                .unwrap()
                .lazy()
            }
            ["name", "unit", "socket", "core", "thread", "value"] => pivot_stable_seq(
                &lf_d.collect().unwrap(),
                [redc.NAME],
                [redc.UNIT],
                Some([redc.VALUE]),
                false,
                None,
                None,
            )
            .unwrap()
            .transpose_seq(Some("Device"), Some(Either::Left(redc.NAME.to_owned())))
            .unwrap()
            .lazy(),
            ["name", "unit", "socket", "module", "core", "thread", "value"] => pivot_stable_seq(
                &lf_d.collect().unwrap(),
                [redc.NAME],
                [redc.UNIT],
                Some([redc.VALUE]),
                false,
                None,
                None,
            )
            .unwrap()
            .transpose_seq(Some("Device"), Some(Either::Left(redc.NAME.to_owned())))
            .unwrap()
            .lazy(),
            ["name", "unit", "socket", "value"] => {
                lf_d = lf_d
                    .sort(
                        [redc.SOCKET],
                        SortMultipleOptions::default().with_multithreaded(false),
                    )
                    .with_column(
                        concat_str([col(redc.SOCKET), col(redc.UNIT)], "_", true).alias("UnitMap"),
                    );
                pivot_stable_seq(
                    &lf_d.collect().unwrap(),
                    [redc.NAME],
                    ["UnitMap"],
                    Some([redc.VALUE]),
                    false,
                    None,
                    None,
                )
                .unwrap()
                .transpose_seq(Some("Device"), Some(Either::Left(redc.NAME.to_owned())))
                .unwrap()
                .lazy()
            }
            _ => {
                panic!("Columns do not match: {:#?}", _cols);
            }
        };

        let device: Option<LazyFrame>;
        if lf.schema().unwrap().get_names().contains(&"Device") {
            device = Some(lf.clone().select_seq([col("Device")]));
            lf = lf.drop(["Device"]);
        } else {
            device = None;
        }

        let summary_metrics_df =
            metric_computer
                .unwrap()
                .compute_metric(lf.clone(), constant_values, false, None);

        match device {
            Some(v) => Some(
                concat_df_horizontal(&[
                    v.collect().unwrap(),
                    summary_metrics_df.collect().unwrap(),
                    lf.collect().unwrap(),
                ])
                .unwrap()
                .lazy(),
            ),
            None => Some(
                concat_df_horizontal(&[
                    summary_metrics_df.collect().unwrap(),
                    lf.collect().unwrap(),
                ])
                .unwrap()
                .lazy(),
            ),
        }
    }

    fn update_summary_values(
        &self,
        df: Option<LazyFrame>,
        event_summary_values: Option<LazyFrame>,
        aggregator_columns: &[&str],
        aggregator_group_by: &[&str],
    ) -> Option<LazyFrame> {
        let lf = df?;
        let schema_A = lf.schema().unwrap();
        let cols_A: HashSet<&str> = HashSet::from_iter(schema_A.get_names());

        let mut new_event_summary_values = lf
            .select_seq([cols(aggregator_columns)])
            .collect()
            .unwrap()
            .sort(
                aggregator_group_by,
                SortMultipleOptions::default().with_multithreaded(false),
            )
            .unwrap();

        if event_summary_values.is_none() {
            return Some(new_event_summary_values.lazy());
        }

        let event_summary_values = event_summary_values.unwrap();

        let schema_B = event_summary_values.schema().unwrap();
        let cols_B: HashSet<&str> = HashSet::from_iter(schema_B.get_names());

        if cols_A.difference(&cols_B).next().is_some() {
            Some(
                event_summary_values
                    .join_builder()
                    .with(new_event_summary_values.lazy())
                    .how(JoinType::Left)
                    .left_on([cols(aggregator_columns)])
                    .right_on([cols(aggregator_columns)])
                    .allow_parallel(false)
                    .finish()
                    .cache(),
            )
        } else {
            unsafe {
                let mut df = event_summary_values
                    .collect()
                    .unwrap()
                    .sort(
                        aggregator_group_by,
                        SortMultipleOptions::default().with_multithreaded(false),
                    )
                    .unwrap();
                let columns = df.get_column_names_owned();

                match df.height().cmp(&new_event_summary_values.height()) {
                    Less => {
                        df = df
                            .lazy()
                            .join_builder()
                            .with(new_event_summary_values.clone().lazy())
                            .how(JoinType::Outer)
                            .left_on(
                                columns[..columns.len() - 2]
                                    .iter()
                                    .map(|x| col(x))
                                    .collect::<Vec<Expr>>(),
                            )
                            .right_on(
                                columns[..columns.len() - 2]
                                    .iter()
                                    .map(|x| col(x))
                                    .collect::<Vec<Expr>>(),
                            )
                            .allow_parallel(false)
                            .suffix("_right")
                            .finish()
                            .with_column(
                                when(col("name").is_null())
                                    .then(col("name_right").alias("name"))
                                    .otherwise(col("name")),
                            )
                            .select_seq([cols(columns)])
                            .collect()
                            .unwrap()
                            .fill_null_seq(FillNullStrategy::Zero)
                            .unwrap();
                    }
                    Greater => {
                        new_event_summary_values = new_event_summary_values
                            .lazy()
                            .join_builder()
                            .with(df.clone().lazy())
                            .how(JoinType::Outer)
                            .left_on(
                                columns[..columns.len() - 2]
                                    .iter()
                                    .map(|x| col(x))
                                    .collect::<Vec<Expr>>(),
                            )
                            .right_on(
                                columns[..columns.len() - 2]
                                    .iter()
                                    .map(|x| col(x))
                                    .collect::<Vec<Expr>>(),
                            )
                            .allow_parallel(false)
                            .suffix("_right")
                            .finish()
                            .with_column(
                                when(col("name").is_null())
                                    .then(col("name_right").alias("name"))
                                    .otherwise(col("name")),
                            )
                            .select_seq([cols(columns)])
                            .collect()
                            .unwrap()
                            .fill_null_seq(FillNullStrategy::Zero)
                            .unwrap();
                    }
                    Equal => {}
                }

                df.get_columns_mut().iter_mut().for_each(|s| {
                    *s = s
                        .f32()
                        .map(|x| {
                            Series::from(
                                x + new_event_summary_values
                                    .column(s.name())
                                    .unwrap()
                                    .f32()
                                    .unwrap(),
                            )
                        })
                        .unwrap_or(s.clone())
                });
                Some(df.lazy())
            }
        }
    }
}

#[derive(Clone)]
pub enum SpecificDataView<'a, T: Clone, S: MetricCompute + Clone> {
    System(SystemDataView<'a, T, S>),
    Socket(SocketDataView<'a, T, S>),
    Core(CoreDataView<'a, T, S>),
    Thread(ThreadDataView<'a, T, S>),
    Uncore(UncoreDataView<'a, T, S>),
}

impl<'a, T: Clone, S: MetricCompute + Clone> SpecificDataView<'a, T, S> {
    pub fn attributes(&self) -> &ViewAttributes<'a, T, S> {
        match self {
            Self::Core(v) => &v.attributes,
            Self::Socket(v) => &v.attributes,
            Self::System(v) => &v.attributes,
            Self::Thread(v) => &v.attributes,
            Self::Uncore(v) => &v.attributes,
        }
    }

    pub fn generate_details(&self, df: Option<LazyFrame>) -> ViewData<'_, T, S> {
        match self {
            Self::Core(v) => v.generate_details(df),
            Self::Socket(v) => v.generate_details(df),
            Self::System(v) => v.generate_details(df),
            Self::Thread(v) => v.generate_details(df),
            Self::Uncore(v) => v.generate_details(df),
        }
    }

    pub fn generate_summary(&self, df: Option<LazyFrame>) -> ViewData<'_, T, S> {
        match self {
            Self::Core(v) => v.generate_summary(df),
            Self::Socket(v) => v.generate_summary(df),
            Self::System(v) => v.generate_summary(df),
            Self::Thread(v) => v.generate_summary(df),
            Self::Uncore(v) => v.generate_summary(df),
        }
    }

    pub fn compute_aggregate(&self, df: Option<LazyFrame>) -> ViewData<'_, T, S> {
        match self {
            Self::Core(v) => v.compute_aggregate(df),
            Self::Socket(v) => v.compute_aggregate(df),
            Self::System(v) => v.compute_aggregate(df),
            Self::Thread(v) => v.compute_aggregate(df),
            Self::Uncore(v) => v.compute_aggregate(df),
        }
    }

    pub fn update_summary(
        &self,
        df: Option<LazyFrame>,
        event_summary_values: Option<LazyFrame>,
    ) -> Option<LazyFrame> {
        match self {
            Self::Core(v) => v.update_summary(df, event_summary_values),
            Self::Socket(v) => v.update_summary(df, event_summary_values),
            Self::System(v) => v.update_summary(df, event_summary_values),
            Self::Thread(v) => v.update_summary(df, event_summary_values),
            Self::Uncore(v) => v.update_summary(df, event_summary_values),
        }
    }

    pub fn update_statistics(&mut self, details_view_df: Option<LazyFrame>) -> Option<LazyFrame> {
        match self {
            Self::Core(v) => v.update_statistics(details_view_df),
            Self::Socket(v) => v.update_statistics(details_view_df),
            Self::System(v) => v.update_statistics(details_view_df),
            Self::Thread(v) => v.update_statistics(details_view_df),
            Self::Uncore(v) => v.update_statistics(details_view_df),
        }
    }
}

trait BaseStat {
    type Output;
    /// Compute aggregated statistic/s for an input dataframe
    fn compute(&mut self, df: &DataFrame);

    /// Return the resulting dataframe after computing the statistic/s
    fn get_stats_values(&self) -> Self::Output;
}

#[derive(Clone)]
struct MinMax {
    stats_df: Option<LazyFrame>,
    columns: [&'static str; 2],
}

impl MinMax {
    fn new() -> Self {
        let columns = [sdf.MIN, sdf.MAX];
        Self {
            stats_df: None,
            columns,
        }
    }
}

impl BaseStat for MinMax {
    type Output = DataFrame;
    fn compute(&mut self, df: &DataFrame) {
        let min = df
            .clone()
            .lazy()
            .select_seq([dtype_col(&DataType::Float32).min()]);
        let max = df
            .clone()
            .lazy()
            .select_seq([dtype_col(&DataType::Float32).max()]);

        let block_stats_df = concat(
            [min, max],
            UnionArgs {
                parallel: false,
                rechunk: true,
                to_supertypes: false,
                diagonal: false,
                from_partitioned_ds: false,
            },
        )
        .unwrap()
        .collect()
        .unwrap()
        .transpose_seq(
            Some("name"),
            Some(Either::Right(
                self.columns
                    .into_iter()
                    .map(|x| x.to_owned())
                    .collect::<Vec<String>>(),
            )),
        )
        .unwrap()
        .lazy();

        match self.stats_df {
            Some(ref v) => {
                let merged_stats_df = v
                    .clone()
                    .join_builder()
                    .with(block_stats_df)
                    .how(JoinType::Outer)
                    .left_on([col("name")])
                    .right_on([col("name")])
                    .allow_parallel(false)
                    .suffix("_right")
                    .finish();
                self.stats_df = Some(
                    merged_stats_df.select_seq([
                        when(col("name").is_null())
                            .then(col("name_right").alias("name"))
                            .otherwise(col("name")),
                        when(
                            col(&(sdf.MIN.to_owned() + "_right"))
                                .is_null()
                                .or(col(sdf.MIN).lt(col(&(sdf.MIN.to_owned() + "_right")))),
                        )
                        .then(col(sdf.MIN))
                        .otherwise(col(&(sdf.MIN.to_owned() + "_right")).alias(sdf.MIN)),
                        when(
                            col(&(sdf.MAX.to_owned() + "_right"))
                                .is_null()
                                .or(col(sdf.MAX).gt(col(&(sdf.MAX.to_owned() + "_right")))),
                        )
                        .then(col(sdf.MAX))
                        .otherwise(col(&(sdf.MAX.to_owned() + "_right")).alias(sdf.MAX)),
                    ]),
                );
            }
            None => self.stats_df = Some(block_stats_df),
        }
    }

    fn get_stats_values(&self) -> Self::Output {
        self.stats_df
            .clone()
            .unwrap()
            .select_seq([cols([sdf.MIN, sdf.MAX])])
            .collect()
            .unwrap()
    }
}

#[derive(Clone)]
struct Percentile {
    event_percentiles: HashMap<String, TDigest>,
    percentile: u8,
    columns: &'static str,
    names: Vec<String>,
}

impl Percentile {
    fn new(percentile: u8) -> Self {
        Self {
            event_percentiles: HashMap::default(),
            percentile,
            columns: sdf.PERCENTILE,
            names: vec![],
        }
    }
}

impl BaseStat for Percentile {
    type Output = DataFrame;
    fn compute(&mut self, df: &DataFrame) {
        let cols: Vec<String> = df
            .iter()
            .filter(|s| s.dtype() == &DataType::Float32)
            .map(|s| s.name().to_owned())
            .collect();

        if self.names.len() < cols.len() {
            self.names.clone_from(&cols);
        }

        for s in df.columns(cols).unwrap() {
            let values: Vec<f64> = s
                .filter(&s.is_not_nan().unwrap())
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .to_vec()
                .into_iter()
                .flatten()
                .collect();
            match self.event_percentiles.get_mut(s.name()) {
                Some(v) => *v = v.merge_unsorted(values),
                None => {
                    self.event_percentiles
                        .insert(s.name().to_owned(), TDigest::new_with_size(25));
                }
            }
        }
    }

    fn get_stats_values(&self) -> Self::Output {
        let values: Vec<f32> = self
            .names
            .iter()
            .map(|v| {
                self.event_percentiles
                    .get(v)
                    .unwrap()
                    .estimate_quantile(self.percentile as f64 / 100.0) as f32
            })
            .collect();
        DataFrame::new(vec![Series::from_vec(self.columns, values)]).unwrap()
    }
}

#[derive(Clone)]
struct Variation {
    event_variance: HashMap<String, f32>,
    event_aggregate: HashMap<String, (f32, f32, f32)>,
    tmp_aggregate: (f32, f32, f32),
    columns: &'static str,
    names: Vec<String>,
}

impl Variation {
    fn new() -> Self {
        Self {
            event_variance: HashMap::default(),
            event_aggregate: HashMap::default(),
            tmp_aggregate: (0.0, 0.0, 0.0),
            columns: sdf.VARIATION,
            names: vec![],
        }
    }

    #[inline]
    fn update_aggregate_value(
        existing_aggregate: (f32, f32, f32),
        new_value: f32,
    ) -> (f32, f32, f32) {
        let (mut count, mut mean, mut M2) = existing_aggregate;
        count += 1.0;
        let delta = new_value - mean;
        mean += delta / count;
        let delta2 = new_value - mean;
        M2 += delta * delta2;
        (count, mean, M2)
    }

    #[inline]
    fn get_variation(existing_aggregate: (f32, f32, f32)) -> f32 {
        let (count, mean, M2) = existing_aggregate;
        if count < 2.0 {
            0.0
        } else {
            (M2 / count).sqrt() / mean
        }
    }

    #[inline]
    fn compute_variance_value(&mut self, value: f32) -> f32 {
        self.tmp_aggregate = Self::update_aggregate_value(self.tmp_aggregate, value);
        Self::get_variation(self.tmp_aggregate)
    }
}

impl BaseStat for Variation {
    type Output = DataFrame;
    fn compute(&mut self, df: &DataFrame) {
        let cols: Vec<String> = df
            .iter()
            .filter(|s| s.dtype() == &DataType::Float32)
            .map(|s| s.name().to_owned())
            .collect();

        if self.names.len() < cols.len() {
            self.names.clone_from(&cols);
        }

        for s in df.columns(cols).unwrap() {
            self.tmp_aggregate = *self
                .event_aggregate
                .get(s.name())
                .unwrap_or(&(0.0, 0.0, 0.0));

            let values = s.drop_nulls();
            let v: f32;
            if !values.is_empty() {
                v = values
                    .f32()
                    .unwrap()
                    .to_vec()
                    .iter()
                    .map(|x| self.compute_variance_value(x.unwrap()))
                    .last()
                    .unwrap();
                self.event_variance.insert(s.name().to_owned(), v);
            }
            self.event_aggregate
                .insert(s.name().to_owned(), self.tmp_aggregate);
        }
    }

    fn get_stats_values(&self) -> Self::Output {
        assert_eq!(self.names.len(), self.event_variance.len());
        let values: Vec<f32> = self
            .names
            .iter()
            .map(|v| *self.event_variance.get(v).unwrap())
            .collect();
        DataFrame::new(vec![Series::from_vec(self.columns, values)]).unwrap()
    }
}

#[derive(Clone)]
enum Stats {
    MinMax(MinMax),
    Percentile(Percentile),
    Variation(Variation),
}

impl BaseStat for Stats {
    type Output = DataFrame;
    fn compute(&mut self, df: &DataFrame) {
        match self {
            Stats::MinMax(v) => v.compute(df),
            Stats::Percentile(v) => v.compute(df),
            Stats::Variation(v) => v.compute(df),
        }
    }

    fn get_stats_values(&self) -> Self::Output {
        match self {
            Stats::MinMax(v) => v.get_stats_values(),
            Stats::Percentile(v) => v.get_stats_values(),
            Stats::Variation(v) => v.get_stats_values(),
        }
    }
}

#[derive(Clone)]
/// Compute various statistics (min, max, percentile) for events and metrics
struct Statistics {
    base_stats: Vec<Stats>,
    stats_df: LazyFrame,
    header: Option<DataFrame>,
}

unsafe impl Sync for Statistics {}

impl Statistics {
    fn new(base_stats: Option<Vec<Stats>>) -> Self {
        let base_stats = match base_stats {
            Some(v) => v,
            None => {
                let items: Vec<Stats> = vec![
                    Stats::MinMax(MinMax::new()),
                    Stats::Percentile(Percentile::new(95)),
                    Stats::Variation(Variation::new()),
                ];
                items
            }
        };
        Self {
            base_stats,
            stats_df: LazyFrame::default(),
            header: None,
        }
    }

    /// Compute statistics from the input data frame and update object state
    /// :param df: input data frame
    fn compute(&mut self, df: &DataFrame) {
        if df.height() == 0 {
            return;
        }
        let names: Vec<&str> = df
            .get_column_names()
            .into_iter()
            .filter(|x| ![redc.GROUP, redc.TIMESTAMP].contains(x))
            .collect();
        if self.header.is_none() || self.header.as_ref().unwrap().height() < names.len() {
            self.header = Some(DataFrame::new(vec![Series::new("name", names)]).unwrap());
        }
        self.base_stats.iter_mut().for_each(|x| x.compute(df));
    }

    /// :return: a data frame with the computed events and metrics statistics (min, max, percentile...)
    fn get_statistics(&mut self) -> LazyFrame {
        let mut stat_data_frames: Vec<DataFrame> = self
            .base_stats
            .iter()
            .map(|x| x.get_stats_values())
            .collect();

        stat_data_frames.insert(0, self.header.clone().unwrap());
        self.stats_df = concat_df_horizontal(&stat_data_frames).unwrap().lazy();
        self.stats_df.clone()
    }
}

#[derive(Clone)]
pub struct SystemDataView<'a, T: Clone, S: MetricCompute + Clone> {
    attributes: ViewAttributes<'a, T, S>,
    summary_group_by: Vec<&'a str>,
    aggregator_columns: Vec<&'a str>,
    aggregator_group_by: Vec<&'a str>,
    details_group_by: Vec<&'a str>,
    details_index: Vec<&'a str>,
    device_filter: Option<Vec<&'a str>>,
    device_filter_mode: &'a str,
    stats: Statistics,
}

impl<'a, T: Clone, S: MetricCompute + Clone> SystemDataView<'a, T, S> {
    fn new(config: ViewAttributes<'a, T, S>) -> Self {
        let device_filter: Option<Vec<&str>> = config.device.as_ref().map(|v| v.exclusions.clone());
        Self {
            attributes: config,
            summary_group_by: vec![redc.NAME],
            aggregator_columns: vec![redc.NAME, redc.SOCKET, redc.UNIT, redc.TSC, redc.VALUE],
            aggregator_group_by: vec![redc.NAME, redc.SOCKET, redc.UNIT],
            details_group_by: vec![redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME],
            details_index: vec![redc.GROUP, redc.TIMESTAMP],
            device_filter,
            device_filter_mode: "exclude",
            stats: Statistics::new(None),
        }
    }
}

impl<T: Clone, S: MetricCompute + Clone> DataView<T, S> for SystemDataView<'_, T, S> {
    type Input = LazyFrame;
    type Output = LazyFrame;

    fn update_summary(
        &self,
        df: Option<Self::Input>,
        event_summary_values: Option<Self::Input>,
    ) -> Option<Self::Output> {
        match self.attributes.view_type {
            ViewType::DETAILS => {
                return df;
            }
            _ => {
                df.as_ref()?;
            }
        }

        self.update_summary_values(
            df,
            event_summary_values,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        )
    }

    fn generate_details(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::DETAILS;

        let df = self.generate_details_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            self.attributes.ref_tsc,
            &self.details_group_by,
            &self.details_index,
            self.attributes.metric_computer.as_ref(),
            self.attributes.constant_values.as_ref(),
            true,
        );
        ViewData::new(attributes, df)
    }

    fn compute_aggregate(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        let df = self.generate_aggregate_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        );

        match df {
            Some(v) => ViewData::new(
                attributes,
                Some(
                    v.sort(
                        [redc.NAME],
                        SortMultipleOptions::default().with_multithreaded(false),
                    )
                    .collect()
                    .unwrap()
                    .lazy(),
                ),
            ),
            None => ViewData::new(attributes, None),
        }
    }

    fn update_statistics(&mut self, df: Option<Self::Input>) -> Option<Self::Output> {
        let details_view_df = df?.collect().unwrap();
        self.stats.compute(&details_view_df);
        Some(self.stats.get_statistics())
    }

    fn generate_summary(&self, event_df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        match self.attributes.view_type {
            ViewType::DETAILS => ViewData::new(attributes, None),
            _ => {
                let df = self.generate_summary_dataframe(
                    event_df,
                    self.attributes.ref_tsc,
                    &self.summary_group_by,
                    self.attributes.metric_computer.as_ref(),
                    self.attributes.constant_values.as_ref(),
                    true,
                );
                let mut res: Option<LazyFrame> = None;
                if df.is_some() {
                    res = Some(
                        concat_df_horizontal(&[
                            DataFrame::new(vec![Series::new("stat", vec!["aggregated"])]).unwrap(),
                            df.unwrap().collect().unwrap(),
                        ])
                        .unwrap()
                        .lazy(),
                    );
                }
                ViewData::new(attributes, res)
            }
        }
    }
}

#[derive(Clone)]
pub struct SocketDataView<'a, T: Clone, S: MetricCompute + Clone> {
    attributes: ViewAttributes<'a, T, S>,
    summary_group_by: Vec<&'a str>,
    aggregator_columns: Vec<&'a str>,
    aggregator_group_by: Vec<&'a str>,
    details_group_by: Vec<&'a str>,
    details_index: Vec<&'a str>,
    device_filter: Option<Vec<&'a str>>,
    device_filter_mode: &'a str,
}

impl<'a, T: Clone, S: MetricCompute + Clone> SocketDataView<'a, T, S> {
    fn new(config: ViewAttributes<'a, T, S>) -> Self {
        let mut config = config;
        let _constant_values = config
            .constant_values
            .as_mut()
            .expect("Constant Values Was Not Initialized");
        _constant_values.insert("system.socket_count".into(), 1.0);
        _constant_values.insert("SOCKET_COUNT".into(), 1.0);
        let device_filter: Option<Vec<&str>> = match config.device {
            Some(ref v) => {
                let mut res = v.exclusions.clone();
                res.push("SYSTEM");
                Some(res)
            }
            None => None,
        };
        Self {
            attributes: config,
            summary_group_by: vec![redc.NAME, redc.SOCKET],
            aggregator_columns: vec![redc.NAME, redc.SOCKET, redc.UNIT, redc.TSC, redc.VALUE],
            aggregator_group_by: vec![redc.NAME, redc.SOCKET, redc.UNIT],
            details_group_by: vec![redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME, redc.SOCKET],
            details_index: vec![redc.GROUP, redc.TIMESTAMP, redc.SOCKET],
            device_filter,
            device_filter_mode: "exclude",
        }
    }
}

impl<T: Clone, S: MetricCompute + Clone> DataView<T, S> for SocketDataView<'_, T, S> {
    type Input = LazyFrame;
    type Output = LazyFrame;

    fn update_summary(
        &self,
        df: Option<Self::Input>,
        event_summary_values: Option<Self::Input>,
    ) -> Option<Self::Output> {
        match self.attributes.view_type {
            ViewType::DETAILS => {
                return df;
            }
            _ => {
                df.as_ref()?;
            }
        }

        self.update_summary_values(
            df,
            event_summary_values,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        )
    }

    fn generate_details(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::DETAILS;

        let df = self.generate_details_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            self.attributes.ref_tsc,
            &self.details_group_by,
            &self.details_index,
            self.attributes.metric_computer.as_ref(),
            self.attributes.constant_values.as_ref(),
            true,
        );
        ViewData::new(attributes, df)
    }

    fn compute_aggregate(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        let df = self.generate_aggregate_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        );

        match df {
            Some(v) => ViewData::new(
                attributes,
                Some(
                    v.sort(
                        [redc.NAME],
                        SortMultipleOptions::default().with_multithreaded(false),
                    )
                    .collect()
                    .unwrap()
                    .lazy(),
                ),
            ),
            None => ViewData::new(attributes, None),
        }
    }

    fn generate_summary(&self, event_df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        match self.attributes.view_type {
            ViewType::DETAILS => ViewData::new(attributes, None),
            _ => {
                let df = self.generate_summary_dataframe(
                    event_df,
                    self.attributes.ref_tsc,
                    &self.summary_group_by,
                    self.attributes.metric_computer.as_ref(),
                    self.attributes.constant_values.as_ref(),
                    true,
                );
                ViewData::new(attributes, df)
            }
        }
    }
}

#[derive(Clone)]
pub struct CoreDataView<'a, T: Clone, S: MetricCompute + Clone> {
    attributes: ViewAttributes<'a, T, S>,
    summary_group_by: Vec<&'a str>,
    aggregator_columns: Vec<&'a str>,
    aggregator_group_by: Vec<&'a str>,
    details_group_by: Vec<&'a str>,
    details_index: Vec<&'a str>,
    device_filter: Option<Vec<&'a str>>,
    device_filter_mode: &'a str,
}

impl<'a, T: Clone, S: MetricCompute + Clone> CoreDataView<'a, T, S> {
    fn new(config: ViewAttributes<'a, T, S>) -> Self {
        let device_filter: Option<Vec<&str>> = config.device.as_ref().map(|v| vec![v.type_name]);
        match config.show_modules {
            true => Self {
                attributes: config,
                summary_group_by: vec![redc.NAME, redc.SOCKET, redc.MODULE, redc.CORE],
                aggregator_columns: vec![
                    redc.NAME,
                    redc.SOCKET,
                    redc.MODULE,
                    redc.CORE,
                    redc.UNIT,
                    redc.TSC,
                    redc.VALUE,
                ],
                aggregator_group_by: vec![
                    redc.NAME,
                    redc.SOCKET,
                    redc.MODULE,
                    redc.CORE,
                    redc.UNIT,
                ],
                details_group_by: vec![
                    redc.GROUP,
                    redc.TIMESTAMP,
                    redc.TSC,
                    redc.NAME,
                    redc.SOCKET,
                    redc.MODULE,
                    redc.CORE,
                ],
                details_index: vec![
                    redc.GROUP,
                    redc.TIMESTAMP,
                    redc.SOCKET,
                    redc.MODULE,
                    redc.CORE,
                ],
                device_filter,
                device_filter_mode: "include",
            },
            false => Self {
                attributes: config,
                summary_group_by: vec![redc.NAME, redc.SOCKET, redc.CORE],
                aggregator_columns: vec![
                    redc.NAME,
                    redc.SOCKET,
                    redc.CORE,
                    redc.UNIT,
                    redc.TSC,
                    redc.VALUE,
                ],
                aggregator_group_by: vec![redc.NAME, redc.SOCKET, redc.CORE, redc.UNIT],
                details_group_by: vec![
                    redc.GROUP,
                    redc.TIMESTAMP,
                    redc.TSC,
                    redc.NAME,
                    redc.SOCKET,
                    redc.CORE,
                ],
                details_index: vec![redc.GROUP, redc.TIMESTAMP, redc.SOCKET, redc.CORE],
                device_filter,
                device_filter_mode: "include",
            },
        }
    }
}

impl<T: Clone, S: MetricCompute + Clone> DataView<T, S> for CoreDataView<'_, T, S> {
    type Input = LazyFrame;
    type Output = LazyFrame;

    fn update_summary(
        &self,
        df: Option<Self::Input>,
        event_summary_values: Option<Self::Input>,
    ) -> Option<Self::Output> {
        match self.attributes.view_type {
            ViewType::DETAILS => {
                return df;
            }
            _ => {
                df.as_ref()?;
            }
        }

        self.update_summary_values(
            df,
            event_summary_values,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        )
    }

    fn generate_details(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::DETAILS;

        let df = self.generate_details_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            self.attributes.ref_tsc,
            &self.details_group_by,
            &self.details_index,
            self.attributes.metric_computer.as_ref(),
            self.attributes.constant_values.as_ref(),
            true,
        );
        ViewData::new(attributes, df)
    }

    fn compute_aggregate(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        let df = self.generate_aggregate_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        );

        match df {
            Some(v) => ViewData::new(
                attributes,
                Some(
                    v.sort(
                        [redc.NAME],
                        SortMultipleOptions::default().with_multithreaded(false),
                    )
                    .collect()
                    .unwrap()
                    .lazy(),
                ),
            ),
            None => ViewData::new(attributes, None),
        }
    }

    fn generate_summary(&self, event_df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        match self.attributes.view_type {
            ViewType::DETAILS => ViewData::new(attributes, None),
            _ => {
                let df = self.generate_summary_dataframe(
                    event_df,
                    self.attributes.ref_tsc,
                    &self.summary_group_by,
                    self.attributes.metric_computer.as_ref(),
                    self.attributes.constant_values.as_ref(),
                    true,
                );
                ViewData::new(attributes, df)
            }
        }
    }
}

#[derive(Clone)]
pub struct ThreadDataView<'a, T: Clone, S: MetricCompute + Clone> {
    attributes: ViewAttributes<'a, T, S>,
    summary_group_by: Vec<&'a str>,
    aggregator_columns: Vec<&'a str>,
    aggregator_group_by: Vec<&'a str>,
    details_group_by: Vec<&'a str>,
    details_index: Vec<&'a str>,
    device_filter: Option<Vec<&'a str>>,
    device_filter_mode: &'a str,
}

impl<'a, T: Clone, S: MetricCompute + Clone> ThreadDataView<'a, T, S> {
    fn new(config: ViewAttributes<'a, T, S>) -> Self {
        let device_filter: Option<Vec<&str>> = config.device.as_ref().map(|v| vec![v.type_name]);
        match config.show_modules {
            true => Self {
                attributes: config,
                summary_group_by: vec![
                    redc.NAME,
                    redc.UNIT,
                    redc.SOCKET,
                    redc.MODULE,
                    redc.CORE,
                    redc.THREAD,
                ],
                aggregator_columns: vec![
                    redc.NAME,
                    redc.SOCKET,
                    redc.MODULE,
                    redc.CORE,
                    redc.THREAD,
                    redc.UNIT,
                    redc.TSC,
                    redc.VALUE,
                ],
                aggregator_group_by: vec![
                    redc.NAME,
                    redc.SOCKET,
                    redc.MODULE,
                    redc.CORE,
                    redc.THREAD,
                    redc.UNIT,
                ],
                details_group_by: vec![
                    redc.GROUP,
                    redc.TIMESTAMP,
                    redc.TSC,
                    redc.NAME,
                    redc.UNIT,
                    redc.MODULE,
                ],
                details_index: vec![redc.GROUP, redc.TIMESTAMP, redc.UNIT, redc.MODULE],
                device_filter,
                device_filter_mode: "include",
            },
            false => Self {
                attributes: config,
                summary_group_by: vec![redc.NAME, redc.UNIT, redc.SOCKET, redc.CORE, redc.THREAD],
                aggregator_columns: vec![
                    redc.NAME,
                    redc.SOCKET,
                    redc.CORE,
                    redc.THREAD,
                    redc.UNIT,
                    redc.TSC,
                    redc.VALUE,
                ],
                aggregator_group_by: vec![
                    redc.NAME,
                    redc.SOCKET,
                    redc.CORE,
                    redc.THREAD,
                    redc.UNIT,
                ],
                details_group_by: vec![redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME, redc.UNIT],
                details_index: vec![redc.GROUP, redc.TIMESTAMP, redc.UNIT],
                device_filter,
                device_filter_mode: "include",
            },
        }
    }
}

impl<T: Clone, S: MetricCompute + Clone> DataView<T, S> for ThreadDataView<'_, T, S> {
    type Input = LazyFrame;
    type Output = LazyFrame;

    fn update_summary(
        &self,
        df: Option<Self::Input>,
        event_summary_values: Option<Self::Input>,
    ) -> Option<Self::Output> {
        match self.attributes.view_type {
            ViewType::DETAILS => {
                return df;
            }
            _ => {
                df.as_ref()?;
            }
        }

        self.update_summary_values(
            df,
            event_summary_values,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        )
    }

    fn generate_details(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::DETAILS;

        let df = self.generate_details_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            self.attributes.ref_tsc,
            &self.details_group_by,
            &self.details_index,
            self.attributes.metric_computer.as_ref(),
            self.attributes.constant_values.as_ref(),
            true,
        );
        ViewData::new(attributes, df)
    }

    fn compute_aggregate(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        let df = self.generate_aggregate_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        );

        match df {
            Some(v) => ViewData::new(
                attributes,
                Some(
                    v.sort(
                        [redc.NAME],
                        SortMultipleOptions::default().with_multithreaded(false),
                    )
                    .collect()
                    .unwrap()
                    .lazy(),
                ),
            ),
            None => ViewData::new(attributes, None),
        }
    }

    fn generate_summary(&self, event_df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        match self.attributes.view_type {
            ViewType::DETAILS => ViewData::new(attributes, None),
            _ => {
                let df = self.generate_summary_dataframe(
                    event_df,
                    self.attributes.ref_tsc,
                    &self.summary_group_by,
                    self.attributes.metric_computer.as_ref(),
                    self.attributes.constant_values.as_ref(),
                    true,
                );
                ViewData::new(attributes, df)
            }
        }
    }
}

#[derive(Clone)]
pub struct UncoreDataView<'a, T: Clone, S: MetricCompute + Clone> {
    attributes: ViewAttributes<'a, T, S>,
    summary_group_by: Vec<&'a str>,
    aggregator_columns: Vec<&'a str>,
    aggregator_group_by: Vec<&'a str>,
    details_group_by: Vec<&'a str>,
    details_index: Vec<&'a str>,
    device_filter: Option<Vec<&'a str>>,
    device_filter_mode: &'a str,
}

impl<'a, T: Clone, S: MetricCompute + Clone> UncoreDataView<'a, T, S> {
    fn new(config: ViewAttributes<'a, T, S>) -> Self {
        let mut config = config;
        let updated_constant_values = config.constant_values.as_mut().unwrap();
        updated_constant_values.insert("system.socket_count".into(), 1.0);
        updated_constant_values.insert("SOCKET_COUNT".into(), 1.0);
        let filtered_keys: Vec<String> = updated_constant_values
            .keys()
            .filter(|x| x.to_lowercase().contains("chas_per_socket"))
            .map(|x| x.to_owned())
            .collect();
        filtered_keys.into_iter().for_each(|x| {
            updated_constant_values.insert(x, 1.0);
        });
        let device_filter: Option<Vec<&str>> = config.device.as_ref().map(|v| vec![v.type_name]);
        Self {
            attributes: config,
            summary_group_by: vec![redc.NAME, redc.UNIT, redc.SOCKET],
            aggregator_columns: vec![redc.NAME, redc.SOCKET, redc.UNIT, redc.TSC, redc.VALUE],
            aggregator_group_by: vec![redc.NAME, redc.SOCKET, redc.UNIT],
            details_group_by: vec![
                redc.GROUP,
                redc.TIMESTAMP,
                redc.SOCKET,
                redc.TSC,
                redc.NAME,
                redc.UNIT,
            ],
            details_index: vec![redc.GROUP, redc.TIMESTAMP, redc.UNIT, redc.SOCKET],
            device_filter,
            device_filter_mode: "include",
        }
    }
}

impl<T: Clone, S: MetricCompute + Clone> DataView<T, S> for UncoreDataView<'_, T, S> {
    type Input = LazyFrame;
    type Output = LazyFrame;

    fn update_summary(
        &self,
        df: Option<Self::Input>,
        event_summary_values: Option<Self::Input>,
    ) -> Option<Self::Output> {
        match self.attributes.view_type {
            ViewType::DETAILS => {
                return df;
            }
            _ => {
                df.as_ref()?;
            }
        }

        self.update_summary_values(
            df,
            event_summary_values,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        )
    }

    fn generate_details(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::DETAILS;

        let df = self.generate_details_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            self.attributes.ref_tsc,
            &self.details_group_by,
            &self.details_index,
            self.attributes.metric_computer.as_ref(),
            self.attributes.constant_values.as_ref(),
            true,
        );
        ViewData::new(attributes, df)
    }

    fn compute_aggregate(&self, df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        let df = self.generate_aggregate_dataframe(
            df,
            self.device_filter.as_deref(),
            self.device_filter_mode,
            &self.aggregator_columns,
            &self.aggregator_group_by,
        );

        match df {
            Some(v) => ViewData::new(
                attributes,
                Some(
                    v.sort(
                        [redc.NAME],
                        SortMultipleOptions::default().with_multithreaded(false),
                    )
                    .collect()
                    .unwrap()
                    .lazy(),
                ),
            ),
            None => ViewData::new(attributes, None),
        }
    }

    fn generate_summary(&self, event_df: Option<Self::Input>) -> ViewData<'_, T, S> {
        let mut attributes = self.attributes.clone();
        attributes.view_type = ViewType::SUMMARY;

        match self.attributes.view_type {
            ViewType::DETAILS => ViewData::new(attributes, None),
            _ => {
                let df = self.generate_summary_dataframe(
                    event_df,
                    self.attributes.ref_tsc,
                    &self.summary_group_by,
                    self.attributes.metric_computer.as_ref(),
                    self.attributes.constant_values.as_ref(),
                    true,
                );
                ViewData::new(attributes, df)
            }
        }
    }
}
