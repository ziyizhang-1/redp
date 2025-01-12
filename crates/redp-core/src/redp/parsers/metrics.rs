use crate::redp::core::types::MetricDefinition;
use quick_xml::de;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct MetricRoot {
    pub metric: Option<Vec<Metric>>,
}

#[derive(Debug, Deserialize)]
pub struct Metric {
    #[serde(rename = "@name")]
    pub name: String,
    // throughput_metric_name: Option<String>,
    #[serde(rename = "$value")]
    pub definition: Vec<Definition>,
}

#[derive(Debug, Deserialize)]
pub enum Definition {
    #[serde(rename = "event")]
    Event {
        #[serde(rename = "@alias")]
        alias: String,
        #[serde(rename = "$value")]
        name: String,
    },
    #[serde(rename = "constant")]
    Constant {
        #[serde(rename = "@alias")]
        alias: String,
        #[serde(rename = "$value")]
        value: String,
    },
    #[serde(rename = "formula")]
    Formula {
        #[serde(rename = "@socket")]
        socket: Option<String>,
        #[serde(rename = "$value")]
        form: String,
    },
    #[serde(rename = "throughput-metric-name")]
    ThroughputMetricName {
        #[serde(rename = "$value")]
        name: String,
    },
    #[serde(rename = "description")]
    Description {
        #[serde(rename = "$value")]
        value: Option<String>,
    },
}

impl From<Metric> for MetricDefinition {
    fn from(value: Metric) -> Self {
        let name = value.name;
        let mut throughput_metric_name: String = String::from("");
        let mut description: String = String::from("");
        let mut constants: HashMap<String, String> = HashMap::new();
        let mut events: HashMap<String, String> = HashMap::new();
        let mut latencies: HashMap<String, String> = HashMap::new();
        let mut formula: String = String::from("");
        for item in value.definition {
            match item {
                Definition::Event { alias, name } => {
                    if !name.contains("retire_latency") {
                        events.insert(alias, name);
                    } else {
                        latencies.insert(alias, name);
                    }
                }
                Definition::Constant { alias, value } => {
                    if value.trim() == "system.cha_count/system.socket_count" {
                        constants.insert(alias, String::from("chas_per_socket"));
                    } else {
                        constants.insert(alias, value);
                    }
                }
                Definition::Formula { socket, form } => {
                    if socket.is_none() {
                        if name == "metric_EDP EMON Sampling time (seconds)" {
                            constants.insert("samples".to_owned(), "$processed_samples".to_owned());
                            formula = form.replace(' ', "") + "/samples";
                        } else {
                            formula = form.replace(' ', "");
                        }
                    }
                }
                Definition::ThroughputMetricName { name } => {
                    if throughput_metric_name.is_empty() {
                        throughput_metric_name = name;
                    }
                }
                Definition::Description { value } => {
                    if description.is_empty() && value.is_some() {
                        description = value.unwrap();
                    }
                }
            }
        }
        Self::new(
            name,
            throughput_metric_name,
            description,
            String::from(""),
            formula,
            events,
            constants,
            latencies,
            None,
        )
    }
}

pub fn parse(path: &Path) -> Vec<MetricDefinition> {
    let reader = BufReader::new(File::open(path).unwrap());
    let root: MetricRoot = de::from_reader(reader).unwrap();
    let metrics: Vec<MetricDefinition> =
        root.metric.unwrap().into_iter().map(|x| x.into()).collect();
    metrics
}
