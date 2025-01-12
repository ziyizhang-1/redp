#![allow(non_snake_case)]
use crate::redp::core::{SliceExt, Vec2Ext};

use polars::prelude::*;
use regex::bytes::Regex;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::from_utf8_unchecked;

static SYSTEM_FEATURES_SECTION: &[u8] = "System Features".as_bytes();
static PROCESSOR_FEATURES_SECTION: &[u8] = "Processor Features".as_bytes();
static UNCORE_UNITS_SECTION: &[u8] = "Uncore Performance Monitoring Units".as_bytes();
static PROCESSOR_MAP_SECTION: &[u8] = "OS Processor <".as_bytes();
static RDT_SECTION: &[u8] = "RDT H/W Support".as_bytes();
static GPU_SECTION: &[u8] = "GPU Information".as_bytes();
static RAM_FEATURES_SECTION: &[u8] = "RAM Features".as_bytes();
static QPI_FEATURES_SECTION: &[u8] = "QPI Link Features".as_bytes();
static IIO_FEATURES_SECTION: &[u8] = "IIO Unit Features".as_bytes();

#[derive(Debug, Clone)]
pub enum ValueType {
    Integer(i32),
    Bool(bool),
    String(String),
    Float(f32),
}

impl From<&ValueType> for f32 {
    #[inline]
    fn from(value: &ValueType) -> Self {
        match value {
            ValueType::Bool(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
            ValueType::Float(v) => *v,
            ValueType::Integer(v) => *v as f32,
            ValueType::String(v) => v.parse::<f32>().unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmonSystemInformationParser {
    // Parse system information stored in EMON data files (emon.dat, emon-v.dat)
    pub socket_map: HashMap<i32, i32>,
    pub core_map: HashMap<i32, i32>,
    pub thread_map: HashMap<i32, i32>,
    pub core_type_map: HashMap<i32, String>,
    pub unique_core_types: Vec<String>,
    pub unique_os_processors: Vec<i32>,
    pub module_map: HashMap<i32, i32>,
    pub has_modules: bool,
    pub ref_tsc: f32,
    attributes: HashMap<String, ValueType>,
    system_features: HashMap<String, ValueType>,
    processor_features: HashMap<String, ValueType>,
    uncore_units: HashMap<String, ValueType>,
    ram_features: HashMap<i32, HashMap<i32, HashMap<i32, i32>>>,
    rdt: HashMap<String, ValueType>,
}

#[allow(unused_variables)]
impl<'a> EmonSystemInformationParser {
    pub fn new(input_file: &'a Path, _ref_tsc_hz: Option<f32>) -> Self {
        let ref_tsc_hz = _ref_tsc_hz.unwrap_or(0.0);
        let mut initial_parser = Self {
            socket_map: HashMap::new(),
            core_map: HashMap::new(),
            thread_map: HashMap::new(),
            core_type_map: HashMap::new(),
            unique_core_types: vec![],
            unique_os_processors: vec![],
            module_map: HashMap::new(),
            has_modules: false,
            ref_tsc: 0.0,
            attributes: HashMap::new(),
            system_features: HashMap::new(),
            processor_features: HashMap::new(),
            uncore_units: HashMap::new(),
            ram_features: HashMap::new(),
            rdt: HashMap::new(),
        };
        let file = OpenOptions::new().read(true).open(input_file);
        let mut parser_state = ParserState::new();
        match file {
            Ok(f) => {
                let mut reader = BufReader::new(f);
                let mut line: Vec<u8> = vec![];
                while let Ok(len) = reader.read_until(0xa, &mut line) {
                    if len > 0 {
                        if let Ok(()) = parser_state.parse(&mut initial_parser, line.trim()) {
                            println!("{}", unsafe { from_utf8_unchecked(line.as_slice().trim()) });
                            break;
                        }
                        line.clear();
                    }
                }
            }
            Err(_) => {
                eprintln!("{:?}: No such file or directory", input_file);
            }
        }

        // Finalize attributes
        match ref_tsc_hz > 0.0 {
            true => {
                initial_parser.ref_tsc = ref_tsc_hz;
            }
            false => {
                let tsc_candidates: Vec<&String> = initial_parser
                    .attributes
                    .keys()
                    .filter(|key| key.starts_with("TSC Freq"))
                    .collect();
                let ref_tsc: f32;
                if !tsc_candidates.is_empty() {
                    match &initial_parser.attributes[tsc_candidates[0]] {
                        ValueType::String(v) => {
                            ref_tsc = v.split_once(' ').unwrap().0.parse().unwrap();
                        }
                        _ => {
                            ref_tsc = 0.0;
                        }
                    }
                } else if initial_parser
                    .attributes
                    .contains_key("Processor Base Freq")
                {
                    match &initial_parser.attributes["Processor Base Freq"] {
                        ValueType::String(v) => {
                            ref_tsc = v.split_once(' ').unwrap().0.parse().unwrap();
                        }
                        _ => {
                            ref_tsc = 0.0;
                        }
                    }
                } else {
                    ref_tsc = 0.0;
                }

                initial_parser.ref_tsc = ref_tsc * 1000000.0;
            }
        }

        initial_parser
    }

    pub fn set_processor_maps(&mut self, mut map_values: Vec<Vec<&str>>) {
        let n_rows = map_values.len();
        let headers = map_values.remove(0);
        let column_values = map_values.transpose();
        let mut series: Vec<Series> = column_values
            .into_iter()
            .enumerate()
            .map(|(i, x)| Series::new(headers[i], &x))
            .collect();
        if !headers.contains(&"Core Type") {
            series.push(Series::new("Core Type", &vec!["core"; n_rows - 1]));
        }

        if !headers.contains(&"Module") {
            series.push(Series::new("Module", &vec!["0"; n_rows - 1]));
        } else {
            self.has_modules = true;
        }
        let mut df = DataFrame::new(series).unwrap();

        unsafe {
            let results = df
                .get_columns_mut()
                .iter_mut()
                .map(|s| {
                    s.strict_cast(&DataType::UInt32)
                        .or(s.strict_cast(&DataType::String))
                })
                .collect::<Result<DataFrame, PolarsError>>();
            if let Ok(df) = results {
                self.socket_map = df
                    .column("OS Processor")
                    .unwrap()
                    .iter()
                    .zip(df.column("Phys. Package").unwrap().iter())
                    .map(|(a, b)| (a.try_extract().unwrap(), b.try_extract().unwrap()))
                    .collect();
                self.core_map = df
                    .column("OS Processor")
                    .unwrap()
                    .iter()
                    .zip(df.column("Core").unwrap().iter())
                    .map(|(a, b)| (a.try_extract().unwrap(), b.try_extract().unwrap()))
                    .collect();
                self.thread_map = df
                    .column("OS Processor")
                    .unwrap()
                    .iter()
                    .zip(df.column("Logical Processor").unwrap().iter())
                    .map(|(a, b)| (a.try_extract().unwrap(), b.try_extract().unwrap()))
                    .collect();
                self.core_type_map = df
                    .column("OS Processor")
                    .unwrap()
                    .iter()
                    .zip(df.column("Core Type").unwrap().iter())
                    .map(|(a, b)| (a.try_extract().unwrap(), b.to_string()))
                    .collect();
                self.module_map = df
                    .column("OS Processor")
                    .unwrap()
                    .iter()
                    .zip(df.column("Module").unwrap().iter())
                    .map(|(a, b)| (a.try_extract().unwrap(), b.try_extract().unwrap()))
                    .collect();
                self.unique_core_types = df
                    .column("Core Type")
                    .unwrap()
                    .unique()
                    .unwrap()
                    .str()
                    .unwrap()
                    .into_no_null_iter()
                    .map(|s| s.to_owned())
                    .collect();
                self.unique_os_processors = self.core_map.keys().copied().collect();
                self.unique_os_processors.sort()
            }
        }
    }
}

#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum CurrState<'a> {
    DefaultState(DefaultState),
    FinalState(FinalState),
    ProcessorMappingState(ProcessorMappingState<'a>),
    SystemFeaturesState(SystemFeaturesState),
    ProcessorFeaturesState(ProcessorFeaturesState),
    UncoreUnitsState(UncoreUnitsState),
    RdtSupportState(RdtSupportState),
    GpuInformationState(GpuInformationState),
    QpiFeaturesState(QpiFeaturesState),
    IioFeaturesState(IioFeaturesState),
    RamFeaturesState(RamFeaturesState),
}

#[allow(dead_code)]
struct ParserState<'a> {
    state: CurrState<'a>,
    INT_LIKE_RE: Regex,
    DOT_SEPARATED_RE: Regex,
    BOOL_VAL_RE: Regex,
    NUMERIC_VAL_RE: Regex,
    DIMM_LOCATION_RE: Regex,
    DIMM_INFO_RE: Regex,
}

impl ParserState<'_> {
    fn new() -> Self {
        Self {
            state: CurrState::DefaultState(DefaultState::new()),
            INT_LIKE_RE: Regex::new(r"\d+[\d,]*").unwrap(),
            DOT_SEPARATED_RE: Regex::new(r"^(?P<name>[^/.]+)\.+(?P<value>[^/.][\s\S]+)$").unwrap(),
            BOOL_VAL_RE: Regex::new(r"^\s*\((?P<name>[\s\S]+)\)\s+\((?P<value>[\s\S]+)\)$")
                .unwrap(),
            NUMERIC_VAL_RE: Regex::new(r"^\s*\((?P<name>[\s\S]+):\s*(?P<value>[\s\S]+)\)$")
                .unwrap(),
            DIMM_LOCATION_RE: Regex::new(r"\((\d+)/(\d+)/(\d+)\)").unwrap(),
            DIMM_INFO_RE: Regex::new(r"\(dimm(?P<id>\d+) info:\s*(?P<value>.*)\)").unwrap(),
        }
    }

    fn adjust_type(&self, value: &str) -> ValueType {
        let bytes = value.as_bytes();
        if self.INT_LIKE_RE.find(bytes).is_some() {
            value
                .replace(',', "")
                .parse::<i32>()
                .map_or(ValueType::String(value.to_string()), |x| {
                    ValueType::Integer(x)
                })
        } else if ["yes", "enabled"].contains(&value.to_lowercase().as_str()) {
            ValueType::Bool(true)
        } else if ["no", "disabled"].contains(&value.to_lowercase().as_str()) {
            ValueType::Bool(false)
        } else {
            ValueType::String(value.to_string())
        }
    }

    fn parse(&mut self, context: &mut EmonSystemInformationParser, line: &[u8]) -> Result<(), ()> {
        match self.state {
            CurrState::DefaultState(ref mut s) => {
                if s.skip_line(line) {
                    return Err(());
                }
                if line.starts_with("Version Info:".as_bytes()) {
                    self.state = CurrState::FinalState(FinalState::new());
                } else if line.starts_with(PROCESSOR_MAP_SECTION) {
                    self.state = CurrState::ProcessorMappingState(ProcessorMappingState::new());
                } else if line.starts_with(SYSTEM_FEATURES_SECTION) {
                    self.state = CurrState::SystemFeaturesState(SystemFeaturesState::new());
                } else if line.starts_with(PROCESSOR_FEATURES_SECTION) {
                    self.state = CurrState::ProcessorFeaturesState(ProcessorFeaturesState::new());
                } else if line.starts_with(UNCORE_UNITS_SECTION) {
                    self.state = CurrState::UncoreUnitsState(UncoreUnitsState::new());
                } else if line.starts_with(RDT_SECTION) {
                    self.state = CurrState::RdtSupportState(RdtSupportState::new());
                } else if line.starts_with(GPU_SECTION) {
                    self.state = CurrState::GpuInformationState(GpuInformationState::new());
                } else if line.starts_with(RAM_FEATURES_SECTION) {
                    self.state = CurrState::RamFeaturesState(RamFeaturesState::new());
                } else if line.starts_with(QPI_FEATURES_SECTION) {
                    self.state = CurrState::QpiFeaturesState(QpiFeaturesState::new());
                } else if line.starts_with(IIO_FEATURES_SECTION) {
                    self.state = CurrState::IioFeaturesState(IioFeaturesState::new());
                } else if line.contains(&b':') {
                    /*
                       # Parse system information with the following format:
                       #   key : value
                       #
                       # Example file section:
                       # ...
                       # NUMA node(s):                    2
                       # NUMA node0 CPU(s):               0-31,64-95
                       # NUMA node1 CPU(s):               32-63,96-127
                       # ...
                    */
                    let line_string: &str = unsafe { from_utf8_unchecked(line) };
                    let (k, v) = line_string.split_once(':').unwrap();
                    let value = self.adjust_type(v.trim());
                    context.attributes.insert(k.trim().to_owned(), value);
                } else {
                    /*
                       # Parse system information with the following format:
                       #   key ......... value
                       #
                       # Example file section:
                       # ...
                       # Device Type ............... Intel(R) Xeon(R) Processor code named Icelake
                       # EMON Database ............. icelake_server
                       # Platform type ............. 125
                       # ...
                    */

                    if let Some(matches) = self.DOT_SEPARATED_RE.captures(line) {
                        let key = unsafe {
                            from_utf8_unchecked(matches.name("name").unwrap().as_bytes()).trim()
                        };
                        let value = unsafe {
                            self.adjust_type(
                                from_utf8_unchecked(matches.name("value").unwrap().as_bytes())
                                    .trim(),
                            )
                        };
                        context.attributes.insert(key.into(), value);
                    }
                }
            }
            CurrState::FinalState(ref mut _s) => {}
            CurrState::GpuInformationState(ref mut _s) => {
                /*
                 * Parses the "GPU Information" section and update system attributes

                    Example file section: ::

                        ...
                        GPU Information:

                        TBD...
                        ...
                */
                if line.is_empty() {
                    self.state = CurrState::DefaultState(DefaultState::new());
                }
            }
            CurrState::IioFeaturesState(ref mut _s) => {
                /*
                 * Parses the "IIO Unit Features" section and update system attributes

                    Example file section: ::

                        ...
                        IIO Unit Features:
                            Package 0 :
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                            Package 1 :
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                                domain:0 bus:0x00 stack:0 mesh: 0
                        ...
                        TBD
                */
                if line.is_empty() {
                    self.state = CurrState::DefaultState(DefaultState::new());
                }
            }
            CurrState::ProcessorFeaturesState(ref mut _s) => {
                /*
                 *  Parses the "Processor Features" section and update system attributes

                    Example file section: ::

                        ...
                        Processor Features:
                            (Thermal Throttling) (Enabled)
                            (Hyper-Threading) (Enabled)
                            (MLC Streamer Prefetching) (Enabled)
                            (MLC Spatial Prefetching) (Enabled)
                            (DCU Streamer Prefetching) (Enabled)
                            (DCU IP Prefetching) (Enabled)
                            (Cores Per Package:   22)
                            (Threads Per Package: 44)
                            (Threads Per Core:    2)
                        ...
                */
                if line.is_empty() {
                    self.state = CurrState::DefaultState(DefaultState::new());
                }

                let mut matches = self.BOOL_VAL_RE.captures(line);
                if matches.is_none() {
                    matches = self.NUMERIC_VAL_RE.captures(line);
                }
                if let Some(v) = matches {
                    unsafe {
                        let value: ValueType = self.adjust_type(
                            from_utf8_unchecked(v.name("value").unwrap().as_bytes()).trim(),
                        );
                        context.processor_features.insert(
                            from_utf8_unchecked(v.name("name").unwrap().as_bytes())
                                .trim()
                                .to_owned(),
                            value,
                        );
                    }
                }
            }
            CurrState::SystemFeaturesState(ref mut _s) => {
                /*
                *  Parses the "System Features" section and update system attributes
                   Example file section: ::
                       ...
                       System Features:
                           (Number of Packages:    1)
                           (Cores Per Package:    16)
                           (Threads Per Package:  24)
                       ...
                */
                if line.is_empty() {
                    self.state = CurrState::DefaultState(DefaultState::new());
                }

                let mut matches = self.BOOL_VAL_RE.captures(line);
                if matches.is_none() {
                    matches = self.NUMERIC_VAL_RE.captures(line);
                }
                if let Some(v) = matches {
                    unsafe {
                        let value: ValueType = self.adjust_type(
                            from_utf8_unchecked(v.name("value").unwrap().as_bytes()).trim(),
                        );
                        context.system_features.insert(
                            from_utf8_unchecked(v.name("name").unwrap().as_bytes())
                                .trim()
                                .to_owned(),
                            value,
                        );
                    }
                }
            }
            CurrState::ProcessorMappingState(ref mut s) => {
                /*
                 *  Parses the "Processor Mapping" section and update system attributes

                     Example file section: ::

                        ...
                        OS Processor <-> Physical/Logical Mapping
                        -----------------------------------------
                            OS Processor	  Phys. Package	      Core	Logical Processor	Core Type	Module
                                0		       0		       0		   0		     bigcore		2
                                1		       0		       0		   0		     smallcore		0
                                2		       0		       1		   0		     smallcore		0
                                3		       0		       2		   0		     smallcore		0
                                4		       0		       3		   0		     smallcore		0
                                5		       0		       0		   0		     smallcore		1
                                6		       0		       1		   0		     smallcore		1
                                7		       0		       2		   0		     smallcore		1
                                8		       0		       3		   0		     smallcore		1
                                9		       0		       0		   1		     bigcore		2
                                10		       0		       0		   0		     bigcore		3
                                11		       0		       0		   1		     bigcore		3
                                12		       0		       0		   0		     bigcore		4
                                13		       0		       0		   1		     bigcore		4
                                14		       0		       0		   0		     bigcore		5
                                15		       0		       0		   1		     bigcore		5
                                16		       0		       0		   0		     bigcore		6
                                17		       0		       0		   1		     bigcore		6
                                18		       0		       0		   0		     bigcore		7
                                19		       0		       0		   1		     bigcore		7
                        -----------------------------------------
                */
                if line == s.MAP_TABLE_SEPARATOR.as_bytes() {
                    match s.is_table_start {
                        true => {
                            s.is_table_start = false;
                        }
                        false => {
                            let _map_values = s
                                .map_values
                                .iter()
                                .map(|s| (*s).iter().map(AsRef::as_ref).collect())
                                .collect();
                            context.set_processor_maps(_map_values);
                            self.state = CurrState::DefaultState(DefaultState::new());
                        }
                    }
                } else {
                    let line_string = unsafe { from_utf8_unchecked(line) };
                    let l: Vec<String> = line_string
                        .split('\t')
                        .map(|s| s.trim().into())
                        .filter(|s| !String::is_empty(s))
                        .collect();
                    s.map_values.push(l);
                }
            }
            CurrState::QpiFeaturesState(ref mut _s) => {
                /*
                 * Parses the "QPI Link Features" section and update system attributes

                    Example file section: ::

                        ...
                        QPI Link Features:
                            Package 0 :
                            Package 1 :
                        ...
                        TBD
                */
                if line.is_empty() {
                    self.state = CurrState::DefaultState(DefaultState::new());
                }
            }
            CurrState::RamFeaturesState(ref mut s) => {
                /*
                 * Parses the "RAM Features" section and stores the information in the ram_features attribute

                    Example file section: ::

                        ...
                        RAM Features:
                            (Package/Memory Controller/Channel)
                            (0/0/0) (Total Number of Ranks on this Channel: 2)
                                (Dimm0 Info: Empty)
                                (Dimm1 Info: Empty)
                            (0/0/1) (Total Number of Ranks on this Channel: 2)
                                (Dimm0 Info: Capacity = 32, # of devices = 32, Device Config = 8Gb(2048Mbx4))
                                (Dimm1 Info: Capacity = 32, # of devices = 32, Device Config = 8Gb(2048Mbx4))
                        ...
                */
                if line.is_empty() {
                    s.finalize_ram_features(context);
                    self.state = CurrState::DefaultState(DefaultState::new());
                }
            }
            CurrState::RdtSupportState(ref mut _s) => {
                /*
                 * Parses the "RDT H/W Support" section and update system attributes

                    Example file section: ::

                        ...
                        RDT H/W Support:
                            L3 Cache Occupancy		: Yes
                            Total Memory Bandwidth	: Yes
                            Local Memory Bandwidth	: Yes
                            L3 Cache Allocation		: Yes
                            L2 Cache Allocation		: No
                            Highest Available RMID	: 255
                            Sample Multiplier		: 65536
                        ...
                */
                if line.is_empty() {
                    self.state = CurrState::DefaultState(DefaultState::new());
                }

                let line_string = unsafe { from_utf8_unchecked(line) };

                let parts: Vec<&str> = line_string.split(':').map(|s| s.trim()).take(2).collect();
                if parts.len() == 2 {
                    let value = self.adjust_type(parts[1]);
                    context.rdt.insert(parts[0].to_owned(), value);
                }
            }
            CurrState::UncoreUnitsState(ref mut _s) => {
                /*
                 * Parses the "Uncore Performance Monitoring Units" section and update system attributes

                    Example file section: ::

                        ...
                        Uncore Performance Monitoring Units:
                            cha             : 32
                            imc             : 8
                            m2m             : 4
                            qpi             : 3
                            r3qpi           : 3
                            iio             : 6
                            irp             : 6
                            pcu             : 1
                            ubox            : 1
                            m2pcie          : 6
                            rdt             : 1
                        ...
                */
                if line.is_empty() {
                    self.state = CurrState::DefaultState(DefaultState::new());
                }

                let line_string = unsafe { from_utf8_unchecked(line) };

                let parts: Vec<&str> = line_string.split(':').map(|s| s.trim()).take(2).collect();
                if parts.len() == 2 {
                    let value = self.adjust_type(parts[1]);
                    context.uncore_units.insert(parts[0].to_owned(), value);
                }
            }
        }

        match self.state {
            CurrState::FinalState(ref _s) => Ok(()),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
struct DefaultState;

impl DefaultState {
    #[inline]
    fn new() -> Self {
        Self
    }
    #[inline]
    fn skip_line(&self, line: &[u8]) -> bool {
        let should_skip = line.starts_with("Copyright".as_bytes())
            || line.starts_with("Application Build Data".as_bytes());
        should_skip
    }
}

#[derive(Debug)]
struct FinalState;

impl FinalState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct ProcessorMappingState<'a> {
    MAP_TABLE_SEPARATOR: &'a str,
    map_values: Vec<Vec<String>>,
    is_table_start: bool,
}

impl ProcessorMappingState<'_> {
    #[inline]
    fn new() -> Self {
        Self {
            MAP_TABLE_SEPARATOR: "-----------------------------------------",
            map_values: vec![],
            is_table_start: true,
        }
    }
}

#[derive(Debug)]
struct SystemFeaturesState;

impl SystemFeaturesState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct ProcessorFeaturesState;

impl ProcessorFeaturesState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct UncoreUnitsState;

impl UncoreUnitsState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct RdtSupportState;

impl RdtSupportState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct GpuInformationState;

impl GpuInformationState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct QpiFeaturesState;

impl QpiFeaturesState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct IioFeaturesState;

impl IioFeaturesState {
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
struct RamFeaturesState {
    dimm_location_map: HashMap<(i32, i32, i32), i32>,
}

impl RamFeaturesState {
    #[inline]
    fn new() -> Self {
        Self {
            dimm_location_map: HashMap::new(),
        }
    }

    fn finalize_ram_features(&self, context: &mut EmonSystemInformationParser) {
        if self.dimm_location_map.is_empty() {
            return;
        }

        let (mut sockets, mut controlers_per_socket, mut channels_per_controller) = self
            .dimm_location_map
            .keys()
            .copied()
            .reduce(|acc, e| (acc.0.max(e.0), acc.1.max(e.1), acc.2.max(e.2)))
            .unwrap();

        sockets += 1;
        controlers_per_socket += 1;
        channels_per_controller += 1;

        let mut ram_features: HashMap<i32, HashMap<i32, HashMap<i32, i32>>> = HashMap::new();
        for socket in 0..sockets {
            ram_features.insert(socket, HashMap::new());
            for controller in 0..controlers_per_socket {
                ram_features
                    .get_mut(&socket)
                    .unwrap()
                    .insert(controller, HashMap::new());
                for channel in 0..channels_per_controller {
                    let value = *self
                        .dimm_location_map
                        .get(&(socket, controller, channel))
                        .unwrap();
                    ram_features
                        .get_mut(&socket)
                        .unwrap()
                        .get_mut(&controller)
                        .unwrap()
                        .insert(channel, value);
                }
            }
        }

        context.ram_features = ram_features;
    }
}

#[derive(Debug)]
pub struct EmonSystemInformationAdaptor<'a> {
    emon_sys_info: &'a EmonSystemInformationParser,
    ref_tsc: ValueType,
}

impl<'a> EmonSystemInformationAdaptor<'a> {
    // Adapt EMON system information data to `MetricComputer` symbol table format (Dict[str, Any])
    pub fn new(emon_sys_info: &'a EmonSystemInformationParser) -> Self {
        Self {
            emon_sys_info,
            ref_tsc: ValueType::Float(emon_sys_info.ref_tsc),
        }
    }

    pub fn get_symbol_table(&self) -> HashMap<String, f32> {
        /*! return: a symbol table for `MetricComputer` from EMOM system information */
        let mut symbol_table: HashMap<String, f32> = HashMap::new();
        symbol_table.insert("system.tsc_freq".into(), (&self.ref_tsc).into());
        symbol_table.insert("SYSTEM_TSC_FREQ".into(), (&self.ref_tsc).into());
        symbol_table.insert(
            "system.socket_count".into(),
            self.emon_sys_info
                .processor_features
                .get("Number of Packages")
                .or(self.emon_sys_info.system_features.get("Number of Packages"))
                .unwrap_or(&ValueType::Float(f32::NAN))
                .into(),
        );
        symbol_table.insert(
            "SOCKET_COUNT".into(),
            self.emon_sys_info
                .processor_features
                .get("Number of Packages")
                .or(self.emon_sys_info.system_features.get("Number of Packages"))
                .unwrap_or(&ValueType::Float(f32::NAN))
                .into(),
        );
        symbol_table.insert("DURATIONTIMEINSECONDS".into(), 1.0);
        symbol_table.insert("DURATIONTIMEINMILLISECONDS".into(), 1000.0);
        symbol_table.insert(
            "system.sockets[0].cores.count".into(),
            self.emon_sys_info
                .processor_features
                .get("Cores Per Package")
                .or(self.emon_sys_info.system_features.get("Cores Per Package"))
                .unwrap_or(&ValueType::Float(f32::NAN))
                .into(),
        );
        symbol_table.insert(
            "CORES_PER_SOCKET".into(),
            self.emon_sys_info
                .processor_features
                .get("Cores Per Package")
                .or(self.emon_sys_info.system_features.get("Cores Per Package"))
                .unwrap_or(&ValueType::Float(f32::NAN))
                .into(),
        );
        symbol_table.insert(
            "system.sockets[0].cpus.count".into(),
            self.emon_sys_info
                .processor_features
                .get("Threads Per Package")
                .or(self
                    .emon_sys_info
                    .system_features
                    .get("Threads Per Package"))
                .unwrap_or(&ValueType::Float(f32::NAN))
                .into(),
        );
        symbol_table.insert(
            "THREADS_PER_SOCKET".into(),
            self.emon_sys_info
                .processor_features
                .get("Threads Per Package")
                .or(self
                    .emon_sys_info
                    .system_features
                    .get("Threads Per Package"))
                .unwrap_or(&ValueType::Float(f32::NAN))
                .into(),
        );
        let ht_on: f32 = self
            .emon_sys_info
            .processor_features
            .get("Hyper-Threading")
            .unwrap_or(&ValueType::Bool(false))
            .into();
        symbol_table.insert(
            "THREADS_PER_CORE".into(),
            self.emon_sys_info
                .processor_features
                .get("Threads Per Core")
                .unwrap_or(&ValueType::Float(ht_on + 1.0))
                .into(),
        );
        symbol_table.insert(
            "system.sockets[0][0].size".into(),
            self.emon_sys_info
                .processor_features
                .get("Threads Per Core")
                .unwrap_or(&ValueType::Float(ht_on + 1.0))
                .into(),
        );
        if self.emon_sys_info.uncore_units.contains_key("cha") {
            symbol_table.insert(
                "CHAS_PER_SOCKET".into(),
                self.emon_sys_info
                    .uncore_units
                    .get("cha")
                    .unwrap_or(&ValueType::Float(f32::NAN))
                    .into(),
            );
            symbol_table.insert(
                "system.cha_count/system.socket_count".into(),
                self.emon_sys_info
                    .uncore_units
                    .get("cha")
                    .unwrap_or(&ValueType::Float(f32::NAN))
                    .into(),
            );
            symbol_table.insert(
                "chas_per_socket".into(),
                self.emon_sys_info
                    .uncore_units
                    .get("cha")
                    .unwrap_or(&ValueType::Float(f32::NAN))
                    .into(),
            );
        }
        symbol_table
    }
}
