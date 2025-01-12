#![allow(non_snake_case)]

pub mod cli;
pub mod redp;
pub(crate) mod utils;

use redp::parsers::emon_system_information::{
    EmonSystemInformationAdaptor, EmonSystemInformationParser,
};

use std::collections::HashMap;

pub struct SymbolTable {
    pub symbols: HashMap<String, f32>,
    pub latency_symbols: HashMap<String, f32>,
}

impl SymbolTable {
    #[allow(unused_variables)]
    pub fn new(
        system_info: &EmonSystemInformationParser,
        core_type: &str,
        latency_file_map: Option<HashMap<&str, &str>>,
    ) -> Self {
        let symbols = EmonSystemInformationAdaptor::new(system_info).get_symbol_table();
        Self {
            symbols,
            latency_symbols: HashMap::default(),
        }
    }
}
