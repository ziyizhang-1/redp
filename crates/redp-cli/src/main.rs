use clap::{Parser, Subcommand};

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "jemalloc")]
use jemallocator::Jemalloc;
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;
use redp_core::cli::writers::{csv::CsvWrite, excel_writer::ExcelWrite};
use redp_core::redp::core::types::{Device, MetricDefinition};
use redp_core::redp::core::views::{
    DataAccumulator, ViewAggregationLevel, ViewCollection, ViewData, ViewGenerator, ViewType,
};
use redp_core::redp::parsers::chunk_reader::{
    get_block_offsets, update_unique_uncore_devices, ChunkRead,
};
use redp_core::redp::parsers::emon::{Block, EmonParser};
use redp_core::redp::parsers::metrics;
use redp_core::SymbolTable;

use chrono::Local;
use polars::prelude::DataFrame;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use rust_xlsxwriter::Workbook;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// CLI Tool Rust Emon Data Parser
#[derive(Parser, Debug)]
#[command(author = "Ziyi Zhang", version = "0.4.1", about = "Emon Data Processor CLI\n
                                                                                 
                                                                                 
██████╗ ███████╗██████╗ ██████╗ 
██╔══██╗██╔════╝██╔══██╗██╔══██╗
██████╔╝█████╗  ██║  ██║██████╔╝
██╔══██╗██╔══╝  ██║  ██║██╔═══╝ 
██║  ██║███████╗██████╔╝██║     
╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝           
                                                                                 
\nWritten by Ziyi Zhang
\nContact Info: ziyi.zhang@intel.com", long_about = None)]
struct Cmds {
    #[command(subcommand)]
    pub commands: SubCommands,
}

#[derive(Subcommand, Debug)]
enum SubCommands {
    Parse(Parse),
    // Replay(Replay), // WIP
}

#[derive(Parser, Debug)]
struct Parse {
    // The input file (emon.dat) path
    #[arg()]
    path: String,
    // The .xml path
    #[arg(long, short)]
    xml: String,
    // add socket view
    #[arg(long, short, action, default_value_t = false)]
    socket_view: bool,
    // add core view
    #[arg(long, short, action, default_value_t = false)]
    core_view: bool,
    // add thread view
    #[arg(long, short, action, default_value_t = false)]
    thread_view: bool,
    // add uncore view
    #[arg(long, short, action, default_value_t = false)]
    uncore_view: bool,
    // no details
    #[arg(long, short, action, default_value_t = false)]
    no_details: bool,
    // chunk size
    #[arg(long)]
    chunk_size: Option<u32>,
    // mem saver mode
    #[arg(long, short, action, default_value_t = false)]
    mem: bool,
}

#[derive(Parser, Debug)]
struct Replay {
    // The input file (emon.dat) path
    #[arg()]
    path: String,
    #[arg(long, short, default_value_t = 10)]
    refresh_rate: u8,
    #[arg(long, short, default_value_t = 30)]
    elapse_time: u32,
    // The .xml path
    #[arg(long, short)]
    xml: String,
    // replay system view [Default]
    #[arg(long, short, action, default_value_t = true)]
    system_view: bool,
    // replay socket view
    #[arg(long, short, action, default_value_t = false)]
    socket_view: bool,
    // replay core view
    #[arg(long, short, action, default_value_t = false)]
    core_view: bool,
    // replay thread view
    #[arg(long, short, action, default_value_t = false)]
    thread_view: bool,
    // replay uncore view
    #[arg(long, short, action, default_value_t = false)]
    uncore_view: bool,
}

#[allow(non_snake_case)]
fn main() {
    let cmds = Cmds::parse();
    let now = Instant::now();
    match cmds.commands {
        SubCommands::Parse(ref args) => {
            let path = Path::new(&args.path);
            let xml_path = Path::new(&args.xml);

            let offsets =
                get_block_offsets(rayon::iter::Either::Left(path), r"==========|Version Info:")
                    .unwrap();
            let num_blocks = offsets.len();

            // Default MAX_THREADS=24
            let MAX_THREADS: usize = match env::var("MAX_THREADS") {
                Ok(v) => v.parse::<usize>().unwrap(),
                Err(_) => {
                    if num_blocks < 100 {
                        24
                    } else if num_blocks < 500 {
                        48
                    } else {
                        60
                    }
                }
            };
            let POLARS_MAX_THREADS: usize = match env::var("POLARS_MAX_THREADS") {
                Ok(v) => v.parse::<usize>().unwrap(),
                Err(_) => match args.no_details {
                    true => 4,
                    false => MAX_THREADS,
                },
            };
            let POOL_MAX_THREADS: usize = match env::var("POOL_MAX_THREADS") {
                Ok(v) => v.parse::<usize>().unwrap(),
                Err(_) => MAX_THREADS,
            };

            env::set_var("POLARS_MAX_THREADS", POLARS_MAX_THREADS.to_string());
            rayon::ThreadPoolBuilder::new()
                .num_threads(POOL_MAX_THREADS)
                .build_global()
                .unwrap();

            let bytes = std::fs::read(path).unwrap();

            let chunks = bytes
                .read_in_chunks(Some(offsets), None, args.chunk_size)
                .unwrap();

            let mut parser = EmonParser::new(path, None, Some(chrono::Local), 0.0);
            let metrics = metrics::parse(xml_path);

            let symbol_table = SymbolTable::new(&parser.system_info, "core", None);
            let requested_aggregation_levels = get_requested_aggregation_levels(args);

            let mut devices: Vec<Device<'_, ViewAggregationLevel, Vec<MetricDefinition>>> =
                vec![Device::new(
                    "\"core\"",
                    None,
                    Some(requested_aggregation_levels),
                    Some(metrics.clone()),
                )];

            let mut unique_uncore_devices = std::collections::HashSet::new();

            if args.uncore_view {
                update_unique_uncore_devices(chunks[0].inner, &mut unique_uncore_devices).unwrap();
                unique_uncore_devices.iter().for_each(|device| {
                    devices.push(Device::new(
                        device,
                        None,
                        Some(vec![ViewAggregationLevel::UNCORE]),
                        Some(metrics.clone()),
                    ))
                })
            }

            let view_collection: ViewCollection<'_, ViewAggregationLevel, Vec<MetricDefinition>> =
                initialize_views(args, Some(symbol_table.symbols), &parser, &devices);
            let view_generator: ViewGenerator<'_, ViewAggregationLevel, Vec<MetricDefinition>> =
                ViewGenerator::new(Some(view_collection), std::marker::PhantomData);

            let mut data_accumulator: DataAccumulator<
                '_,
                ViewAggregationLevel,
                Vec<MetricDefinition>,
            > = DataAccumulator::new(&view_generator);
            let da_ptr = &mut data_accumulator
                as *mut DataAccumulator<'_, ViewAggregationLevel, Vec<MetricDefinition>>;

            let mut details_dst: HashMap<String, DataFrame> = HashMap::new();
            let details_ptr =
                &std::sync::Mutex::new(&mut details_dst as *mut HashMap<String, DataFrame>)
                    as *const std::sync::Mutex<*mut HashMap<String, DataFrame>>;

            let chunk_iterator = parser.event_reader(
                &view_generator,
                da_ptr,
                details_ptr,
                None,
                None,
                None,
                None,
                chunks,
                args.no_details,
            );

            let blocks: Vec<Block<'_, Local>> = ParallelIterator::collect(chunk_iterator);

            match args.mem {
                true => {
                    blocks
                        .into_iter()
                        .par_bridge()
                        .for_each(move |block| block.update());
                }
                false => {
                    std::thread::scope(|s| {
                        blocks.into_iter().for_each(move |block| {
                            s.spawn(move || block.update());
                        });
                    });
                }
            }

            let summary_views: HashMap<
                String,
                ViewData<'_, ViewAggregationLevel, Vec<MetricDefinition>>,
            > = data_accumulator.generate_summary_views();

            let mut workbook = Workbook::new();
            summary_views.into_iter().for_each(|(name, view)| {
                let worksheet = workbook.add_worksheet();
                match worksheet.set_name(&name) {
                    Ok(_) => {}
                    Err(_) => {
                        worksheet
                            .set_name(
                                name.strip_prefix("__edp_")
                                    .expect("Expect prefix with __edp_"),
                            )
                            .unwrap();
                    }
                }
                worksheet.write_view(view).unwrap();
            });

            match args.no_details {
                true => {}
                false => {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(details_dst.len())
                        .build()
                        .unwrap();
                    details_dst.into_par_iter().for_each(|(name, df)| {
                        let path = name.clone() + ".csv";
                        if df.to_csv(Path::new(&path), args.mem).is_ok() {
                            println!(
                                "redp: details saved to {:?}",
                                env::current_dir().unwrap().to_string_lossy() + "/" + path.as_str()
                            );
                        }
                    });
                }
            };

            workbook.save(Path::new("summary.xlsx")).unwrap();
            println!(
                "redp: summaries saved to {:?}",
                env::current_dir().unwrap().to_string_lossy() + "/summary.xlsx"
            );
            println!(
                "redp: EMON data processing completed in {:#?}",
                now.elapsed()
            );
        }
    };
}

fn get_requested_aggregation_levels(args: &Parse) -> Vec<ViewAggregationLevel> {
    let mut requested_aggregation_levels = vec![ViewAggregationLevel::SYSTEM];
    if args.socket_view {
        requested_aggregation_levels.push(ViewAggregationLevel::SOCKET);
    }
    if args.core_view {
        requested_aggregation_levels.push(ViewAggregationLevel::CORE);
    }
    if args.thread_view {
        requested_aggregation_levels.push(ViewAggregationLevel::THREAD);
    }

    requested_aggregation_levels
}

fn initialize_views<'a>(
    args: &Parse,
    constant_values: Option<HashMap<String, f32>>,
    emon_parser: &EmonParser<Local>,
    devices: &[Device<'a, ViewAggregationLevel, Vec<MetricDefinition>>],
) -> ViewCollection<'a, ViewAggregationLevel, Vec<MetricDefinition>> {
    let mut view_collection: ViewCollection<'a, ViewAggregationLevel, Vec<MetricDefinition>> =
        ViewCollection::new();
    for device in devices.iter() {
        for agg_level in device.aggregation_levels.as_ref().unwrap().iter() {
            let mut view_name_template = format!(
                "__edp{}_{}_view_summary",
                device.decorate_label("_", "").unwrap_or("".to_owned()),
                String::from(agg_level)
            );
            view_collection.add_view(
                view_name_template,
                ViewType::SUMMARY,
                *agg_level,
                Some(device.clone()),
                emon_parser.system_info.ref_tsc,
                emon_parser.system_info.has_modules,
                device.metric_computer.clone(),
                constant_values.clone(),
                emon_parser.event_info.clone(),
            );

            if !args.no_details {
                view_name_template = format!(
                    "__edp{}_{}_view_details",
                    device.decorate_label("_", "").unwrap_or("".to_owned()),
                    String::from(agg_level)
                );
                view_collection.add_view(
                    view_name_template,
                    ViewType::DETAILS,
                    *agg_level,
                    Some(device.clone()),
                    emon_parser.system_info.ref_tsc,
                    emon_parser.system_info.has_modules,
                    device.metric_computer.clone(),
                    constant_values.clone(),
                    emon_parser.event_info.clone(),
                );
            }
        }
    }

    view_collection
}

#[test]
fn compute_metric() {
    use polars::df;
    use polars::lazy::dsl::Expr;
    use polars::prelude::LazyFrame;
    use redp_core::define;
    use redp_core::redp::core::Arithmetic;

    let m1 = define!(|a, b, c, d, e, f, g, h, i, j| 100
        * ([
            ((((28 * (((a) / b) * c / (1000000000) / (d / 1000)))
                - (3 * (((a) / b) * c / (1000000000) / (d / 1000))))
                * (e * (f / (f + g)))
                + ((27 * (((a) / b) * c / (1000000000) / (d / 1000)))
                    - (3 * (((a) / b) * c / (1000000000) / (d / 1000))))
                    * (h))
                * (1 + (i / (j)) / 2)
                / (a)),
            (1)
        ]
        .min));
    let m2 = define!(|a, b, c, d, e, f, g, h, i, j| 100
        * ([
            ((16 * [(0), (a - b)].max + (a / c) * ((10) * d + ([(e - 0), (f - 0)].min))) / (e)),
            (1)
        ]
        .min));
    let m3 = define!(|a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r| 100
        * (a / ((b / (c + d + b + e)) * (f))));

    let df = df!("a" => (1..=50000).map(|x| if x==1 {None} else {Some(x as f32)}).collect::<Vec<Option<f32>>>(), "b" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "c" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "d" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(),
                            "e" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "f" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "g" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "h" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(),
                            "i" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "j" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "k" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "l" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(),
                            "m" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "n" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "o" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "p" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(),
                            "q" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "r" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "s" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "t" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(),
                            "u" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "v" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "w" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "x" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(),
                            "y" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>(), "z" => (1..=50000).map(|x| x as f32).collect::<Vec<f32>>()
                        ).unwrap();
    // rayon::ThreadPoolBuilder::new().num_threads(56).build_global().unwrap();
    // Arithmetic in lazy mode (Automatically parallelized when collect)
    let now = Instant::now();
    let out_lazy: Vec<Expr> = ([m1.clone(), m2.clone(), m3.clone()]
        .into_iter()
        .cycle()
        .take(544)
        .collect::<Vec<MetricDefinition>>())
    .iter()
    .enumerate()
    .map(|(idx, m)| m.arithmetic(&df).alias(idx.to_string().as_str()))
    .collect();
    let out_df = LazyFrame::default()
        .with_columns(out_lazy)
        .collect()
        .unwrap();
    println!("Elapsed in: {:?}", now.elapsed());
    println!("{:#?}", out_df);
}
