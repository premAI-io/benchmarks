// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use env_logger::Env;
use log::info;
use rand::Rng;
use std::io::prelude::*;
use std::{
    env,
    fs::{self, OpenOptions},
    io,
    path::PathBuf,
};

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    use_flash_attn: bool,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Number of repetitions
    #[arg(long, default_value_t = 2)]
    repetitions: usize,

    #[arg(long)]
    log_file: String,
}

fn init_logger() {
    env::set_var("RUST_LOG", "info");
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .format_module_path(false)
        .format_level(true)
        .init();
}

// Function to get all files with a specific extension in a directory
fn get_files_with_extension(
    directory: &str,
    extension: &str,
) -> Result<Vec<String>, std::io::Error> {
    let entries = fs::read_dir(directory)?;

    let filenames: Vec<String> = entries
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.ends_with(extension) {
                        return Some(entry.path().to_str().unwrap().to_owned());
                    }
                }
            }
            None
        })
        .collect();

    Ok(filenames)
}

fn load_llama_model(
    local_weights: &str,
    use_flash_attn: bool,
    no_kv_cache: bool,
    dtype: DType,
    device: &Device,
) -> Result<(Llama, PathBuf, model::Cache)> {
    let tokenizer_filename = (local_weights.to_owned() + "tokenizer.json").into();

    let config_filename: String = (local_weights.to_owned() + "config.json").into();
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(&config_filename)?)?;
    let config = config.into_config(use_flash_attn);

    let filenames = get_files_with_extension(local_weights, ".safetensors")?;

    let cache = model::Cache::new(!no_kv_cache, dtype, &config, device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
    Ok((Llama::load(vb, &config)?, tokenizer_filename, cache))
}

fn generate_random_numbers(n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}
fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    init_logger();

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("float16") => DType::F16,
        Some("bfloat16") => DType::BF16,
        Some("float32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());

    let repetitions = args.repetitions;
    let mut tokens_per_second = Vec::with_capacity(repetitions);
    let seeds = generate_random_numbers(repetitions);
    info!("Running candle benchmark");

    for r in 0..repetitions {
        let width = repetitions.to_string().len();
        let message = format!("Running repetition [{:0width$}/{}]", r + 1, repetitions);
        info!("{}", message);
        let (llama, tokenizer_filename, mut cache) = load_llama_model(
            &args.local_weights,
            args.use_flash_attn,
            args.no_kv_cache,
            dtype,
            &device,
        )?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);
        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let mut logits_processor = LogitsProcessor::new(seeds[r], args.temperature, args.top_p);
        let start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut token_generated: f64 = 0.0;
        for index in 0..args.sample_len {
            let context_size = if cache.use_kv_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
            let logits = llama.forward(&input, index_pos, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            token_generated += 1.0;
            tokens.push(next_token);

            if Some(next_token) == eos_token_id {
                break;
            }
        }
        let dt = start_gen.elapsed();
        tokens_per_second.push(token_generated / dt.as_secs_f64());
    }

    let average_tokens_per_second = tokens_per_second.iter().sum::<f64>() / repetitions as f64;

    let standard_deviation = if repetitions > 1 {
        let mean = tokens_per_second.iter().sum::<f64>() / repetitions as f64;
        let variance = tokens_per_second
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / repetitions as f64;
        variance.sqrt()
    } else {
        0.0
    };
    info!(
        "candle, {:?} : {:.2} ± {:.2}",
        dtype, average_tokens_per_second, standard_deviation
    );

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(args.log_file)
        .unwrap();
    let mut file_writer = io::BufWriter::new(file);
    writeln!(
        file_writer,
        "{}",
        format!(
            "candle, {:?} : {:.2} ± {:.2}",
            dtype, average_tokens_per_second, standard_deviation
        )
    )
    .unwrap();

    Ok(())
}
