use llama::model::*;
use llama::token::LlamaTokenizer;

use num_traits::cast::ToPrimitive;
use std::io::prelude::*;
use std::{env, error::Error, fs::OpenOptions, io, time::Instant};

use burn_tch::{TchBackend, TchDevice};

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Data, Tensor},
};
use env_logger::Env;
use log::{error, info};

use burn::record::{self, BinFileRecorder, HalfPrecisionSettings, Recorder};

fn init_logger() {
    env::set_var("RUST_LOG", "info");
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .format_module_path(false)
        .format_level(true)
        .init();
}

fn load_llama<B: Backend>(model_name: &str) -> Result<(Llama<B>, LlamaConfig), Box<dyn Error>> {
    let config = LlamaConfig::load(&format!("{model_name}.cfg"))?;
    let llama = load_llama_model_file(&config, model_name)?;

    Ok((llama, config))
}

fn load_llama_model_file<B: Backend>(
    config: &LlamaConfig,
    filename: &str,
) -> Result<Llama<B>, record::RecorderError> {
    BinFileRecorder::<HalfPrecisionSettings>::new()
        .load(filename.into())
        .map(|record| config.init().load_record(record))
}

fn sample_llama<B: Backend>(
    llama: &Llama<B>,
    tokenizer: &LlamaTokenizer,
    prompt: &str,
    n_tokens: usize,
) -> String {
    let device = llama.devices()[0].clone();

    let mut tokens = tokenizer.encode(prompt, true, false);
    let mut text = String::new();

    for _ in 0..n_tokens {
        let token_tensor = Tensor::from_ints(Data::from_usize(Data::new(
            tokens.iter().map(|&t| t as usize).collect(),
            [tokens.len()].into(),
        )))
        .unsqueeze::<2>()
        .to_device(&device);

        let out = llama.forward(token_tensor);

        let [_, n_token, _] = out.dims();
        let last_row: Tensor<B, 1> = out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);

        let token_id = last_row.argmax(0).into_scalar().to_i64().unwrap();

        tokens.push(token_id);

        let token_text = tokenizer.decode(&[token_id], true);

        text += &token_text;
    }

    text
}

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

use std::process;

fn main() {
    type Backend = TchBackend<Elem>;
    init_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 {
        error!(
            "Usage: {} <model_name> <tokenizer_filepath> <prompt> <n_tokens> <device> <repetitions> <log_file>",
            args[0]
        );
        std::process::exit(1);
    }

    let model_name = &args[1];
    let tokenizer_filepath = &args[2];
    let prompt = &args[3];
    let n_tokens: usize = args[4].parse().unwrap_or_else(|_| {
        error!("Error: Invalid number of tokens");
        process::exit(1);
    });

    // Specify device based on command line argument
    let device_param = &args[5];
    let device = if device_param == "cpu" {
        TchDevice::Cpu
    } else if device_param == "gpu" {
        #[cfg(not(target_os = "macos"))]
        let device = TchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = TchDevice::Mps;

        device
    } else {
        error!("Error: Invalid device parameter (must be 'cpu' or 'gpu')");
        process::exit(1);
    };

    let repetitions: usize = args[6].parse().unwrap_or_else(|_| {
        error!("Error: Invalid number of repetitions");
        std::process::exit(1);
    });

    let log_file: String = args[7].parse().unwrap_or_else(|_| {
        error!("Error: Invalid log filename.");
        std::process::exit(1);
    });

    let tokenizer = match LlamaTokenizer::new(tokenizer_filepath) {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            error!("Failed to load tokenizer: {:?}", e);
            process::exit(1);
        }
    };

    let (llama, _llama_config): (Llama<Backend>, LlamaConfig) = match load_llama(model_name) {
        Ok((llama, llama_config)) => (llama, llama_config),
        Err(e) => {
            error!("Failed to load llama model: {:?}", e);
            process::exit(1);
        }
    };

    let llama = llama.to_device(&device);

    let mut tokens_per_second_values = Vec::new();
    info!("Running burn benchmark");

    for r in 0..repetitions {
        let start_time = Instant::now();
        let width = repetitions.to_string().len();
        let message = format!("Running repetition [{:0width$}/{}]", r + 1, repetitions);
        info!("{}", message);
        let _ = sample_llama(&llama, &tokenizer, prompt, n_tokens);
        let elapsed_time = start_time.elapsed();
        let elapsed_seconds = elapsed_time.as_secs_f64();
        let tokens_per_second = (n_tokens) as f64 / elapsed_seconds;
        tokens_per_second_values.push(tokens_per_second);
    }

    let average_tokens_per_second =
        tokens_per_second_values.iter().sum::<f64>() / repetitions as f64;

    let standard_deviation = if repetitions > 1 {
        let mean = tokens_per_second_values.iter().sum::<f64>() / repetitions as f64;
        let variance = tokens_per_second_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / repetitions as f64;
        variance.sqrt()
    } else {
        0.0
    };

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)
        .unwrap();
    let mut file_writer = io::BufWriter::new(file);
    let elem_type = if cfg!(feature = "f16") {
        "float16"
    } else {
        "float32"
    };
    info!(
        "burn, {} : {:.2} ± {:.2}",
        elem_type, average_tokens_per_second, standard_deviation
    );
    writeln!(
        file_writer,
        "{}",
        format!(
            "burn, {} : {:.2} ± {:.2}",
            elem_type, average_tokens_per_second, standard_deviation
        )
    )
    .unwrap();
}
