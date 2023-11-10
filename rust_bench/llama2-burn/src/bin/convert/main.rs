use llama::model::*;

use std::{error::Error, process};

use burn_tch::{TchBackend, TchDevice};
use clap::Parser;

use burn::{config::Config, module::Module, tensor::backend::Backend};

use burn::record::{self, BinFileRecorder, HalfPrecisionSettings, Recorder};

fn convert_llama_dump_to_model<B: Backend>(
    dump_path: &str,
    model_name: &str,
    output_dir: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    let (llama, llama_config): (Llama<B>, LlamaConfig) = load_llama_dump(dump_path, device)?;

    save_llama_model_file(llama, &format!("{output_dir}/{model_name}"))?;
    llama_config.save(&format!("{output_dir}/{model_name}.cfg"))?;

    Ok(())
}

fn save_llama_model_file<B: Backend>(
    llama: Llama<B>,
    name: &str,
) -> Result<(), record::RecorderError> {
    BinFileRecorder::<HalfPrecisionSettings>::new().record(llama.into_record(), name.into())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the llama dump.
    #[clap(required = true, help = "Path to the llama dump")]
    dump_path: String,

    /// Name of the output model.
    #[clap(required = true, help = "Name of the output model")]
    model_name: String,

    /// Output directory for the model.
    #[clap(required = true, help = "Output directory for the model")]
    output_dir: String,
}

fn main() {
    type Backend = TchBackend<f32>;

    // might crash if lacking enough GPU memory so use CPU for conversion
    let device = TchDevice::Cpu;

    let args = Args::parse();

    let dump_path = &args.dump_path;
    let model_name = &args.model_name;
    let output_dir = &args.output_dir;

    if let Err(e) =
        convert_llama_dump_to_model::<Backend>(dump_path, model_name, &output_dir, &device)
    {
        eprintln!("Failed to convert llama dump to model: {:?}", e);
        process::exit(1);
    }

    println!("Successfully converted {} to {}", dump_path, model_name);
}
