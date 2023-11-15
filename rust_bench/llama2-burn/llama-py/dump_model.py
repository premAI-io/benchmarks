import torch
from pathlib import Path
import json
import argparse
import logging
import sys

import dump
from model import Transformer, ModelArgs
import tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_dir):
    tok = tokenizer.Tokenizer(model_path=str(model_dir / "tokenizer.model"))
    checkpoints = sorted((model_dir).glob("*.pth"))
    if len(checkpoints) == 0:
        raise ValueError(f"No checkpoint files found in {model_dir}")
    
    weights = [torch.load(filename, map_location="cpu") for filename in checkpoints]
    with open(model_dir / "params.json", "r") as f:
        params = json.loads(f.read())
    
    model_args: ModelArgs = ModelArgs(
        max_batch_size=1,
        **params,
    )
    model_args.vocab_size = tok.n_words
    model = Transformer(model_args)
    model.load_state_dict(concat_weights(weights), strict=False)
    model.max_seq_len = model.tok_embeddings.weight.shape[0]
    logger.info('Loaded model')

    return model


def concat_weights(models):
    def convert(name) -> torch.Tensor:
        disk_tensors = [model[name] for model in models]
        if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
            return disk_tensors[0]
        axis = 1 if name.startswith('tok_embeddings.') or name.endswith('.attention.wo.weight') or name.endswith('.feed_forward.w2.weight') else 0
        return disk_tensors[0].cat(*disk_tensors[1:], dim=axis)
    return {name: convert(name) for name in {name: None for model in models for name in model}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and dump transformer model.')
    parser.add_argument('--model-dir', type=Path, required=True, help='Path to the directory containing the model checkpoints')
    parser.add_argument('--output-dir', type=Path, required=True, help='Path to the directory where to dump the model.')

    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    # Check if the model-dir/params directory already exists
    params_dir = output_dir / "params"
    if params_dir.is_dir():
        logger.info(f"The {params_dir} directory already exists. Model dump will not be performed.")
        sys.exit(0)

    # Check that the model dir contains the required files
    if not (model_dir / "params.json").is_file() or not (model_dir / "tokenizer.model").is_file() or not any(model_dir.glob("*.pth")):
        logger.error("The model directory must contain params.json, tokenizer.model, and at least one .pth file")
        sys.exit(1)


    try:
        logger.info(f"Loading model from {model_dir}")
        llama = load_model(model_dir)

        logger.info('Dumping model...')
        dump.save_transformer(llama, params_dir)
        logger.info(f'Dump saved in {params_dir} folder.')
    except Exception as e:
        logger.error(f"An error occurred: {e}")