import argparse
import logging
import os

import torch
from optimum.nvidia import AutoModelForCausalLM

# Some points to note:
# - the conversion is super simple, and it assumes batch size to be 1 and
#   num beams to be 1
# - it also assumes a standard prompt length of 512 tokens


def build_engine(hf_model_path: str, out_model_dir: str, torch_dtype: str):
    if not os.path.isdir(out_model_dir):
        os.makedirs(out_model_dir, exist_ok=True)

    dtype_mapper = {"float16": torch.float16, "float32": torch.float32}

    try:
        logging.info("Starting to build the model engine")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=hf_model_path,
            max_batch_size=1,
            max_prompt_length=512,
            num_beams=1,
            torch_dtype=dtype_mapper[torch_dtype],
        )

        model.save_pretrained(save_directory=out_model_dir)
    except Exception as e:
        logging.info(f"Error: {e}")
        os.rmdir(out_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HF Optimum builder engine CLI")
    parser.add_argument(
        "--hf_dir",
        type=str,
        help="Hugging Face model weights path",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        help="The output engine dir",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        help="The precision in which it will be saved. Supported: 'float16' and 'float32",
    )

    args = parser.parse_args()
    build_engine(
        hf_model_path=args.hf_dir, out_model_dir=args.out_dir, torch_dtype=args.dtype
    )
