import argparse
import os
import logging
from collections import defaultdict
from typing import List
import shutil

import torch
from safetensors.torch import load_file, save_file


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"The file size different is more than 1%:\n - {sf_filename}: {sf_size}\n - {pt_filename}: {pt_size}"
        )


def rename(pt_filename: str) -> str:
    filename, _ = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local


def copy_file(src: str, dest: str):
    try:
        shutil.copy(src, dest)
        logging.info(f"Copying {src} to {dest}")
    except FileNotFoundError:
        logging.warning(f"{src} not found. Skipping copy.")


def convert_file(pt_filename: str, sf_filename: str):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    # Adjust sf_filename to ensure correct formatting
    sf_filename = os.path.join(
        os.path.dirname(sf_filename), os.path.basename(rename(pt_filename))
    )

    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_multi(input_dir: str, output_dir: str) -> List[str]:
    if os.path.exists(output_dir):
        logging.warning(f"{output_dir} already exists!")
        return []
    else:
        os.mkdir(output_dir)

    config_src = os.path.join(input_dir, "config.json")
    tokenizer_src = os.path.join(input_dir, "tokenizer.json")

    if not os.path.exists(config_src) or not os.path.exists(tokenizer_src):
        logging.warning(f"{config_src} or {tokenizer_src} not found. Skipping copy.")
        return []
    else:
        copy_file(config_src, output_dir)
        copy_file(tokenizer_src, output_dir)

    filenames = [file for file in os.listdir(input_dir) if file.endswith(".bin")]

    local_filenames = []
    for filename in filenames:
        pt_filename = os.path.join(input_dir, filename)

        sf_filename = rename(pt_filename)
        sf_filename = os.path.join(output_dir, os.path.basename(sf_filename))

        logging.info(f"Converting {pt_filename} to {sf_filename}")
        convert_file(pt_filename, sf_filename)
        local_filenames.append(sf_filename)

    return local_filenames


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Convert .bin files to .safetensors")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing .bin files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory for .safetensors files",
    )
    args = parser.parse_args()

    output_filenames = convert_multi(args.input_dir, args.output_dir)

    logging.info("Conversion successful. Output files:")
    for filename in output_filenames:
        logging.info(filename)
