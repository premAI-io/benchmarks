# This code is been inspired from Sliver267's implementation
# https://github.com/Silver267/pytorch-to-safetensor-converter/blob/main/convert_to_safetensor.py

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm


class TorchBinToSafeTensorsConverter:
    @classmethod
    def convert(cls, folder_path: str, delete_bins: Optional[bool] = True) -> None:
        """Converts pytorch .bin files to .safetensors
        Args:
            folder_path (str): The path to the huggingface model folder.
            delete_bins Optional[bool]: Whether to delete the pytorch weights or not. Defaults to True
        Returns:
            None
        """
        instance = cls()
        for filename in os.listdir(folder_path):
            if filename == "pytorch_model.bin":
                instance.convert_single(folder_path, delete=delete_bins)
                sys.exit(0)
        instance.convert_multi(folder_path, delete_bins)

    def rename(self, pt_filename: str) -> str:
        filename, _ = os.path.splitext(pt_filename)
        local = f"{filename}.safetensors"
        local = local.replace("pytorch_model", "model")
        return local

    def shared_pointers(self, tensors):
        ptrs = defaultdict(list)
        for k, v in tensors.items():
            ptrs[v.data_ptr()].append(k)
        failing = []
        for ptr, names in ptrs.items():
            if len(names) > 1:
                failing.append(names)
        return failing

    def check_file_size(self, sf_filename: str, pt_filename: str):
        sf_size = os.stat(sf_filename).st_size
        pt_size = os.stat(pt_filename).st_size

        if (sf_size - pt_size) / pt_size > 0.01:
            raise RuntimeError(
                f"""The file size different is more than 1%:
            - {sf_filename}: {sf_size}
            - {pt_filename}: {pt_size}
            """
            )

    def convert_file(
        self,
        pt_filename: str,
        sf_filename: str,
    ):
        loaded = torch.load(pt_filename, map_location="cpu")
        if "state_dict" in loaded:
            loaded = loaded["state_dict"]
        shared = self.shared_pointers(loaded)
        for shared_weights in shared:
            for name in shared_weights[1:]:
                loaded.pop(name)

        # For tensors to be contiguous
        loaded = {k: v.contiguous().half() for k, v in loaded.items()}

        dirname = os.path.dirname(sf_filename)
        os.makedirs(dirname, exist_ok=True)
        save_file(loaded, sf_filename, metadata={"format": "pt"})
        self.check_file_size(sf_filename, pt_filename)
        reloaded = load_file(sf_filename)
        for k in loaded:
            pt_tensor = loaded[k]
            sf_tensor = reloaded[k]
            if not torch.equal(pt_tensor, sf_tensor):
                raise RuntimeError(f"The output tensors do not match for key {k}")

    def convert_single(self, folder_path: str, delete: Optional[bool] = False) -> None:
        pytorch_filename = "pytorch_model.bin"
        safetensor_filename = os.path.join(folder_path, "model.safetensors")
        self.convert_file(pytorch_filename, safetensor_filename)
        if delete:
            os.remove(pytorch_filename)
        return

    def convert_multi(self, folder: str, delprv: bool):
        filename = "pytorch_model.bin.index.json"
        with open(os.path.join(folder, filename), "r") as f:
            data = json.load(f)

        filenames = set(data["weight_map"].values())
        local_filenames = []
        for filename in tqdm(filenames):
            pt_filename = filename

            sf_filename = self.rename(pt_filename)
            sf_filename = os.path.join(folder, sf_filename)
            self.convert_file(
                os.path.join(folder, pt_filename), os.path.join(folder, sf_filename)
            )
            local_filenames.append(os.path.join(folder, sf_filename))
            if delprv:
                os.remove(os.path.join(folder, pt_filename))

        index = os.path.join(folder, "model.safetensors.index.json")
        with open(index, "w") as f:
            newdata = {k: v for k, v in data.items()}
            newmap = {k: self.rename(v) for k, v in data["weight_map"].items()}
            newdata["weight_map"] = newmap
            json.dump(newdata, f, indent=4)
        local_filenames.append(index)
        if delprv:
            os.remove("pytorch_model.bin.index.json")
        return


parser = argparse.ArgumentParser(
    description="Convert pytorch .bin files to .safetensors"
)
parser.add_argument(
    "folder_path", type=str, help="Path to the huggingface model folder"
)
parser.add_argument(
    "--delete_bins",
    action="store_true",
    help="Whether to delete the pytorch weights or not (default is True)",
)
args = parser.parse_args()

TorchBinToSafeTensorsConverter.convert(args.folder_path, delete_bins=args.delete_bins)
