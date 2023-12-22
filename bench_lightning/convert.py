# Parts of script is taken from LitGPT by LightningAI
# repo: https://github.com/Lightning-AI/lit-gpt.git

import gc
import json
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from lightning.fabric.utilities.load import \
    _NotYetLoadedTensor as NotYetLoadedTensor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config  # noqa: E402
from lit_gpt.utils import incremental_save, lazy_load  # noqa: E402


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(
    param: Union[torch.Tensor, NotYetLoadedTensor],
    name: str,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if (
        dtype is not None
        and type(dtype) is not NotYetLoadedTensor
        and dtype != param.dtype
    ):
        print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{l}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{l}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{l}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{l}.norm_2.weight",
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{l}.norm_2.bias",
        "model.norm.weight": "transformer.ln_f.weight",
        "model.norm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config._mlp_class == "LLaMAMoE":
        weight_map.update(
            {
                "model.layers.{}.block_sparse_moe.gate.weight": "transformer.h.{l}.mlp.gate.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "transformer.h.{l}.mlp.experts.{e}.fc_1.weight",  # noqa: E501
                "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "transformer.h.{l}.mlp.experts.{e}.fc_2.weight",  # noqa: E501
                "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "transformer.h.{l}.mlp.experts.{e}.proj.weight",  # noqa: E501
            }
        )
    elif config._mlp_class == "LLaMAMLP":
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{l}.mlp.fc_1.weight",
                "model.layers.{}.mlp.up_proj.weight": "transformer.h.{l}.mlp.fc_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{l}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, l = layer_template(name, 2)  # noqa: E741
            e = None
            if "block_sparse_moe.experts" in name:
                from_name, e = layer_template(from_name, 5)
            qkv = qkv_weights.setdefault(l, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param
            elif "k_proj" in name:
                qkv[1] = param
            elif "v_proj" in name:
                qkv[2] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l=l, e=e)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype)
        k = load_param(k, f"layer {i} k", dtype)
        v = load_param(v, f"layer {i} v", dtype)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    model_name = "Llama-2-7b-hf"
    config = Config.from_name(model_name)
    config_dict = asdict(config)
    print(f"Model config {config_dict}")
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(config_dict, json_config)

    qkv_weights = {}
    copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        bin_files = set(checkpoint_dir.glob("*.bin"))
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")

    with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end
        for bin_file in sorted(bin_files):
            print("Processing", bin_file)
            hf_weights = lazy_load(bin_file)
            copy_fn(sd, hf_weights, saver=saver, dtype=dtype)
        gc.collect()
        print("Saving converted checkpoint")
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)
