import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from tinygrad.helpers import CI, dtypes, getenv
from tinygrad.jit import JIT_SUPPORTED_DEVICE, TinyJit
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.state import load_state_dict, safe_load, torch_load
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor

from python_bench.benchmark import Benchmark

logging.getLogger("tinygrad").setLevel(logging.ERROR)
np.set_printoptions(linewidth=200)


MAX_CONTEXT = 1024


# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[: (dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return Tensor.stack([Tensor.cos(freqs), Tensor.sin(freqs)], dim=-1).reshape(
        1, end, 1, dim // 2, 2
    )


# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
    a, b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
    ro = a * c - b * d
    co = a * d + b * c
    return ro.cat(co, dim=-1)


def apply_rotary_emb(xq, xk, freqs_cis) -> tuple[Tensor, Tensor]:
    assert (
        freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[1]
    ), f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == 5 and len(xk.shape) == 5 and len(freqs_cis.shape) == 5
    c, d = (
        freqs_cis[:, : xq.shape[1], :, :, 0:1],
        freqs_cis[:, : xq.shape[1], :, :, 1:2],
    )
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.reshape(bs, seqlen, n_kv_heads, 1, head_dim)
        .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor):
        # TODO: convert to float?
        return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight


class Attention:
    def __init__(self, dim, n_heads, n_kv_heads, linear=Linear):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

    def __call__(
        self,
        x: Tensor,
        start_pos: Variable | int,
        freqs_cis: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
        xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
        xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        bsz, seqlen, n_heads, head_dim = xq.shape

        # create kv cache
        if not hasattr(self, "cache_k"):
            self.cache_k, self.cache_v = Tensor.zeros(
                bsz, MAX_CONTEXT, self.n_kv_heads, self.head_dim
            ), Tensor.zeros(bsz, MAX_CONTEXT, self.n_kv_heads, self.head_dim)

        keys = self.cache_k.shrink((None, (0, start_pos), None, None)).cat(xk, dim=1)
        values = self.cache_v.shrink((None, (0, start_pos), None, None)).cat(xv, dim=1)

        # update the cache
        self.cache_k.assign(
            keys.pad(
                (None, (0, MAX_CONTEXT - start_pos - seqlen), None, None)
            ).contiguous()
        ).realize()
        self.cache_v.assign(
            values.pad(
                (None, (0, MAX_CONTEXT - start_pos - seqlen), None, None)
            ).contiguous()
        ).realize()

        keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)

        xq, keys, values = (
            xq.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
        )
        attn = (
            xq.scaled_dot_product_attention(keys, values, mask)
            .transpose(1, 2)
            .reshape(bsz, seqlen, -1)
        )
        return self.wo(attn)


class FeedForward:
    def __init__(
        self, dim, hidden_dim, multiple_of, linear=Linear, ffn_dim_multiplier=None
    ):
        # TODO: what is this?
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = linear(dim, hidden_dim, bias=False)
        self.w2 = linear(hidden_dim, dim, bias=False)
        self.w3 = linear(dim, hidden_dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))


class TransformerBlock:
    def __init__(
        self,
        dim,
        multiple_of,
        n_heads,
        n_kv_heads,
        norm_eps,
        linear=Linear,
        ffn_dim_multiplier=None,
    ):
        self.attention = Attention(dim, n_heads, n_kv_heads, linear)
        self.feed_forward = FeedForward(
            dim, 4 * dim, multiple_of, linear, ffn_dim_multiplier
        )
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def __call__(
        self,
        x: Tensor,
        start_pos: Variable | int,
        freqs_cis: Tensor,
        mask: Tensor | None,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return (h + self.feed_forward(self.ffn_norm(h))).realize()


class Transformer:
    def __init__(
        self,
        dim,
        multiple_of,
        n_heads,
        n_layers,
        norm_eps,
        vocab_size,
        device,
        linear=Linear,
        max_batch_size=32,
        max_seq_len=1024,
        ffn_dim_multiplier=None,
        n_kv_heads=None,
        rope_theta=10000,
    ):
        self.JIT = getenv("JIT", 0 if CI else int(device in JIT_SUPPORTED_DEVICE))
        self.layers = [
            TransformerBlock(
                dim,
                multiple_of,
                n_heads,
                n_kv_heads,
                norm_eps,
                linear,
                ffn_dim_multiplier,
            )
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim, norm_eps)
        self.tok_embeddings = Embedding(vocab_size, dim)
        self.output = linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(
            dim // n_heads, max_seq_len * 2, rope_theta
        )
        self.forward_jit = TinyJit(self.forward)

    def forward(
        self, tokens: Tensor, start_pos: Variable | int, temperature: float = 0.0
    ):
        _bsz, seqlen = tokens.shape
        freqs_cis = self.freqs_cis.shrink(
            (None, (start_pos, start_pos + seqlen), None, None, None)
        )
        mask = (
            Tensor.full(
                (1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=dtypes.float32
            )
            .triu(start_pos + 1)
            .realize()
            if seqlen > 1
            else None
        )

        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        logits = self.output(self.norm(h))
        return (logits[:, -1, :] / (temperature + 1e-10)).softmax().flatten().realize()

    def __call__(self, tokens: Tensor, start_pos: Variable, temperature: float = 0.0):
        # TODO: better way to handle the first call v.s. the rest?
        if tokens.shape[0:2] == (1, 1) and self.JIT:
            assert start_pos > 0
            return self.forward_jit(
                tokens,
                Variable("start_pos", 1, MAX_CONTEXT).bind(start_pos),
                temperature,
            )
        return self.forward(tokens, start_pos, temperature)


# **** files and arguments ****
MODEL_PARAMS = {
    "1": {
        "7B": {
            "args": {
                "dim": 4096,
                "multiple_of": 256,
                "n_heads": 32,
                "n_layers": 32,
                "norm_eps": 1e-06,
                "vocab_size": 32000,
            },
            "files": 1,
        },
        "13B": {
            "args": {
                "dim": 5120,
                "multiple_of": 256,
                "n_heads": 40,
                "n_layers": 40,
                "norm_eps": 1e-06,
                "vocab_size": 32000,
            },
            "files": 2,
        },
        "30B": {
            "args": {
                "dim": 6656,
                "multiple_of": 256,
                "n_heads": 52,
                "n_layers": 60,
                "norm_eps": 1e-06,
                "vocab_size": 32000,
            },
            "files": 4,
        },
        "65B": {
            "args": {
                "dim": 8192,
                "multiple_of": 256,
                "n_heads": 64,
                "n_layers": 80,
                "norm_eps": 1e-05,
                "vocab_size": 32000,
            },
            "files": 8,
        },
    },
    "2": {
        "7B": {
            "args": {
                "dim": 4096,
                "multiple_of": 256,
                "n_heads": 32,
                "n_layers": 32,
                "norm_eps": 1e-05,
                "vocab_size": 32000,
            },
            "files": 1,
        },
        "13B": {
            "args": {
                "dim": 5120,
                "multiple_of": 256,
                "n_heads": 40,
                "n_layers": 40,
                "norm_eps": 1e-05,
                "vocab_size": 32000,
            },
            "files": 2,
        },
        "70B": {
            "args": {
                "dim": 8192,
                "multiple_of": 4096,
                "ffn_dim_multiplier": 1.3,
                "n_heads": 64,
                "n_kv_heads": 8,
                "n_layers": 80,
                "norm_eps": 1e-05,
                "vocab_size": 32000,
            },
            "files": 8,
        },
    },
    "code": {
        "7B": {
            "args": {
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016,
            },
            "files": 1,
        },
        "7B-Python": {
            "args": {
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000,
            },
            "files": 1,
        },
        "7B-Instruct": {
            "args": {
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016,
            },
            "files": 1,
        },
        "13B": {
            "args": {
                "dim": 5120,
                "n_layers": 40,
                "n_heads": 40,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016,
            },
            "files": 2,
        },
        "13B-Python": {
            "args": {
                "dim": 5120,
                "n_layers": 40,
                "n_heads": 40,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000,
            },
            "files": 2,
        },
        "13B-Instruct": {
            "args": {
                "dim": 5120,
                "n_layers": 40,
                "n_heads": 40,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32016,
            },
            "files": 2,
        },
        "34B": {
            "args": {
                "dim": 8192,
                "n_layers": 48,
                "n_heads": 64,
                "n_kv_heads": 8,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000,
            },
            "files": 4,
        },
        "34B-Python": {
            "args": {
                "dim": 8192,
                "n_layers": 48,
                "n_heads": 64,
                "n_kv_heads": 8,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000,
            },
            "files": 4,
        },
        "34B-Instruct": {
            "args": {
                "dim": 8192,
                "n_layers": 48,
                "n_heads": 64,
                "n_kv_heads": 8,
                "multiple_of": 256,
                "ffn_dim_multiplier": 1.0,
                "norm_eps": 1e-5,
                "rope_theta": 1000000,
                "vocab_size": 32000,
            },
            "files": 4,
        },
    },
}


# **** helper functions ****
def concat_weights(models, device):
    def convert(name) -> Tensor:
        disk_tensors = [model[name] for model in models]
        if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
            return disk_tensors[0].to(device=device)
        axis = (
            1
            if name.startswith("tok_embeddings.")
            or name.endswith(".attention.wo.weight")
            or name.endswith(".feed_forward.w2.weight")
            else 0
        )
        lazy_tensors = [data.to(device=device) for data in disk_tensors]
        return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)

    return {
        name: convert(name)
        for name in {name: None for model in models for name in model}
    }


def load(fn: str):
    if fn.endswith(".index.json"):
        with open(fn) as fp:
            weight_map = json.load(fp)["weight_map"]
        parts = {
            n: load(str(Path(fn).parent / Path(n).name))
            for n in set(weight_map.values())
        }
        return {k: parts[n][k] for k, n in weight_map.items()}
    elif fn.endswith(".safetensors"):
        return safe_load(fn)
    else:
        return torch_load(fn)


def convert_from_huggingface(weights, model):
    keymap = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        **{
            f"model.layers.{layer}.input_layernorm.weight": f"layers.{layer}.attention_norm.weight"
            for layer in range(len(model.layers))
        },
        **{
            f"model.layers.{layer}.self_attn.{x}_proj.weight": f"layers.{layer}.attention.w{x}.weight"
            for x in ["q", "k", "v", "o"]
            for layer in range(len(model.layers))
        },
        **{
            f"model.layers.{layer}.post_attention_layernorm.weight": f"layers.{layer}.ffn_norm.weight"
            for layer in range(len(model.layers))
        },
        **{
            f"model.layers.{layer}.mlp.{x}_proj.weight": f"layers.{layer}.feed_forward.w{y}.weight"
            for x, y in {"gate": "1", "down": "2", "up": "3"}.items()
            for layer in range(len(model.layers))
        },
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    return {keymap[k]: v for k, v in weights.items() if ".rotary_emb." not in k}


class AbsmaxQuantizedLinear:
    def __init__(self, in_features, out_features, bias=False):
        assert not bias
        self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
        self.scale = Tensor.ones(out_features, dtype=dtypes.half)

    def __call__(self, x):
        return x.dot(self.weight.cast(dtype=dtypes.half).T * self.scale)

    @staticmethod
    def quantize(tensors):
        new_tensors = {}
        for name, v in tensors.items():
            if (
                "feed_forward" in name
                or ("attention.w") in name
                or name == "output.weight"
            ):
                scale = v.abs().max(axis=1) / 127.0
                int8_weight = (v.T / scale).T.cast(dtype=dtypes.int8)
                new_tensors[name] = int8_weight
                new_tensors[name.replace("weight", "scale")] = scale
            else:
                new_tensors[name] = v
        return new_tensors


class LLaMa:
    @staticmethod
    def build(
        model_path,
        tokenizer_path,
        device,
        model_gen="1",
        model_size="7B",
        quantize=False,
    ):
        from sentencepiece import SentencePieceProcessor

        sp_model = SentencePieceProcessor(model_file=str(tokenizer_path))
        assert (
            sp_model.vocab_size()
            == MODEL_PARAMS[model_gen][model_size]["args"]["vocab_size"]
        ), f"{sp_model.vocab_size()=} not equal to {MODEL_PARAMS[model_gen][model_size]['args']['vocab_size']}"

        params = MODEL_PARAMS[model_gen][model_size]
        model = (
            Transformer(**params["args"], device=device, linear=AbsmaxQuantizedLinear)
            if quantize
            else Transformer(**params["args"], device=device)
        )

        if model_path.is_dir():
            weights = concat_weights(
                [
                    load(filename)
                    for filename in [
                        f"{model_path}/consolidated.{i:02d}.pth"
                        for i in range(params["files"])
                    ]
                ],
                device=device,
            )
        else:
            weights = load(str(model_path))
        if "model.embed_tokens.weight" in weights:
            weights = convert_from_huggingface(weights, model)

        if quantize:
            weights = AbsmaxQuantizedLinear.quantize(weights)
            for _, v in weights.items():
                v.realize()
        load_state_dict(model, weights, strict=False)

        return LLaMa(model, sp_model)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


class TinyGradBenchmark(Benchmark):
    def __init__(
        self, model_path, device, quantize, gen="2", temperature=0.7, model_size="7B"
    ):
        super().__init__(model_path)
        self.model = None
        self.quantize = quantize
        self.model_gen = gen
        self.temperature = temperature
        self.model_size = model_size
        self.device = device

    def load_model(self) -> Benchmark:
        self.model = LLaMa.build(
            Path(os.path.join(self.model_path, "pytorch_model.bin.index.json")),
            Path(os.path.join(self.model_path, "tokenizer.model")),
            model_gen=self.model_gen,
            model_size=self.model_size,
            quantize=self.quantize,
            device=self.device,
        )
        return self

    def run_model(self, prompt, max_tokens) -> float:
        Tensor.no_grad = True
        toks = [self.model.tokenizer.bos_id()] + self.model.tokenizer.encode(prompt)
        start_pos = 0
        outputted = self.model.tokenizer.decode(toks)

        new_toks = [self.model.tokenizer.bos_id()] + self.model.tokenizer.encode(
            outputted
        )
        assert toks == new_toks[: len(toks)]
        toks = new_toks
        assert outputted == self.model.tokenizer.decode(toks)
        times = []
        for _ in range(max_tokens):
            start = time.time()
            probs = self.model.model(
                Tensor([toks[start_pos:]]), start_pos, self.temperature
            ).realize()
            times.append(time.time() - start)
            probs_np = probs.numpy()
            tok = int(np.random.choice(len(probs_np), p=probs_np))
            start_pos = len(toks)
            toks.append(tok)
            cur = self.model.tokenizer.decode(toks)
            outputted = cur
        return len(times) / sum(times)
