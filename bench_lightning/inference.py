# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import Any, Literal, Optional

import lightning as L
import litgpt.utils as utils
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from litgpt import GPT, Config, PromptStyle, Tokenizer
from litgpt.prompts import has_prompt_style, load_prompt_style


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None
) -> torch.Tensor:
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0:
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(
    model: GPT, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    logits = model(x, input_pos)
    next = sample(logits, **kwargs)
    return next.to(dtype=x.dtype)


@torch.inference_mode()
def _generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
        )

    device = prompt.device
    tokens = [prompt]
    input_pos = torch.tensor([T], device=device)
    token = next_token(
        model,
        torch.arange(0, T, device=device),
        prompt.view(1, -1),
        temperature=temperature,
        top_k=top_k,
    ).clone()
    tokens.append(token)
    for _ in range(2, max_returned_tokens - T + 1):
        token = next_token(
            model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k
        ).clone()
        tokens.append(token)
        if token == eos_id:
            break
        input_pos = input_pos.add_(1)
    return torch.cat(tokens)


@torch.inference_mode()
def load_model(
    checkpoint_dir: str,
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]
    ] = None,
    precision: Optional[str] = None,
    compile: bool = False,
):
    plugins = None
    precision = precision or utils.get_default_supported_precision(training=False)

    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    utils.check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )

    with fabric.init_module(empty_init=True):
        model = GPT(config)

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        # NOTE: Hardcoding this part only for benchmark
        model.max_seq_length = 1024
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")

    model = fabric.setup_module(model)
    utils.load_checkpoint(fabric, model, checkpoint_path)
    return model, tokenizer, prompt_style, fabric


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt_style,
    fabric,
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 50,
    temperature: float = 0.8,
) -> None:
    prompt = prompt_style.apply(prompt)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    L.seed_everything(1234)
    for i in range(num_samples):
        y = _generate(
            model,
            encoded,
            max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
        )

        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()

    # Now decode here

    output = y.detach().cpu().tolist()
    output = output[prompt_length:]

    return {"output_tokens": output, "num_output_tokens": len(output)}
