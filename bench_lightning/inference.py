# Script is taken LitGPT
# repo: https://github.com/Lightning-AI/lit-gpt.git

import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch._dynamo.config
import torch._inductor.config

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT  # noqa E402


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
def generate(
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
