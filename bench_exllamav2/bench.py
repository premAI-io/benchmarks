import time
from dataclasses import dataclass

import torch
from exllamav2 import ExLlamaV2Cache, model_init
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler


@dataclass
class Config:
    model_dir: str
    prompt: str
    tokens: int
    length: int = 2048
    rope_scale: float = 1.0
    rope_alpha: float = 1.0
    no_flash_attn: bool = False
    low_mem: bool = False
    gpu_split: str = None


config = Config(
    model_dir="/home/paperspace/workspace/benchmarks/models/llama-2-7b-exllamav2",
    prompt="hello world",
    tokens=100,
)

# model init logic

model, tokenizer = model_init.init(config, allow_auto_split=True)

if not model.loaded:
    cache = ExLlamaV2Cache(model)
    model.load_autosplit(cache)
    cache = None

if config.prompt:
    with torch.inference_mode():
        cache = ExLlamaV2Cache(model)
        ids = tokenizer.encode(config.prompt)
        tokens_prompt = ids.shape[-1]
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        generator.warmup()

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.15
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

        start = time.time()
        output = generator.generate_simple(
            config.prompt, settings, config.tokens, token_healing=True
        )
        torch.cuda.synchronize()
        delta = time.time() - start

    print(output)
    print()
    cache = None
