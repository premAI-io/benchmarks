import os 
import argparse
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from inference import generate

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import (
    get_default_supported_precision, 
    load_checkpoint, 
    check_valid_checkpoint_dir
) 

from lightning.fabric.plugins import BitsandbytesPrecision
import lightning as L 

logging.getLogger("lightning_ai").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Assumption: "Although quantization in Lightning AI supports both gptq and normal with bitsandbytes."
# For benchmarking purposes we are doing with only bitsandbytes, otherwise we need to have two seperate rows
# for LightninAI which can be a bit in-consistent. 

class LlamaPyTorchLightningBenchmark:
    def __init__(self, model_path: str, precision: str, device: str) -> None:
        assert precision in ['fp16', 'fp32', 'int8', 'int4'], "Supported precision: 'fp16', 'fp32', 'int8', 'int4'"
        assert device in ['cuda', 'cpu'], f"Device {device} is not supported. Supported devices are: 'cuda' or 'cpu'"
        
        self.model_path, self.precision = model_path, precision
        self.device = 0 if device  == 'cpu' else 1 
        dtype = {
            'fp16': torch.float16,
            'fp32': torch.float32,
        }
        self.plugins = None 
        
        if self.precision in ['int8', 'int4']:
            self.weight_dtype = dtype["fp16"] # using default fp16 since fp32 not supported.
            self.quant_precision = self.precision
            self.plugins = BitsandbytesPrecision(self.quant_precision, self.weight_dtype)
            self.precision = None 
        
        else:
            self.precision = '16-true' if precision == 'fp16' else '32-true'
            assert self.device == 'cpu' and self.precision == '32-true', 'When precision is set to 32, then only CPU is supported'
        
        self.fabric = L.Fabric(devices=self.device, precision=self.precision, plugins=self.plugins)
        self.checkpoint_dir = os.path.dirname(self.model_path)
        check_valid_checkpoint_dir(self.checkpoint_dir)
        
        self.config = Config.from_json()
        
    def load_model(self):
        self.tokenizer = Tokenizer(self.checkpoint_dir)
        with self.fabric.init_module(empty_init=True):
            self.model = GPT(self.config)
        
        with self.fabric.init_tensor():
            self.model.set_kv_cache(batch_size=1)
        
        self.model.eval()
        load_checkpoint(self.fabric, self.model, self.model_path)
        return self 
    
    @torch.inference_mode()
    def run_model(self, prompt: str, max_tokens: int) -> float:
        encoded = self.tokenizer.encode(prompt, device=self.fabric.device)
        prompt_length = encoded.size(0)
        
        start_time = time.perf_counter()
        generated = generate(self.model, encoded, max_tokens)
        delta = time.perf_counter() - start_time
        
        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        tokens_generated = generated.size(0) - prompt_length
        return len(tokens_generated) / delta
        
    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)