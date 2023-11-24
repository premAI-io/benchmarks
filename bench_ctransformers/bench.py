import argparse
import logging 
import sys
import time 
from typing import Optional
from collections import defaultdict
import numpy as np
from ctransformers import AutoModelForCausalLM

logging.getLogger('ctransformers').setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class CTransformersBenchmark:
    def __init__(self, model_path: str, device: Optional[str]='cpu', model_type: Optional[str]=None) -> None:
        self.model_path, self.device = model_path, device
        self.model_map = {
            'gpt2' : {
                'devices': ['cpu'],
                'type': 'gpt2' 
            },
            'gptj': {
                'type': 'gptj',
                'devices': ['cpu'],
            },
            'gpt4allj': {
                'devices': ['cpu'],
                'type': 'gptj'
            },
            'gpt-neo': {
                'devices': ['cpu'],
                'type': 'gpt_neox'
            },
            'falcon':{
                'devices': ['cpu', 'cuda'],
                'type': 'falcon'
            },
            'llama': {
                'devices': ['cpu', 'cuda', 'metal'],
                'type': 'llama'
            }, 
            'mpt': {
                'devices': ['cpu', 'cuda'],
                'type': 'mpt'
            },
            'starcoder': {
                'devices': ['cpu'],
                'type': 'gpt_bigcode'
            }, 
            'dolly': {
                'devices': ['cpu'],
                'type': 'dolly-v2'
            },
            'replit': {
                'devices': ['cpu'],
                'type': 'replit'
            }
        }
        self.results = [] 
        # check if the model path falls under this 
        _model_name = model_path.split('/')[-1].lower()
        matched_key_from_map = [key for key in self.model_map if key in _model_name]
        if not matched_key_from_map and model_type is None:
            raise ValueError(
                f"The model: {_model_name} does not fall under the following model categories: {list(self.model_map.keys())}"
                f"If you think, that your model path falls under any of the model architecture, then place the value inside model_type argument"
            )
        
        self.model_type = matched_key_from_map[0] if model_type is None else model_type
        
        # check if the selected model supports that device else choose default device (i.e. first value of the list)
        self.device = device if device is not None and device in self.model_map[self.model_type]['devices'] else self.model_map[self.model_type]['devices'][0]
        
    def load_model(self):
        # FIXME: Not sure how to get num layers for each model to know how many to fit into VRAM.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            model_type=self.model_type,
            gpu_layers=50 if self.device == 'cuda' else 0
        )
        return self
    
    def run_model(self, prompt: str, max_tokens: int) -> float:
        start = time.time()
        output = self.model(prompt, max_new_tokens=max_tokens)
        tokens = len(self.model.tokenize(output))
        return tokens / (time.time() - start)

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )   
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)
    
   
   
path = "/home/anindya/Downloads/replit-openorca.ggmlv1.q4_0.bin"
ben = CTransformersBenchmark(
    model_path=path, device='cpu'
).load_model()

ben.benchmark(prompt="hello", max_tokens=3, repetitions=2)
print(ben.results) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTransformers Benchmark.")
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt for the model.",
    )
    parser.add_argument("--max_tokens", type=int, help="The maximum number of tokens.")
    parser.add_argument(
        "--repetitions",
        type=int,
        help="The number of repetitions for the benchmark.",
    )
    parser.add_argument(
        "--device",
        help="Device to use for the benchmark.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to the log file for writing logs (in append mode).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        help="Path to the models directory.",
    )
    args = parser.parse_args()
    logging.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} "
        + f"repetitions={args.repetitions} device={args.device}"
    )
    report = defaultdict(lambda: defaultdict(float))