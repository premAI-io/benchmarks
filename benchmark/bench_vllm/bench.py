import sys 
from vllm import LLM
from vllm.model_executor.parallel_utils import parallel_state

import logging
from benchmark.base import LlamaBenchmarkBase, benchmark_arg_parser

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class LlamavLLMBenchmark(LlamaBenchmarkBase):
    def __init__(self, model_dir_path: str, device: str, precision: str) -> None:
        assert device == "cuda", ValueError("Supported device is cuda only.")
        assert precision in ["fp16", "fp32", "int4"], ValueError(
            "supported precision are: fp16, fp32 and int4"
        )

        self.precision = precision
        self.precision_map = {"fp16": "float16", "fp32": "float32"}
        super().__init__(model_dir_path=model_dir_path, device=device)
    
    def load_model(self):
        if self.precision != "int4":
            self.model = LLM(model=self.model_path)
            self.model.dtype = self.precision_map[self.precision]
        else:
            self.model = LLM(model=self.model_path, quantization="AWQ")
        return self

    def run_model(self, prompt: str, max_tokens: int) -> float:
        self.model.max_num_seqs = max_tokens
        output = self.model.generate(prompts=[prompt])
        return output
    
    def benchmark(self, prompt: str, max_tokens: int, repetitions: int, *args, **kwargs) -> None:
        super().benchmark(prompt, max_tokens, repetitions, *args, **kwargs)
        
        if self.device == "cuda":
            parallel_state.destroy_model_parallel()
    
    
benchmark_arg_parser(name="vLLM", benchmark_class=LlamavLLMBenchmark)