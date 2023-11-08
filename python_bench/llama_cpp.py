import time
import logging
from python_bench.benchmark import Benchmark
from llama_cpp import Llama

logging.getLogger("llama_cpp").setLevel(logging.ERROR)


class LlamaCPPBenchmark(Benchmark):
    def __init__(self, model_path, gpu):
        super().__init__(model_path)
        self.gpu = gpu

    def load_model(self) -> Benchmark:
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1 if self.gpu else 0,
            verbose=False,
        )
        return self

    def run_model(self, prompt, max_tokens):
        start = time.time()
        output = self.model.create_completion(prompt, max_tokens=max_tokens)
        tokens = output["usage"]["completion_tokens"]
        return tokens / (time.time() - start)
