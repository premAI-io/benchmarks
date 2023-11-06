import time
from benchmarking.benchmark import Benchmark
from llama_cpp import Llama


class LlamaCPPBenchmark(Benchmark):
    def __init__(self, model_path):
        super().__init__(model_path)

    def load_model(self) -> Benchmark:
        self.model = Llama(model_path=self.model_path, verbose=False)
        return self

    def run_model(self, prompt, max_tokens):
        start = time.time()
        output = self.model.create_completion(prompt, max_tokens=max_tokens)
        return len(output) / (time.time() - start)
