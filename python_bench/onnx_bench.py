import time

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

from python_bench.benchmark import Benchmark


class ONNXBenchmark(Benchmark):
    def __init__(self, model_path, device="CPU"):
        super().__init__(model_path)
        self.device = device
        self.provider = (
            "CUDAExecutionProvider" if device == "GPU" else "CPUExecutionProvider"
        )

    def load_model(self) -> Benchmark:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = ORTModelForCausalLM.from_pretrained(
            self.model_path,
            use_cache=False,
            use_io_binding=False,
            provider=self.provider,
        )
        return self

    def run_model(self, prompt, max_tokens) -> float:
        device_str = "cuda" if self.device == "GPU" else "cpu"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device_str)
        start = time.time()
        gen_tokens = self.model.generate(**inputs, max_length=max_tokens)
        tokens_per_second = (gen_tokens.shape[1] - inputs["input_ids"].shape[1]) / (
            time.time() - start
        )
        return tokens_per_second
