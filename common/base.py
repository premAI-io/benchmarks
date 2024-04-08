from abc import ABC, abstractmethod

import torch

from common.memory_tracker import MemoryTracker


class BaseBenchmarkClass(ABC):
    def __init__(self, model_path: str, precision: int, device: str) -> None:
        self.model_path = model_path
        self.precision = precision
        self.device = device

        # tps = token/sec
        self.tps_results = []
        self.memory_usage_results = []
        self.memory_tracker = MemoryTracker()

    def preprocess_cuda(self, prompt: str):
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        return {"input_tokens": input_tokens, "num_tokens": len(input_tokens[0])}

    def bench_tracker_cuda(
        self, func, input_token_or_prompt, num_input_tokens, max_tokens
    ):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with self.memory_tracker.track():
            torch.cuda.synchronize()

            start_event.record()
            output = func(input_token_or_prompt, max_tokens)
            end_event.record()

            torch.cuda.synchronize()

        if len(output) > max_tokens:
            output = output[num_input_tokens:]

        latency_sec = start_event.elapsed_time(end_event) / 1000
        peak_nvml_mb = self.memory_tracker.peak_memory

        token_per_sec = len(output) / latency_sec
        gpu_mem_consumed = round(peak_nvml_mb, 2)
        return (token_per_sec, gpu_mem_consumed)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def run_model(self, input_token_or_prompt: str, max_tokens: int):
        pass

    def benchmark_cuda(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            self.logger.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )

            inputs = self.preprocess_cuda(prompt=prompt)
            tok_per_sec, gpu_memory_consumed = self.bench_tracker_cuda(
                self.run_model,
                input_token_or_prompt=inputs["input_tokens"],
                num_input_tokens=inputs["num_tokens"],
                max_tokens=max_tokens,
            )
            self.tps_results.append(tok_per_sec)
            self.memory_usage_results.append(gpu_memory_consumed)

        self.on_exit()

    def on_exit(self):
        pass
