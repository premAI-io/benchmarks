from __future__ import annotations
from abc import ABC, abstractmethod


class Benchmark(ABC):
    """
    An abstract class for benchmarking different machine learning frameworks.

    This class provides a skeleton for benchmarking the performance of different
    machine learning frameworks. It includes methods for loading a model, running
    the model, and benchmarking the model's performance. The actual implementation
    of these methods is left to the subclasses.

    Attributes:
        model_path (str): The path to the model file.

    Methods:
        load_model(): An abstract method that loads the model.
        run_model(prompt, max_tokens): An abstract method that runs the model and estimate tokens/second.
        benchmark(prompt, max_tokens, repetitions=10): Runs the model several times
            and calculates the average tokens per second.
    """

    def __init__(self, model_path):
        self.model_path = model_path

    @abstractmethod
    def load_model(self) -> Benchmark:
        pass

    @abstractmethod
    def run_model(self, prompt, max_tokens) -> float:
        pass

    def benchmark(self, prompt, max_tokens, repetitions) -> float:
        results = []
        for _ in range(repetitions):
            tokens_per_second = self.run_model(prompt, max_tokens)
            results.append(tokens_per_second)
        return sum(results) / len(results)
