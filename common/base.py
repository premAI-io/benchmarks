import json
import os
from abc import ABC, abstractmethod

import torch
from tqdm.auto import tqdm

from common.memory_tracker import MemoryTracker
from common.utils import get_logger


class BaseBenchmarkClass(ABC):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        """Benchmark base class. This class can be extended to other classes so that we can benchmark newer
            engines with minimal lines of code

        Args:
            model_path (str): The path of the model
            model_name (str): The name of the model (supported options: 'mistral', 'llama')
            benchmark_name (str): The name of engine to benchmark on
            precision (str): The precision in which the model is loaded for benchmarking
                (supported options: 'float32', 'float16', 'int8' and 'int4')
            device (str): The device in which benchmarking is done (supported options: 'cuda', 'cpu' and 'metal')
        """

        assert model_name in ["llama", "mistral"], ValueError(
            "Model other than 'llama' or 'mistral' is not supported"
        )

        assert precision in ["float32", "float16", "int8", "int4"], ValueError(
            "Precision other than 'float32', 'float16', 'int8' and 'int4' are not supported"
        )

        assert device in ["cuda", "cpu", "metal", "cuda:0"], ValueError(
            "Device other than 'cuda'/'cuda:0', 'cpu' and 'metal' are not supported"
        )

        self.model_name = model_name
        self.model_path = model_path
        self.precision = precision
        self.benchmark_name = benchmark_name
        self.device = device
        self.experiment_name = experiment_name

        # Make an experiment folder for each of the benchmark
        self.log_folder = os.path.join(os.getcwd(), "logs", model_name, experiment_name)
        self._log_file_path = os.path.join(self.log_folder, "performance.log")
        os.makedirs(self.log_folder)

        self.logger = get_logger(
            benchmark_name=benchmark_name, log_file_path=self._log_file_path
        )

        # Fetch the questions for quality checks
        self._questions_json_path = os.path.join(os.getcwd(), "questions.json")
        self.answers_json_path = os.path.join(self.log_folder, "quality_check.json")
        self.questions = json.load(open(self._questions_json_path, "r"))

        # different things to track
        self.tps_results, self.memory_usage_results, self.answers = [], [], []
        self.memory_tracker = MemoryTracker()

    @abstractmethod
    def load_model_and_tokenizer(self):
        """Loads the model and tokenizer for the engine"""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, prompt: str, chat_mode: bool = False):
        """
        Should return a dict
            - prompt: (str) The prompt
            - input_tokens: (List[int]) the list of tokens
            - tensor: the input tensor (Optional, default = None)
            - num_tokens: the number of input tokens
        """
        raise NotImplementedError

    @abstractmethod
    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        """Method to run the model

        Args:
            inputs: (dict) it should contain three keys:
                - prompt: (str) The prompt
                - input_tokens: (List[int]) the list of tokens
                - tensor: the input tensor (Optional, default = None)
                - num_input_tokens: the number of input tokens

            max_tokens: (int) The max number of output tokens
            temperature: (float) Sampling parameter, lesser value (example: 0.1) makes the output more determnistic

        Returns:
            dict: This should contain the following keys
                - 'output_tokens': A list of output tokens
                - 'num_output_tokens': This should contain the number of output tokens (without the input)
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, output: dict) -> str:
        """There are different ways of decoding the output
        This method expects to return a string
        """
        raise NotImplementedError

    def on_exit(self):
        pass

    def _benchmark_cuda(self, prompt: str, max_tokens: int, temperature: float):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        inputs = self.preprocess(prompt=prompt)

        with self.memory_tracker.track():
            torch.cuda.synchronize()

            start_event.record()
            output_dict = self.run_model(inputs, max_tokens, temperature)
            end_event.record()

            torch.cuda.synchronize()

        num_output_tokens = output_dict["num_output_tokens"]

        latency_sec = start_event.elapsed_time(end_event) / 1000
        peak_nvml_mb = self.memory_tracker.peak_memory

        token_per_sec = num_output_tokens / latency_sec
        gpu_mem_consumed = round(peak_nvml_mb, 2)

        return (token_per_sec, gpu_mem_consumed)

    def benchmark(
        self, prompt: str, max_tokens: int, repetitions: int, temperature: float
    ) -> None:
        for i in range(repetitions):
            self.logger.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )

            if self.device == "cuda":
                tok_per_sec, gpu_memory_consumed = self._benchmark_cuda(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature
                )
                self.tps_results.append(tok_per_sec)
                self.memory_usage_results.append(gpu_memory_consumed)
            else:
                raise NotImplementedError(
                    "For other device base benchmark is not implemented"
                )
        self.on_exit()

    def get_answers(self):
        self.logger.info("=> Running quality checks for LLM")

        for question in tqdm(self.questions, total=len(self.questions)):
            prompt = question["prompt"]
            max_tokens = question["max_tokens"]
            temperature = question["temperature"]
            expected = question["expected"]

            inputs = self.preprocess(prompt=prompt)
            output_dict = self.run_model(
                inputs=inputs, max_tokens=max_tokens, temperature=temperature
            )
            output = self.postprocess(output_dict)

            self.answers.append(
                {
                    "question": question,
                    "max_token": max_tokens,
                    "temperature": temperature,
                    "actual": output,
                    "expected": expected,
                }
            )
