import os
import sys

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache
from exllamav2.config import ExLlamaV2Config
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from exllamav2.tokenizer.tokenizer import ExLlamaV2Tokenizer
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class ExLlamaV2Benchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        assert precision in ["int8", "int4"], ValueError(
            "Available precision: 'int8', 'int4'"
        )
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            benchmark_name=benchmark_name,
            experiment_name=experiment_name,
            precision=precision,
            device=device,
        )

    def load_model_and_tokenizer(self):
        # set up model config
        self.config = ExLlamaV2Config()
        self.config.model_dir = self.model_path
        self.config.prepare()

        # set up model and cache
        self._model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self._model, lazy=True)
        self._model.load_autosplit(self.cache)
        self.tokenizer_exllama = ExLlamaV2Tokenizer(self.config)
        self.model = ExLlamaV2BaseGenerator(
            self._model, self.cache, self.tokenizer_exllama
        )
        self.model.warmup()

        # set up the huggingface tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # set up exllamav2 settings
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.disallow_tokens(
            self.tokenizer_exllama, [self.tokenizer_exllama.eos_token_id]
        )
        return self

    def preprocess(
        self, prompt: str, chat_mode: bool = True, for_benchmarks: bool = True
    ):
        if chat_mode:
            template = self.get_chat_template_with_instruction(
                prompt=prompt, for_benchmarks=for_benchmarks
            )
            prompt = self.tokenizer.apply_chat_template(template, tokenize=False)
        tokenized_input = self.tokenizer.encode(text=prompt)
        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": None,
            "num_input_tokens": len(tokenized_input),
        }

    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        # first set up the settings
        self.settings.token_repetition_penalty = 1.01
        self.settings.temperature = temperature
        self.settings.top_k = 50
        self.settings.top_p = 0.1

        # now run the model
        prompt = inputs["prompt"]
        output_text = self.model.generate_simple(
            prompt,
            self.settings,
            max_tokens,
            seed=1234,
            completion_only=True,
            decode_special_tokens=True,
        )

        tokenized_output = self.tokenizer.encode(output_text)
        return {
            "output_text": output_text,
            "output_tokens": tokenized_output,
            "num_output_tokens": len(tokenized_output),
        }

    def postprocess(self, output: dict) -> str:
        return output["output_text"]

    def on_exit(self):
        if self.device == "cuda":
            del self.model
            torch.cuda.synchronize()
        else:
            del self.model


if __name__ == "__main__":
    parser = launch_cli(description="ExLlamaV2 Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-exllamav2-"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-exllamav2-"
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "int4",
                "model_path": os.path.join(model_folder, model_name + "4.0-bit"),
            },
            {
                "precision": "int8",
                "model_path": os.path.join(model_folder, model_name + "8.0-bit"),
            },
        ]
    }

    make_report(
        args=args,
        benchmark_class=ExLlamaV2Benchmark,
        runner_dict=runner_dict,
        benchmark_name="ExLlamaV2",
        is_bench_pytorch=False,
    )
