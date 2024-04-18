import os
import sys

from llama_cpp import Llama
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class LlamaCPPBenchmark(BaseBenchmarkClass):
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
            "Precision should set either 'int8' or 'int4'"
        )
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            benchmark_name=benchmark_name,
            experiment_name=experiment_name,
            precision=precision,
            device=device,
        )

        if model_name == "llama":
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "llama-2-7b-chat-hf"
            )
        else:
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "mistral-7b-v0.1-instruct-hf"
            )

    def load_model_and_tokenizer(self):
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=0 if self.device == "cpu" else -1,
            verbose=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)
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

    def run_model(
        self, inputs: dict, max_tokens: int, temperature: float = 0.1
    ) -> dict:
        prompt = inputs["prompt"]
        output = self.model.create_completion(
            prompt, max_tokens=max_tokens, temperature=temperature
        )

        output_prompt = output["choices"][0]["text"]
        num_tokens = output["usage"]["completion_tokens"]
        return {"output_prompt": output_prompt, "num_output_tokens": num_tokens}

    def postprocess(self, output: dict) -> str:
        return output["output_prompt"]


if __name__ == "__main__":
    parser = launch_cli(description="LlamaCPP Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-gguf/llama-2-7b-chat."
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-gguf/mistral-7b-instruct-v0.1."
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "int4",
                "model_path": os.path.join(model_folder, model_name + "Q4_K_M.gguf"),
            },
            {
                "precision": "int8",
                "model_path": os.path.join(model_folder, model_name + "Q8_0.gguf"),
            },
        ]
    }

    make_report(
        args=args,
        benchmark_class=LlamaCPPBenchmark,
        runner_dict=runner_dict,
        benchmark_name="LlamaCPP",
        is_bench_pytorch=False,
    )
