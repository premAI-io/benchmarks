import os
import sys

from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class CTransformersBenchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        super().__init__(
            model_path=model_path,
            model_name=model_name,
            benchmark_name=benchmark_name,
            precision=precision,
            device=device,
            experiment_name=experiment_name,
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)

        model_file_mapping = {
            "llama": {
                "int4": "llama-2-7b-chat.Q4_K_M.gguf",
                "int8": "llama-2-7b-chat.Q8_0.gguf",
            },
            "mistral": {
                "int4": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                "int8": "mistral-7b-instruct-v0.1.Q8_0.gguf",
            },
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            model_file=model_file_mapping[self.model_name][self.precision],
            model_type=self.model_name,
            gpu_layers=50 if self.device in ["cuda", "metal"] else 0,
            # context_length=1024 (This exceeds the memory without changing the quality)
        )
        return self

    def preprocess(self, prompt: str, chat_mode: bool = True):
        if chat_mode:
            prompt = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)

        tokenized_input = self.tokenizer.encode(text=prompt)
        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": None,
            "num_input_tokens": len(tokenized_input),
        }

    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        prompt = inputs["prompt"]
        output = self.model(
            prompt, stream=False, max_new_tokens=max_tokens, temperature=temperature
        )
        generated_tokens = self.tokenizer.encode(output)

        # Note: CTransformers produces tokens after the input tokens
        return {
            "output_prompt": output,
            "output_tokens": generated_tokens,
            "num_output_tokens": len(generated_tokens),
        }

    def postprocess(self, output: dict) -> str:
        output_tokens = output["output_tokens"]
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    parser = launch_cli(description="CTransformers Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-gguf"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-gguf"
    )

    runner_dict = {
        "cuda": [
            {"precision": "int4", "model_path": os.path.join(model_folder, model_name)},
            {"precision": "int8", "model_path": os.path.join(model_folder, model_name)},
        ]
    }

    make_report(
        args=args,
        benchmark_class=CTransformersBenchmark,
        runner_dict=runner_dict,
        benchmark_name="CTransformers",
    )
