import gc
import os
import sys
import time

import torch
from onnxruntime import InferenceSession
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoConfig, AutoTokenizer

sys.path.append("/mnt")
sys.path.append("/mnt/benchmarks/")

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class ONNXOptimumBenchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        assert precision in ["float32", "float16"], ValueError(
            "Supported precision: 'float32' and 'float16'"
        )
        assert device in ["cuda"], ValueError(
            "Current implement is only supported for device = 'cuda'"
        )
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            benchmark_name=benchmark_name,
            experiment_name=experiment_name,
            precision=precision,
            device=device,
            root_folder="/mnt/benchmarks",
        )

        if model_name == "llama":
            self.tokenizer_folder = os.path.join(
                self.root_folder, "models", "llama-2-7b-chat-hf"
            )
        else:
            self.tokenizer_folder = os.path.join(
                self.root_folder, "models", "mistral-7b-v0.1-instruct-hf"
            )

    def load_model_and_tokenizer(self):
        start_time = time.perf_counter()
        onnx_path = os.path.join(self.model_path, "model.onnx")
        config = AutoConfig.from_pretrained(self.model_path)

        # load the session and the model
        self.session = InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        self.model = ORTModelForCausalLM(
            self.session, config, use_cache=False, use_io_binding=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)
        delta = time.perf_counter() - start_time
        self.logger.info(f"Model Loading time took: {delta:.2f} seconds")
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
        tensor = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": tensor,
            "num_input_tokens": len(tokenized_input),
        }

    @torch.inference_mode(mode=True)
    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        tensor = inputs["tensor"]
        num_input_tokens = inputs["num_input_tokens"]

        generated = self.model.generate(
            **tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_tokens = generated[0].detach().tolist()[num_input_tokens:]
        return {"output_tokens": output_tokens, "num_output_tokens": len(output_tokens)}

    def postprocess(self, output: dict) -> str:
        output_tokens = output["output_tokens"]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return output_text

    def on_exit(self):
        if self.device in ["cuda", "cuda:0"]:
            del self.model
            del self.session
            torch.cuda.synchronize()
            gc.collect()
        else:
            del self.model
            del self.session


if __name__ == "__main__":
    parser = launch_cli(description="ONNX HF-Optimum Benchmark.")
    args = parser.parse_args()

    model_folder = "/mnt/benchmarks/models"
    model_name = (
        f"{args.model_name}-2-7b-chat-onnx"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-onnx"
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "float32",
                "model_path": os.path.join(model_folder, model_name + "-float32"),
            },
            {
                "precision": "float16",
                "model_path": os.path.join(model_folder, model_name + "-float16"),
            },
        ]
    }

    make_report(
        args=args,
        benchmark_class=ONNXOptimumBenchmark,
        runner_dict=runner_dict,
        benchmark_name="ONNX-HF-Optimum",
        is_bench_pytorch=False,
    )
