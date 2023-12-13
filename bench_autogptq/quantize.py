import argparse
import os

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer


class PrepareGPTForQuantization:
    @classmethod
    def convert(
        cls,
        hf_model_dir: str,
        quantized_dir: str,
        precision: int,
        conversion_parquet_file: str,
    ) -> None:
        assert precision in [
            4,
            8,
        ], "For benchmarks supported precision are 4 and 8 bits."
        if len(os.listdir(quantized_dir)):
            print("=> Already quantization exists, exiting ... ")
            return

        # Note: We are keeping group size to default i.e. 1024, so that we can see in standard lower VRAM requirement
        # However optimal VARM size is 128

        quantization_config = BaseQuantizeConfig(bits=precision)
        model = AutoGPTQForCausalLM.from_pretrained(hf_model_dir, quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)

        dataset = Dataset.from_parquet(conversion_parquet_file)
        assert "text" in list(dataset.features.keys()), ValueError(
            "The key: 'text' must be present for starting quantization"
        )

        examples = [tokenizer(example) for example in dataset["text"]][:1]
        print(f"=> Starting to quantize the model in {precision} bit precision ...")

        try:
            model.quantize(examples)
            model.save_pretrained(quantized_dir)
        except Exception as e:
            print(f"Unexpected error occuared: {e}")
        print("=> Conversion finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGPT conversion Script.")
    parser.add_argument("--hf_dir", type=str, help="HuggingFace Model dir.")

    parser.add_argument(
        "--q_dir",
        type=str,
        help="The quantized model dir",
    )

    parser.add_argument(
        "--precision", type=int, help="The precison of weights for conversion"
    )

    parser.add_argument(
        "--parquet",
        type=str,
        help="The parquet file that contains some set of text examples for doing quantization.",
    )

    args = parser.parse_args()
    PrepareGPTForQuantization.convert(
        hf_model_dir=args.hf_dir,
        quantized_dir=args.q_dir,
        precision=args.precision,
        conversion_parquet_file=args.parquet,
    )
