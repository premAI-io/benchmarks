import os

os.environ["CT2_VERBOSE"] = "2"

import time

import ctranslate2
import sentencepiece as spm

from python_bench.benchmark import Benchmark

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class CTranslateBenchmark(Benchmark):
    def __init__(self, model_path, gpu, compute_type):
        super().__init__(model_path)
        self.gpu = gpu
        self.compute_type=compute_type

    def load_model(self) -> Benchmark:
        self.generator = ctranslate2.Generator(
            self.model_path, device="cuda" if self.gpu else "cpu", compute_type=self.compute_type
        )
        self.sp = spm.SentencePieceProcessor(
            os.path.join(self.model_path, "tokenizer.model")
        )
        return self

    def run_model(self, prompt, max_tokens):
        prompt_tokens = ["<s>"] + self.sp.encode_as_pieces(
            f"{B_INST} {prompt.strip()} {E_INST}"
        )
        start = time.time()
        step_results = self.generator.generate_tokens(
            prompt_tokens,
            max_length=max_tokens,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
        )
        count = 0
        for _ in self.generate_words(step_results):
            count += 1
        return count / (time.time() - start)

    def generate_words(self, step_results):
        tokens_buffer = []

        for step_result in step_results:
            is_new_word = step_result.token.startswith("▁")

            if is_new_word and tokens_buffer:
                word = self.sp.decode(tokens_buffer)
                if word:
                    yield word
                tokens_buffer = []

            tokens_buffer.append(step_result.token_id)

        if tokens_buffer:
            word = self.sp.decode(tokens_buffer)
            if word:
                yield word
