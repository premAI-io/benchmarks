# This file is adapted from the LLama project:
# https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py

# Original LLama code by Facebook AI Research
# Adapted by Gadersd

import logging

from sentencepiece import SentencePieceProcessor

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        logger.info(
            f"#words: {self.n_words} BOS ID: {self.bos_id} EOS ID: {self.eos_id} PAD ID: {self.pad_id}"
        )

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int]) -> str:
        return self.sp_model.decode(t)
