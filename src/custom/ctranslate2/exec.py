import os
import sys
import time

import ctranslate2
import sentencepiece as spm

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# The code below is adapted from
# https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L225-L268


def main():
    if len(sys.argv) > 2:
        print(r"""Insufficient argument count:
    <arg1> : MODEL_DIR
    <arg2> : PROMPT
        """)
        sys.exit(-1)
    model_dir = sys.argv[1]
    prompt = sys.argv[2]

    print("Loading the model...")
    generator = ctranslate2.Generator(model_dir)
    # generator.device
    sp = spm.SentencePieceProcessor(os.path.join(model_dir, "tokenizer.model"))

    # defaults to 100 consider providing customizability
    max_generation_length = 100

    # prompt tokenization
    prompt_tokens = ["<s>"] + sp.encode_as_pieces(
        f"{B_INST} {prompt.strip()} {E_INST}"
    )
    time_start = time.perf_counter_ns()
    step_results = generator.generate_tokens(
        prompt_tokens,
        max_length=max_generation_length,
        sampling_temperature=0.6,
        sampling_topk=20,
        sampling_topp=1,
    )
    end_time = time.perf_counter_ns() - time_start

    print("")
    print("Llama2: ", end="", flush=True)

    text_output = ""
    count = 0
    for word in generate_words(sp, step_results):
        count += 1
        if text_output:
            word = " " + word
        print(word, end="", flush=True)
        text_output += word

    print("")

    # report perf
    print("===================")
    total_time = 1e9/end_time
    print("total time:", str(total_time))
    print("tokens generated: ", str(count))
    print("avg tokens/sec: ", str(total_time / count))


def generate_words(sp, step_results):
    tokens_buffer = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and tokens_buffer:
            word = sp.decode(tokens_buffer)
            if word:
                yield word
            tokens_buffer = []

        tokens_buffer.append(step_result.token_id)

    if tokens_buffer:
        word = sp.decode(tokens_buffer)
        if word:
            yield word


if __name__ == "__main__":
    main()
