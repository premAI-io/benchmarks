from time import perf_counter

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

import sys

model_name = "meta-llama/Llama-2-7b-hf"
prompt = sys.argv[1] if len(sys.argv) > 1 else "write a hello world in rust" 

# quantization_config_4bit = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="cuda:0",
                                             torch_dtype=torch.float16,
                                             # use_flash_attention_2=True
                                             # quantization_config=quantization_config_4bit
                                             )
model = model.to_bettertransformer()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

model.generate(**model_inputs, max_new_tokens=100)
print("model generate-->")
st = perf_counter()

# with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
output = model.generate(**model_inputs, max_new_tokens=1000)

print("tokens: ", output[0])
tokens = len(output[0])
print("tokens count: ", str(tokens))

print("==========output===========")
print(tokenizer.decode(output[0], skip_special_tokens=True))

print("==========perf===========")
tt = perf_counter() - st
print("Total time: ", str(tt))
print("Tokens/sec: ", str(tokens/tt))
print("==========perf===========")


