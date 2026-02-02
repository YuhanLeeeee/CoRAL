import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# add special token -- <FP_TOKEN>

model = AutoModelForCausalLM.from_pretrained("./Qwen3-8B")
tokenizer_Qwen = AutoTokenizer.from_pretrained("./Qwen3-8B")

new_tokens = ["<FP_TOKEN>", '.', '>', '>>', ')', '(']
num_added = tokenizer_Qwen.add_tokens(new_tokens)
print(len(tokenizer_Qwen))

model.resize_token_embeddings(len(tokenizer_Qwen))
print(tokenizer_Qwen.encode('<FP_TOKEN>'))

save_directory = "./Qwen3-8B-FP"
model = model.to(dtype=torch.bfloat16)
model.save_pretrained(save_directory)
tokenizer_Qwen.save_pretrained(save_directory)