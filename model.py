import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

def read_olmo():
    olmo_tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
    olmo_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct").to(device)
    return olmo_model, olmo_tokenizer

def read_qwen3b():
    qwen3b_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    qwen3b_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B").to(device)
    return qwen3b_model, qwen3b_tokenizer