from transformers import AutoTokenizer, AutoModelForCausalLM

olmo_tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
olmo_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")