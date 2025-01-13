import torch

def embed(ds, model, tokenizer):
    def process_example(example):
        text = example["text"]
        inputs = tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # Last layer's hidden states
        embedding = mean_pooling(last_hidden_state, inputs["attention_mask"])
        embedding = embedding.cpu()
        return {"embedding": embedding}
    
    return ds.map(process_example, batched=True, batch_size=4)

def mean_pooling(hidden_states, attention_mask):
    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    sum_hidden_states = masked_hidden_states.sum(dim=1)
    sum_attention_mask = attention_mask.sum(dim=1).unsqueeze(-1)
    sum_attention_mask = torch.clamp(sum_attention_mask, min=1e-9)
    mean_pooled = sum_hidden_states / sum_attention_mask
    return mean_pooled
