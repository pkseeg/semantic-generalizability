import torch

# Function to embed texts in the dataset using a model
def embed(ds, model, tokenizer):
    # Tokenize the text column of the dataset
    def process_example(example):
        text = example["text"]
        inputs = tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        # Move inputs to the same device as the model
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling on GPU
        embedding = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        return {"embedding": embedding}
    
    # Apply the embedding logic to the dataset
    return ds.map(process_example, batched=True, batch_size=4) # FIXME batch size?

# Function to compute mean pooling from the model's hidden states
def mean_pooling(hidden_states, attention_mask):
    # Multiply hidden states by the attention mask to zero out padding
    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    # Compute the sum of hidden states across the sequence
    sum_hidden_states = masked_hidden_states.sum(dim=1)
    # Compute the sum of attention mask values across the sequence
    sum_attention_mask = attention_mask.sum(dim=1).unsqueeze(-1)
    # Avoid division by zero
    sum_attention_mask = torch.clamp(sum_attention_mask, min=1e-9)
    # Compute the mean pooled embeddings
    mean_pooled = sum_hidden_states / sum_attention_mask
    return mean_pooled
