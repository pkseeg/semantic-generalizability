import torch
from tqdm.auto import tqdm

def embed(ds, model, tokenizer):
    embeddings = []
    for example in tqdm(ds):
        print("Embedding a single example")
        text = example["text"]
        print(f"Tokenizing: {text}")
        inputs = tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        print(f"Put inputs on GPU")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        print(f"Getting model outputs")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        print(f"Accessing hidden state")
        last_hidden_state = outputs.hidden_states[-1]  # Last layer's hidden states
        print(f"Mean pooling")
        embedding = mean_pooling(last_hidden_state, inputs["attention_mask"])
        print(f"Shape of a single embedding: {embedding.shape}")
        embedding = embedding.cpu()
        embeddings.append(embedding)
    return {"embedding": embeddings}

    # def process_example(example):
    #     print("Embedding a single example")
    #     assert "text" in example
    #     text = example["text"]
    #     print(f"Tokenizing: {text}")
    #     inputs = tokenizer(
    #         text, 
    #         padding=True, 
    #         truncation=True, 
    #         return_tensors="pt"
    #     )
    #     print(f"Put inputs on GPU")
    #     inputs = {key: value.to(model.device) for key, value in inputs.items()}
    #     print(f"Getting model outputs")
    #     with torch.no_grad():
    #         outputs = model(**inputs, output_hidden_states=True)
    #     print(f"Accessing hidden state")
    #     last_hidden_state = outputs.hidden_states[-1]  # Last layer's hidden states
    #     print(f"Mean pooling")
    #     embedding = mean_pooling(last_hidden_state, inputs["attention_mask"])
    #     print(f"Shape of a single embedding: {embedding.shape}")
    #     #assert False
    #     embedding = embedding.cpu()
    #     # assert False
    #     return {"embedding": embedding}
    
    #print("Embedding dataset.....")
    #return ds.map(process_example, batched=False)

def mean_pooling(hidden_states, attention_mask):
    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    sum_hidden_states = masked_hidden_states.sum(dim=1)
    sum_attention_mask = attention_mask.sum(dim=1).unsqueeze(-1)
    sum_attention_mask = torch.clamp(sum_attention_mask, min=1e-9)
    mean_pooled = sum_hidden_states / sum_attention_mask
    return mean_pooled
