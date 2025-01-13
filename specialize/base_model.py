class BaseModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def specialize(self, a):
        pass

    def predict_classification(self, b):
        pass

    def model_out(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            inputs["input_ids"], 
            max_new_tokens=512,
            num_return_sequences=1, 
            do_sample=True,
            temperature=1.0
        )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_output