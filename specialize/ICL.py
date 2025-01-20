from specialize.base_model import BaseModel
from tqdm.auto import trange


class ICLModel(BaseModel):
    def __init__(self, model, tokenizer):
        super(ICLModel, self).__init__(model, tokenizer)
        self.prompt = self.get_prompt()
    
    def get_prompt(self):
        prompt = '''Give the star rating (1-5) of the following reviews.

        Review: {example_0}
        Rating: {label_0}

        Review: {example_1}
        Rating: {label_1}

        Review: {example_2}
        Rating: {label_2}

        Review: {example_3}
        Rating: {label_3}

        Review: {example_4}
        Rating: {label_4}

        Review: {text}'''
        return prompt

    def format_prompt(self, sample):
        example_0 = self.examples["text"][0]
        label_0 = str(int(self.examples["label"][0]))

        example_1 = self.examples["text"][1]
        label_1 = str(int(self.examples["label"][1]))

        example_2 = self.examples["text"][2]
        label_2 = str(int(self.examples["label"][2]))

        example_3 = self.examples["text"][3]
        label_3 = str(int(self.examples["label"][3]))

        example_4 = self.examples["text"][4]
        label_4 = str(int(self.examples["label"][4]))

        text = sample["text"]

        return self.prompt.format(example_0=example_0, label_0=label_0, example_1=example_1, label_1=label_1, example_2=example_2, label_2=label_2, example_3=example_3, label_3=label_3, example_4=example_4, label_4=label_4, text=text)

    
    def select_random(self, a, k = 5):
        examples = a.select(range(k))
        return examples
    
    def specialize(self, a):
        self.data_pool = a
    
    def format_out(self, output):
        if "1" in output:
            return 1
        elif "2" in output:
            return 2
        elif "3" in output:
            return 3
        elif "4" in output:
            return 4
        return 5
        
    
    def predict_classification(self, b, batch_size = 2): # batch size of 2 should work on the colab A100 for OlMO
        ytrues = []
        yhats = []
        for i in trange(0, len(b), batch_size):
            # every batch selects  a new set of examples
            self.examples = self.select_random(self.data_pool)
            samples = b.select(range(i, min(i + batch_size, len(b))))
            prompts = [self.format_prompt(sample) for sample in samples]
            decoded_outputs = self.model_out(prompts)
            yhat = [self.format_out(output[len(prompt):].strip()) for output, prompt in zip(decoded_outputs, prompts)]
            ytrues.extend([sample["label"] for sample in samples])
            yhats.extend(yhat)
        return ytrues, yhats



            
        