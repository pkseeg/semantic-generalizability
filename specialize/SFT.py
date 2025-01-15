from specialize.base_model import BaseModel
from tqdm.auto import trange


class SFTModel(BaseModel):
    def __init__(self, model, tokenizer):
        super(SFTModel, self).__init__(model, tokenizer)
        self.prompt = self.get_prompt()
    
    def get_prompt(self):
        prompt = '''Give the star rating (1-5) of the following reviews.

        Review: {text}
        Rating: 
        '''
        return prompt

    def format_prompt(self, sample):
        text = sample["text"]
        return self.prompt.format(text=text)

    
    def select_random(self, a, k = 5):
        examples = a.select(range(k))
        return examples
    
    def specialize(self, a):
        # a is a huggingface ds containing text and label
        # self.model and self.tokenizer are huggingface model and tokenizer
        # we want to fine-tune self.model using batched instances in the form of
        # ds["text"][i] -> str(s["label"][i])
        self.examples = self.select_random(a)
    
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
        
    
    def predict_classification(self, b, batch_size = 32):
        ytrues = []
        yhats = []
        for i in trange(0, len(b), batch_size):
            samples = b.select(range(i, min(i + batch_size, len(b))))
            prompts = [self.format_prompt(sample) for sample in samples]
            decoded_outputs = self.model_out(prompts)
            yhat = [self.format_out(output) for output in decoded_outputs]
            ytrues.extend([sample["label"] for sample in samples])
            yhats.extend(yhat)
        return ytrues, yhats



            
        