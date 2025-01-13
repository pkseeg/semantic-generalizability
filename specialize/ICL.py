from specialize.base_model import BaseModel


class ICLModel(BaseModel):
    def __init__(self, model):
        super(ICLModel, self).__init__(model)
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

        Review: {text}
        Rating: 
        '''
        return prompt

    def format_prompt(self, y):
        example_0 = self.examples["text"][0]
        label_0 = self.examples["label"][0]

        example_1 = self.examples["text"][1]
        label_1 = self.examples["label"][1]

        example_2 = self.examples["text"][2]
        label_2 = self.examples["label"][2]

        example_3 = self.examples["text"][3]
        label_3 = self.examples["label"][3]

        example_4 = self.examples["text"][4]
        label_4 = self.examples["label"][4]

        text = y

        return self.prompt.format(example_0, label_0, example_1, label_1, example_2, label_2, example_3, label_3, example_4, label_4, text)

    
    def select_random(self, a, k = 5):
        examples = a.select(range(k))
        return examples
    
    def specialize(self, a):
        self.examples = self.select_random(a)
    
    def predict_classification(self, b):
        for sample in b:
            prompt = self.format_prompt(sample)
            print(sample)
            print(prompt)
            assert False
        