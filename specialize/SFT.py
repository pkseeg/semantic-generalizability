from specialize.base_model import BaseModel
from tqdm.auto import trange
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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
        print(type(a))
        print(len(a))
        def preprocess_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding=True, max_length=512
            )
        print(f"Tokenizing")
        tokenized_ds = a.map(preprocess_function, batched=True, batch_size = 4, writer_batch_size=8)

        print(f"Renaming labels")
        tokenized_ds = tokenized_ds.map(lambda x: {"labels": x["label"]})

        print(f"Setting up LoRA")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type="SEQ_CLS"
        )

        print(f"Prep for kbit")
        self.model = prepare_model_for_kbit_training(self.model)

        print(f"PEFT")
        self.model = get_peft_model(self.model, lora_config)

        print(f"Setting training args")
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=5e-5,
            per_device_train_batch_size=16, # start with 16 I guess
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy=None,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        print(f"Data collator")
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        print(f"trainer")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        print(f"training")
        trainer.train()

        # Save the fine-tuned model
        #trainer.save_model("./fine_tuned_model")
        #print("Fine-tuning completed and model saved.")
    
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



            
        