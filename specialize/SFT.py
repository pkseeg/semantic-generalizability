from specialize.base_model import BaseModel
from tqdm.auto import trange
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class SFTModel(BaseModel):
    def __init__(self, model, tokenizer, task = "classification"):
        super(SFTModel, self).__init__(model, tokenizer)
        if task == "classification":
            self.prompt = self.get_prompt_classification()
        elif task == "qa":
            self.prompt = self.get_prompt_qa()
        self.task = task

        # FIXME ADJUST CLASSIFICATION
    
    def get_prompt_classification(self):
        prompt = '''Give the star rating (1-5) of the following reviews.

        Review: {text}
        Rating:'''
        return prompt
    
    def get_prompt_qa(self):
        prompt = '''Answer the following question, given the context.

        Context: {context}
        Question: {question}
        Answer:'''
        return prompt

    def format_prompt(self, examples):
        if self.task == "classification":
            return [self.prompt.format(text=text) for text in examples["text"]]
        elif self.task == "qa":
            return [self.prompt.format(context=context, question=question) for context, question in zip(examples["context"], examples["question"])]
    

    def add_new_column(self, df, col_name, col_values):
        # Define a function to add the new column
        def create_column(updated_df):
            updated_df[col_name] = col_values  # Assign specific values
            return updated_df

        # Apply the function to each item in the dataset
        df = df.map(create_column)

        return df

    def specialize(self, a):

        # let's do the dataset first
        self.add_new_column(a, "output", [ans[0] for ans in a["answers"]])
        self.add_new_column(a, "input", [self.prompt.format(context=context, question=question) for context, question in zip(a["context"], a["question"])])
        #a.add_column("output", [ans[0] for ans in a["answers"]])
        #a.add_column("input", [self.prompt.format(context=context, question=question) for context, question in zip(a["context"], a["question"])])

        print(a)
        print(a[0])

        def preprocess_function(examples):
            inputs = examples["input"]
            targets = examples["output"]
            model_inputs = self.tokenizer(inputs, truncation=True, padding=True, max_length=512)
            labels = self.tokenizer(targets, truncation=True, padding=True, max_length=512)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        print(f"Tokenizing")
        tokenized_ds = a.map(preprocess_function, batched=True)

        #print(f"Renaming labels")
        #tokenized_ds = tokenized_ds.map(lambda x: {"labels": self.tokenizer(x["answers"][0], truncation=True, padding=True)})
        #tokenized_ds = tokenized_ds.map(lambda x: {"labels": x["answers"][0]})

        print(tokenized_ds[0])

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
            load_best_model_at_end=False,
            report_to="none"
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



            
        