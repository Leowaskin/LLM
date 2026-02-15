from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch

#Loading the dataset
dataset = load_dataset("json", data_files="custom_data/resume_faq.jsonl")["train"]
num_labels = len(set(dataset['label']))

#Loadding The Model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

#Tokenization 
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding=False,
    )

dataset = dataset.map(tokenize)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Training arguments
training_args = TrainingArguments(
    output_dir='./faq_model',
    eval_strategy='no',
    save_strategy='epoch',
    learning_rate = 2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=5,
    report_to="none",
    save_total_limit=1)

#Train(make a trainer and fit it to the training arguments)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

trainer.train()


#saving the traned model and tokenizer
trainer.model.save_pretrained("./faq_model")
tokenizer.save_pretrained("./faq_model")


print("Model saved to ./faq_model")


