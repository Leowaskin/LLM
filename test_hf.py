from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

#Load data
dataset = load_dataset("imdb")
train = dataset["train"].shuffle(seed=42).select(range(2000))
test = dataset["test"].shuffle(seed=42).select(range(2000))

#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#Tokenize Data
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

train = train.map(tokenize_fn, batched=True)
test = test.map(tokenize_fn, batched=True)

#Load Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

#Training Settings(Arguments)
training_args = TrainingArguments(
    output_dir = "results",
    num_train_epochs = 1,
    per_device_train_batch_size = 4,
    eval_strategy = "epoch",
    logging_steps = 10, 
    learning_rate = 2e-5)
    
#trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train,
    eval_dataset = test, 
    processing_class = tokenizer)

#Train
trainer.train()
