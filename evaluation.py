from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

model_path = "./results/checkpoint-500"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_dataset("imdb")

# -----------------------
# 1. Tokenize the dataset
# -----------------------
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized_test = dataset["test"].map(preprocess, batched=True)

# use only 500 samples
small_test = tokenized_test.shuffle(seed=42).select(range(500))

# -----------------------
# 2. Run evaluation
# -----------------------
trainer = Trainer(model=model, tokenizer=tokenizer)
metrics = trainer.evaluate(eval_dataset=small_test)

print(metrics)
