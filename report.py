# report.py
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load test dataset
# -----------------------------
dataset = load_dataset("imdb")
small_test = dataset["test"].shuffle(seed=42).select(range(500))  # For speed

# -----------------------------
# 2. Load fine-tuned model pipeline
# -----------------------------
classifier = pipeline(
    "sentiment-analysis",
    model="/Users/leowaskin/Documents/Projects/LLM/results/checkpoint-500",       # path to your fine-tuned model
    tokenizer="/Users/leowaskin/Documents/Projects/LLM/results/checkpoint-500",   # path to tokenizer
    device="mps:0",                         # use MPS on Mac GPU
    truncation=True,                         # truncate long sequences
    max_length=512                            # max tokens for DistilBERT
)

# -----------------------------
# 3. Run predictions
# -----------------------------
preds = []
labels = []

for example in small_test:
    result = classifier(example["text"])[0]
    predicted_label = result["label"]
    
    # Convert model output to 0/1
    pred_id = 1 if predicted_label in ["LABEL_1", "POSITIVE"] else 0
    preds.append(pred_id)
    labels.append(example["label"])

# -----------------------------
# 4. Print classification report
# -----------------------------
print(classification_report(labels, preds, target_names=["NEGATIVE", "POSITIVE"]))
