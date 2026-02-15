from transformers import pipeline
import torch
from answer_mapping import LABEL_TO_ANSWER

classifier = pipeline(
    "text-classification",
    model="./faq_model",
    tokenizer="./faq_model",
    device="mps:0" if torch.backends.mps.is_available() else -1
)

def ask_chatbot(question):
    results = classifier(question, truncation=True, top_k=2)
    
    best = results[0]
    second = results[1]
    
    #to handle LABEL_0/LABEL_1
    label_id = int(best['label'].split("_")[-1])

    if best['score'] - second['score'] < 0.05:
        return "I'm not confident enough to answer to that question!"
    
    return LABEL_TO_ANSWER.get(label_id, "I dont have an answer for that question")


#Questions
questions = [
    "Who is Barkath Baig Mirza?",
    "What is his GPA?",
    "What tools and frameworks has he used?",
    "Has he worked with machine learning?",
    "What is his career goal?",
    "How do I connect with Barkath online?"
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask_chatbot(q)}")