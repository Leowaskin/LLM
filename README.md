# Barkath Baig Mirza FAQ Bot

This project implements a fine-tuned DistilBERT model designed to answer frequently asked questions about Barkath Baig Mirza based on his resume and professional background. It classifies user queries into predefined categories (labels) and maps them to specific answers.

## Project Structure

- **`train_faq.py`**: The main script for training the model. It loads the dataset, tokenizes it, and fine-tunes a `distilbert-base-uncased` model using the Hugging Face `Trainer` API.
- **`evaluation.py`**: A script to evaluate the trained model on a test set (currently configured to use IMDb for demonstration, but intended for project-specific evaluation).
- **`report.py`**: Generates a classification report for the model's performance on a test dataset.
- **`answer_mapping.py`**: specific mapping dictionary (`LABEL_TO_ANSWER`) that links model output labels (integers 0-11) to actual text responses about Barkath.
- **`custom_data/resume_faq.jsonl`**: The training dataset in JSONL format, containing question-label pairs.
- **`faq_model/`**: Directory where the trained model and tokenizer are saved.
- **`results/`**: Directory for training checkpoints.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Leowaskin/LLM.git
    cd LLM
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install transformers datasets torch scikit-learn accelerate
    ```

## Usage

### Training the Model

To train the model on the custom FAQ dataset:

```bash
python train_faq.py
```

This will:
- Load data from `custom_data/resume_faq.jsonl`.
- Fine-tune `distilbert-base-uncased` for 10 epochs.
- Save the trained model to the `./faq_model` directory.

### Inference

You can use the trained model to classify new questions. Here is a basic example using the `pipeline` API:

```python
from transformers import pipeline
from answer_mapping import LABEL_TO_ANSWER

# Load the trained model
classifier = pipeline("text-classification", model="./faq_model")

# Example question
question = "What is his GPA?"

# Get prediction
result = classifier(question)[0]
label_id = int(result['label'].split('_')[-1])  # Assuming label format like LABEL_1

# Get the answer
answer = LABEL_TO_ANSWER.get(label_id, "Sorry, I don't have an answer for that.")
print(f"Question: {question}")
print(f"Answer: {answer}")
```

### Evaluation

To run the evaluation script:

```bash
python evaluation.py
```

## Labels and Topics

The model classifies questions into the following categories:

| Label | Topic |
| :--- | :--- |
| 0 | Introducation & Education |
| 1 | GPA & Academic Performance |
| 2 | Programming Languages |
| 3 | Tools & Frameworks |
| 4 | Career Goals |
| 5 | Project: FinanceFlow |
| 6 | Project: Patient Diagnosis System |
| 7 | Project: Network Design |
| 8 | Project: File Transfer Application |
| 9 | Project: School Management System |
| 10 | Volunteering Experience |
| 11 | LinkedIn Profile |

## License

[MIT License](LICENSE)
