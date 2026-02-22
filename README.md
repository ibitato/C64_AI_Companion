# C64_AI_Companion

## Description
This project aims to create a fine-tuned LLM (Large Language Model) with specific knowledge about the Commodore 64, using the Mistral AI Ministral 3 Thinking 14B base model in bf16 format. The fine-tuning will be performed on a Corsair AI Workstation server with an AMD GPU and 96GB of VRAM.

## System Requirements

### Hardware
- **GPU**: AMD with ROCm 7.x support.
- **VRAM**: 96GB minimum.
- **Storage**: NVMe recommended for data and models.

### Software
- **Operating System**: Linux (Ubuntu 22.04 LTS recommended).
- **Drivers**: ROCm 7.x.
- **Preinstalled Tools**:
  - `llama.cpp`
  - `ollama`
  - `gh` (GitHub CLI)

## Project Structure

```
C64_AI_Companion/
├── data/                  # Training and validation data
├── models/               # Saved models
├── scripts/              # Training and evaluation scripts
├── docs/                 # Additional documentation
├── AGENTS.md             # Best practices for AI agents
├── README.md             # This file
└── .gitignore            # Files ignored by Git
```

## Initial Setup

### 1. Clone the Repository
```bash
gh repo clone ibitato/C64_AI_Companion
cd C64_AI_Companion
```

### 2. Set Up the Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch transformers datasets peft accelerate flash-attn lm-eval
```

## Data Preparation

### 1. Download Data
Training data should be stored in the `data/` directory. Example structure:

```
data/
├── raw/          # Raw data
├── processed/   # Processed data ready for training
└── val/          # Validation data
```

### 2. Preprocessing
Example script for preprocessing data:

```python
from datasets import load_dataset

# Load data
dataset = load_dataset("csv", data_files={"train": "data/raw/train.csv", "val": "data/raw/val.csv"})

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save processed data
tokenized_datasets.save_to_disk("data/processed")
```

## Model Fine-Tuning

### 1. Download Base Model
```bash
ollama pull mistral:14b
```

### 2. Set Up Training Script
Example training script (`scripts/train.py`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-3-Thinking-14B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-3-Thinking-14B")

# Load data
train_dataset = load_from_disk("data/processed/train")
val_dataset = load_from_disk("data/processed/val")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
trainer.save_model("models/c64-finetuned")
```

### 3. Run Training
```bash
python scripts/train.py
```

## Model Evaluation

### 1. Load Trained Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/c64-finetuned")
tokenizer = AutoTokenizer.from_pretrained("models/c64-finetuned")
```

### 2. Evaluate with LM Evaluation Harness
```python
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag", "truthfulqa"],
    num_fewshot=0,
)
```

### 3. Manual Testing
```python
input_text = "Explain the Commodore 64"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Contribution

### 1. Create a Branch
```bash
git checkout -b feature/new-feature
```

### 2. Make Commits
```bash
git commit -m "Add new feature"
```

### 3. Create Pull Request
```bash
gh pr create --fill
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For questions or suggestions, open an issue in the repository or contact the maintainer.

## Additional Resources
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Credits

This project was developed using the following tools and models:

- **Mistral AI Vibe CLI**: Used as the development agent for project initialization and management.
- **Devstral 2 Model**: Utilized as the development LLM for generating documentation and code.

### Author
- **David R. Lopez B.**
- Email: ibitato@gmail.com
