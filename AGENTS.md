# Best Practices for Using AI Agents in the C64_AI_Companion Project

## Introduction
This document describes the best practices for using AI agents in the context of fine-tuning LLM models for Commodore 64 knowledge. The goal is to ensure efficiency, reproducibility, and maintainability of the project.

## 1. Environment Setup

### 1.1. Hardware
- **Minimum Requirements**:
  - AMD GPU with ROCm 7.x support.
  - 96GB of VRAM to handle large models like Mistral AI Ministral 3 Thinking (14B).
  - Fast storage (NVMe recommended) for data and models.

### 1.2. Software
- **Operating System**: Linux (Ubuntu 22.04 LTS recommended).
- **Drivers**: ROCm 7.x installed and configured.
- **Preinstalled Tools**:
  - `llama.cpp` for efficient inference.
  - `ollama` for model management.
  - `gh` (GitHub CLI) for repository management.

### 1.3. Python Environment
- **Python Version**: 3.10 or higher.
- **Virtual Environment**: Use `venv` or `conda` to isolate dependencies.
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

## 2. Model Management

### 2.1. Download and Storage
- **Location**: Store models in the `models/` directory.
- **Format**: Prefer optimized formats like `bf16` to reduce memory usage.
- **Example download with `ollama`**:
  ```bash
  ollama pull mistral:14b
  ```

### 2.2. Model Conversion
- Use tools like `llama.cpp` to convert models to efficient formats:
  ```bash
  python -m llama_cpp.convert --outfile models/mistral-14b.gguf --outtype q4_0 mistral-14b.gguf
  ```

## 3. Fine-Tuning

### 3.1. Data Preparation
- **Location**: Store data in the `data/` directory.
- **Format**: Use structured formats like JSON or CSV for easier processing.
- **Preprocessing**:
  - Data cleaning (remove noise, correct errors).
  - Tokenization with the base model's tokenizer.

### 3.2. Training Configuration
- **Hyperparameters**:
  - `batch_size`: Adjust based on available VRAM (example: 8 for 96GB VRAM).
  - `learning_rate`: Typical values between 1e-5 and 5e-5.
  - `epochs`: Start with 3-5 and adjust based on results.

- **Example training script**:
  ```python
  from transformers import Trainer, TrainingArguments
  
  training_args = TrainingArguments(
      output_dir="./results",
      per_device_train_batch_size=8,
      learning_rate=2e-5,
      num_train_epochs=3,
      save_steps=10_000,
      save_total_limit=2,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
  )
  
  trainer.train()
  ```

### 3.3. Optimization for AMD
- **ROCm**: Ensure PyTorch is compiled with ROCm support.
- **Flash Attention**: Enable to improve performance on AMD GPUs.
  ```bash
  pip install flash-attn --no-build-isolation
  ```

## 4. Documentation

### 4.1. Documentation Maintenance
- **Continuous Updates**: Documentation should be updated with every significant change in the project.
- **Format**: Use Markdown for easy reading and editing.
- **Structure**:
  - `README.md`: General project description.
  - `AGENTS.md`: Best practices and technical guides.
  - `docs/`: Detailed documentation (e.g., installation guides, tutorials).

### 4.2. Example Documentation Structure
```markdown
# Document Title

## Section 1
Detailed description.

### Subsection 1.1
- List of items.
- Code example.

```python
# Code example
def hello():
    print("Hello, C64!")
```
```

## 5. Evaluation and Testing

### 5.1. Evaluation Metrics
- **Accuracy**: Measure the accuracy of the model's responses.
- **Loss**: Monitor loss during training.
- **Evaluation Example**:
  ```python
  from lm_eval import evaluator
  
  results = evaluator.simple_evaluate(
      model=model,
      tasks=["hellaswag", "truthfulqa"],
      num_fewshot=0,
  )
  ```

### 5.2. Local Testing
- **Inference with `llama.cpp`**:
  ```bash
  ./main -m models/mistral-14b.gguf -p "Explain the Commodore 64"
  ```

## 6. Collaboration and Version Control

### 6.1. Git and GitHub
- **Branches**: Use branches for new features or experiments.
  ```bash
  git checkout -b feature/new-feature
  ```
- **Commits**: Descriptive and atomic messages.
  ```bash
  git commit -m "Add data preprocessing script"
  ```
- **Pull Requests**: Code review before merging to `main`.

### 6.2. Issues and Project Management
- Use GitHub Issues to track tasks and bugs.
- Label issues with labels like `bug`, `enhancement`, `documentation`.

## 7. Security

### 7.1. Handling Sensitive Data
- **File Exclusion**: Use `.gitignore` to avoid uploading sensitive data.
  ```gitignore
  # .gitignore
  *.gguf
  data/raw/*
  venv/
  ```

### 7.2. Dependencies
- **Audit**: Regularly review dependencies with tools like `safety` or `dependabot`.
  ```bash
  pip install safety
  safety check
  ```

## 8. Additional Resources

### 8.1. Official Documentation
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

### 8.2. Communities
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)

## Conclusion
By following these best practices, the C64_AI_Companion project can maintain an efficient, reproducible, and well-documented workflow. Continuous documentation updates are key to long-term success.

## Tools Used

The following tools and models were used in the development of this project:

- **Mistral AI Vibe CLI**: A CLI agent used for project initialization, file creation, and management.
- **Devstral 2 Model**: An LLM model used for generating documentation, code, and providing guidance throughout the project.

### Author
- **David R. Lopez B.**
- Email: ibitato@gmail.com
