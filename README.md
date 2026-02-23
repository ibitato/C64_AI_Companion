# C64_AI_Companion

## Description
This project aims to create a fine-tuned LLM (Large Language Model) with specific knowledge about the Commodore 64, using the Ministral 3 8B Thinking base model in bf16/fp16 format. The fine-tuning will be performed on a Corsair AI Workstation server with an AMD GPU.

## System Requirements

### Hardware
- **CPU**: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S Graphics (16 cores, 32 threads).
- **GPU**: Radeon 8060S Graphics (RADV GFX1151).
- **VRAM**: 74.32 GiB.
- **RAM**: 30 GiB.
- **Storage**: NVMe (3.7 TB total with LVM).

### Software
- **Operating System**: Fedora Linux 43 (Server Edition).
- **Kernel**: 6.18.8-200.fc43.x86_64.
- **Drivers**: ROCm 6.4.4 (required for GPU acceleration).
- **Preinstalled Tools**:
  - `llama.cpp`
  - `ollama`
  - `gh` (GitHub CLI)
  - `vulkan-tools`
  - `mesa-demos`

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
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install ROCm 6.4.4 (Optional, if not already installed)
```bash
# Install ROCm 6.4.4 for GPU acceleration
sudo dnf install rocm
```

### 5. Install PyTorch with ROCm 6.4 Support
```bash
# Install PyTorch with ROCm 6.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

### 6. Install OCR Tools (recommended for scanned manuals)
```bash
sudo dnf install ocrmypdf tesseract tesseract-langpack-eng poppler-utils
sudo dnf install ghostscript unpaper
```

## Data Preparation

### 1. Put C64 docs in `c64_docs/`
This repository already expects PDFs/HTML manuals under `c64_docs/`.

### 2. Run the end-to-end pipeline
```bash
python scripts/data_pipeline.py --stage all --allow-ocr
```

This generates:
- `data/interim/manifest/manifest.parquet`
- `data/interim/extracted/pages.parquet`
- `data/interim/normalized/pages_normalized.parquet`
- `data/interim/dedup/pages_dedup.parquet`
- `data/processed/dapt/{train,validation,test}.parquet`
- `data/processed/sft/{train,validation,test}.jsonl`
- `data/processed/validation_report.json`

### 3. Generate QC markdown report
```bash
python scripts/data_qc_report.py \
  --input data/processed/validation_report.json \
  --output docs/data_qc_report.md
```

## Model Fine-Tuning

### 1. Download Base Model
```bash
# Download Ministral 3 8B Thinking and place it in:
# models/Ministral-3-8B-Thinking
```

### 2. Run Training
Use the unified script with mode selection:

### DAPT only
```bash
python scripts/fine_tune_mistral_8b.py \
  --mode dapt \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --output-dir models/fine-tuned-dapt \
  --use-lora
```

### SFT only
```bash
python scripts/fine_tune_mistral_8b.py \
  --mode sft \
  --model-path models/Ministral-3-8B-Thinking \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned-sft \
  --use-lora
```

### DAPT + SFT
```bash
python scripts/fine_tune_mistral_8b.py \
  --mode both \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned \
  --use-lora
```

## Model Evaluation

### 1. Load Trained Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("models/fine-tuned")
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
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

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
