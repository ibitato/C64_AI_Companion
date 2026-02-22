# Required Software for Fine-Tuning with AMD ROCm 7.x

## Introduction
This document lists the necessary software for fine-tuning LLM models on AMD GPUs using ROCm 7.x, based on the research conducted.

## Tools and Libraries

### 1. AMD ROCm Tools
- **ROCm Platform**: Software platform for GPU acceleration on AMD.
  - Required version: 7.x
  - [Official Documentation](https://rocm.docs.amd.com/)

- **GPU-accelerated libraries**: Libraries optimized for AMD GPUs.

### 2. Deep Learning Frameworks
- **PyTorch**: Main framework for model training.
  - Version with ROCm support.
  - Installation:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/rocm7.0
    ```

- **TensorFlow**: Alternative to PyTorch.
  - Version with ROCm support.

- **JAX**: Framework for numerical computation.
  - Version with ROCm support.

### 3. LLM Libraries
- **Transformers**: Hugging Face library for LLM models.
  - Installation:
    ```bash
    pip install transformers
    ```

- **Accelerate**: Library for distributing training.
  - Installation:
    ```bash
    pip install accelerate
    ```

- **PEFT (Parameter-Efficient Fine-Tuning)**: To optimize VRAM usage.
  - Installation:
    ```bash
    pip install peft
    ```

- **Datasets**: For handling training data.
  - Installation:
    ```bash
    pip install datasets
    ```

- **Flash Attention**: Optimization for attention in large models.
  - Installation:
    ```bash
    pip install flash-attn --no-build-isolation
    ```

### 4. Evaluation Tools
- **LM Evaluation Harness**: To evaluate model performance.
  - Installation:
    ```bash
    pip install lm-eval
    ```

### 5. Additional Tools
- **llama.cpp**: For efficient inference.
  - Installation:
    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    ```

- **ollama**: For model management.
  - Installation:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

## Environment Configuration

### 1. Install ROCm 7.x
Follow the official instructions to install ROCm on your system:
- [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch transformers accelerate peft datasets flash-attn lm-eval
```

## Additional Resources
- [ROCm Documentation for Fine-Tuning](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/index.html)
- [Guide for Deploying LLM with AMD](https://gpuopen.com/learn/pytorch-windows-amd-llm-guide/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## Conclusion
This document provides a complete list of the software needed to perform fine-tuning of LLM models on AMD GPUs with ROCm 7.x. Make sure to install and configure all the mentioned tools and libraries for an optimal workflow.

## Development Tools

The following tools were used in the development of this project:

- **Mistral AI Vibe CLI**: A CLI agent used for project initialization, file creation, and management.
- **Devstral 2 Model**: An LLM model used for generating documentation, code, and providing guidance throughout the project.

### Author
- **David R. Lopez B.**
- Email: ibitato@gmail.com
