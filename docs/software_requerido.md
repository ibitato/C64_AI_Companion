# Software Requerido para Fine-Tuning con AMD ROCm 7.x

## Introducción
Este documento lista el software necesario para realizar fine-tuning de modelos LLM en GPUs AMD utilizando ROCm 7.x, basado en la investigación realizada.

## Herramientas y Librerías

### 1. Herramientas de AMD ROCm
- **ROCm Platform**: Plataforma de software para aceleración GPU en AMD.
  - Versión requerida: 7.x
  - [Documentación oficial](https://rocm.docs.amd.com/)

- **GPU-accelerated libraries**: Librerías optimizadas para AMD GPUs.

### 2. Frameworks de Deep Learning
- **PyTorch**: Framework principal para entrenamiento de modelos.
  - Versión con soporte para ROCm.
  - Instalación:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/rocm7.0
    ```

- **TensorFlow**: Alternativa a PyTorch.
  - Versión con soporte para ROCm.

- **JAX**: Framework para computación numérica.
  - Versión con soporte para ROCm.

### 3. Librerías para LLM
- **Transformers**: Librería de Hugging Face para modelos LLM.
  - Instalación:
    ```bash
    pip install transformers
    ```

- **Accelerate**: Librería para distribuir el entrenamiento.
  - Instalación:
    ```bash
    pip install accelerate
    ```

- **PEFT (Parameter-Efficient Fine-Tuning)**: Para optimizar el uso de VRAM.
  - Instalación:
    ```bash
    pip install peft
    ```

- **Datasets**: Para manejo de datos de entrenamiento.
  - Instalación:
    ```bash
    pip install datasets
    ```

- **Flash Attention**: Optimización para atención en modelos grandes.
  - Instalación:
    ```bash
    pip install flash-attn --no-build-isolation
    ```

### 4. Herramientas de Evaluación
- **LM Evaluation Harness**: Para evaluar el rendimiento del modelo.
  - Instalación:
    ```bash
    pip install lm-eval
    ```

### 5. Herramientas Adicionales
- **llama.cpp**: Para inferencia eficiente.
  - Instalación:
    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    ```

- **ollama**: Para gestión de modelos.
  - Instalación:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

## Configuración del Entorno

### 1. Instalar ROCm 7.x
Sigue las instrucciones oficiales para instalar ROCm en tu sistema:
- [Guía de instalación de ROCm](https://rocm.docs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

### 2. Configurar Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install torch transformers accelerate peft datasets flash-attn lm-eval
```

## Recursos Adicionales
- [Documentación de ROCm para Fine-Tuning](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/index.html)
- [Guía para Deploy de LLM con AMD](https://gpuopen.com/learn/pytorch-windows-amd-llm-guide/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## Conclusión
Este documento proporciona una lista completa del software necesario para realizar fine-tuning de modelos LLM en GPUs AMD con ROCm 7.x. Asegúrate de instalar y configurar todas las herramientas y librerías mencionadas para un flujo de trabajo óptimo.
