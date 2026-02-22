# C64_AI_Companion

## Descripción
Este proyecto tiene como objetivo crear un modelo LLM (Large Language Model) tuneado con conocimiento específico sobre el Commodore 64, utilizando el modelo base Mistral AI Ministral 3 Thinking de 14B en formato bf16. El fine-tuning se realizará en un servidor Corsair AI Workstation con GPU AMD y 96GB de VRAM.

## Requisitos del Sistema

### Hardware
- **GPU**: AMD con soporte para ROCm 7.x.
- **VRAM**: 96GB mínimo.
- **Almacenamiento**: NVMe recomendado para datos y modelos.

### Software
- **Sistema Operativo**: Linux (Ubuntu 22.04 LTS recomendado).
- **Controladores**: ROCm 7.x.
- **Herramientas preinstaladas**:
  - `llama.cpp`
  - `ollama`
  - `gh` (GitHub CLI)

## Estructura del Proyecto

```
C64_AI_Companion/
├── data/                  # Datos de entrenamiento y validación
├── models/               # Modelos guardados
├── scripts/              # Scripts para entrenamiento y evaluación
├── docs/                 # Documentación adicional
├── AGENTS.md             # Mejores prácticas para agentes de IA
├── README.md             # Este archivo
└── .gitignore            # Archivos ignorados por Git
```

## Configuración Inicial

### 1. Clonar el Repositorio
```bash
gh repo clone ibitato/C64_AI_Companion
cd C64_AI_Companion
```

### 2. Configurar el Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install torch transformers datasets peft accelerate flash-attn lm-eval
```

## Preparación de Datos

### 1. Descargar Datos
Los datos de entrenamiento deben almacenarse en el directorio `data/`. Ejemplo de estructura:

```
data/
├── raw/          # Datos sin procesar
├── processed/   # Datos procesados y listos para entrenamiento
└── val/          # Datos de validación
```

### 2. Preprocesamiento
Ejemplo de script para preprocesar datos:

```python
from datasets import load_dataset

# Cargar datos
dataset = load_dataset("csv", data_files={"train": "data/raw/train.csv", "val": "data/raw/val.csv"})

# Tokenizar datos
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Guardar datos procesados
tokenized_datasets.save_to_disk("data/processed")
```

## Fine-Tuning del Modelo

### 1. Descargar Modelo Base
```bash
ollama pull mistral:14b
```

### 2. Configurar Script de Entrenamiento
Ejemplo de script de entrenamiento (`scripts/train.py`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Cargar modelo y tokenizador
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-3-Thinking-14B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-3-Thinking-14B")

# Cargar datos
train_dataset = load_from_disk("data/processed/train")
val_dataset = load_from_disk("data/processed/val")

# Configurar argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,
)

# Inicializar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Entrenar modelo
trainer.train()

# Guardar modelo
trainer.save_model("models/c64-finetuned")
```

### 3. Ejecutar Entrenamiento
```bash
python scripts/train.py
```

## Evaluación del Modelo

### 1. Cargar Modelo Entrenado
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/c64-finetuned")
tokenizer = AutoTokenizer.from_pretrained("models/c64-finetuned")
```

### 2. Evaluar con LM Evaluation Harness
```python
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag", "truthfulqa"],
    num_fewshot=0,
)
```

### 3. Pruebas Manuales
```python
input_text = "Explica el Commodore 64"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Contribución

### 1. Crear un Branch
```bash
git checkout -b feature/nueva-caracteristica
```

### 2. Hacer Commits
```bash
git commit -m "Añadir nueva característica"
```

### 3. Crear Pull Request
```bash
gh pr create --fill
```

## Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Contacto
Para preguntas o sugerencias, abre un issue en el repositorio o contacta al mantenedor.

## Recursos Adicionales
- [Documentación de ROCm](https://rocm.docs.amd.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
