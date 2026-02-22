# Mejores Prácticas para el Uso de Agentes de IA en el Proyecto C64_AI_Companion

## Introducción
Este documento describe las mejores prácticas para el uso de agentes de IA en el contexto del fine-tuning de modelos LLM para el conocimiento del Commodore 64. El objetivo es garantizar la eficiencia, reproducibilidad y mantenimiento del proyecto.

## 1. Configuración del Entorno

### 1.1. Hardware
- **Requisitos mínimos**:
  - GPU AMD con soporte para ROCm 7.x.
  - 96GB de VRAM para manejar modelos grandes como Mistral AI Ministral 3 Thinking (14B).
  - Almacenamiento rápido (NVMe recomendado) para datos y modelos.

### 1.2. Software
- **Sistema Operativo**: Linux (Ubuntu 22.04 LTS recomendado).
- **Controladores**: ROCm 7.x instalado y configurado.
- **Herramientas preinstaladas**:
  - `llama.cpp` para inferencia eficiente.
  - `ollama` para gestión de modelos.
  - `gh` (GitHub CLI) para gestión del repositorio.

### 1.3. Entorno de Python
- **Versión de Python**: 3.10 o superior.
- **Entorno virtual**: Usar `venv` o `conda` para aislar dependencias.
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

## 2. Gestión de Modelos

### 2.1. Descarga y Almacenamiento
- **Ubicación**: Almacenar modelos en el directorio `models/`.
- **Formato**: Preferir formatos optimizados como `bf16` para reducir uso de memoria.
- **Ejemplo de descarga con `ollama`**:
  ```bash
  ollama pull mistral:14b
  ```

### 2.2. Conversión de Modelos
- Usar herramientas como `llama.cpp` para convertir modelos a formatos eficientes:
  ```bash
  python -m llama_cpp.convert --outfile models/mistral-14b.gguf --outtype q4_0 mistral-14b.gguf
  ```

## 3. Fine-Tuning

### 3.1. Preparación de Datos
- **Ubicación**: Almacenar datos en el directorio `data/`.
- **Formato**: Usar formatos estructurados como JSON o CSV para facilitar el procesamiento.
- **Preprocesamiento**:
  - Limpieza de datos (eliminar ruido, corregir errores).
  - Tokenización con el tokenizador del modelo base.

### 3.2. Configuración del Entrenamiento
- **Hiperparámetros**:
  - `batch_size`: Ajustar según la VRAM disponible (ejemplo: 8 para 96GB VRAM).
  - `learning_rate`: Valores típicos entre 1e-5 y 5e-5.
  - `epochs`: Comenzar con 3-5 y ajustar según resultados.

- **Ejemplo de script de entrenamiento**:
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

### 3.3. Optimización para AMD
- **ROCm**: Asegurar que PyTorch esté compilado con soporte para ROCm.
- **Flash Attention**: Habilitar para mejorar el rendimiento en GPUs AMD.
  ```bash
  pip install flash-attn --no-build-isolation
  ```

## 4. Documentación

### 4.1. Mantenimiento de la Documentación
- **Actualización continua**: La documentación debe actualizarse con cada cambio significativo en el proyecto.
- **Formato**: Usar Markdown para facilitar la lectura y edición.
- **Estructura**:
  - `README.md`: Descripción general del proyecto.
  - `AGENTS.md`: Mejores prácticas y guías técnicas.
  - `docs/`: Documentación detallada (ejemplo: guías de instalación, tutoriales).

### 4.2. Ejemplo de Estructura de Documentación
```markdown
# Título del Documento

## Sección 1
Descripción detallada.

### Subsección 1.1
- Lista de items.
- Código de ejemplo.

```python
# Ejemplo de código
def hello():
    print("Hello, C64!")
```
```

## 5. Evaluación y Pruebas

### 5.1. Métricas de Evaluación
- **Precisión**: Medir la exactitud de las respuestas del modelo.
- **Pérdida (Loss)**: Monitorear la pérdida durante el entrenamiento.
- **Ejemplo de evaluación**:
  ```python
  from lm_eval import evaluator
  
  results = evaluator.simple_evaluate(
      model=model,
      tasks=["hellaswag", "truthfulqa"],
      num_fewshot=0,
  )
  ```

### 5.2. Pruebas Locales
- **Inferencia con `llama.cpp`**:
  ```bash
  ./main -m models/mistral-14b.gguf -p "Explica el Commodore 64"
  ```

## 6. Colaboración y Control de Versiones

### 6.1. Git y GitHub
- **Branches**: Usar branches para nuevas características o experimentos.
  ```bash
  git checkout -b feature/nueva-caracteristica
  ```
- **Commits**: Mensajes descriptivos y atómicos.
  ```bash
  git commit -m "Añadir script de preprocesamiento de datos"
  ```
- **Pull Requests**: Revisión de código antes de fusionar a `main`.

### 6.2. Issues y Project Management
- Usar GitHub Issues para rastrear tareas y bugs.
- Etiquetar issues con labels como `bug`, `enhancement`, `documentation`.

## 7. Seguridad

### 7.1. Manejo de Datos Sensibles
- **Exclusión de archivos**: Usar `.gitignore` para evitar subir datos sensibles.
  ```gitignore
  # .gitignore
  *.gguf
  data/raw/*
  venv/
  ```

### 7.2. Dependencias
- **Auditoría**: Revisar dependencias regularmente con herramientas como `safety` o `dependabot`.
  ```bash
  pip install safety
  safety check
  ```

## 8. Recursos Adicionales

### 8.1. Documentación Oficial
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

### 8.2. Comunidades
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)

## Conclusión
Siguiendo estas mejores prácticas, el proyecto C64_AI_Companion podrá mantener un flujo de trabajo eficiente, reproducible y bien documentado. La actualización continua de la documentación es clave para el éxito a largo plazo.
