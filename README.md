# Chatbot FAQ - UPQ (`chatbot-upq`)

Chatbot local para responder preguntas frecuentes de la Universidad Politécnica de Querétaro (UPQ). Utiliza clasificación supervisada con `TF-IDF` y `scikit-learn`, con interfaz en `Streamlit` y sin depender de APIs externas ni LLMs.

## Integrantes

| Integrante | Rol |
| :--- | :--- |
| `[Nombre Completo Integrante 1]` | Datos y corpus (`data/corpus_upq.csv`) |
| `[Nombre Completo Integrante 2]` | Preprocesamiento y modelos (`src/preprocess.py`, `notebooks/`, `models/`) |
| `[Nombre Completo Integrante 3]` | Interfaz y documentación (`app.py`, `docs/`, `README.md`) |

## Cómo funciona

- **Corpus (`data/corpus_upq.csv`):** 810 preguntas distribuidas en 15 intenciones (54 por clase), con columnas `texto`, `intencion` y `respuesta_oficial`.
- **Preprocesamiento (`src/preprocess.py`):** `limpiar_texto` pasa a minúsculas, quita acentos y signos de puntuación, y normaliza espacios.
- **Entrenamiento (`notebooks/entrenamiento_evaluacion.ipynb` y `src/entrenamiento.py`):** Partición estratificada 80/20 (`random_state=42`), `TfidfVectorizer(ngram_range=(1, 2))` ajustado solo en entrenamiento y validación cruzada de 5 pliegues con `Pipeline`.
- **Interfaz (`app.py`):** Carga `models/vectorizador.pkl` y `models/modelo.pkl`, predice la intención con Top-3 y aplica un umbral calibrado de `0.46` para descartar preguntas fuera de tema.

## Resultados

| Modelo | Accuracy (Test) | Precision Macro | Recall Macro | F1 Macro (Test) | F1 Macro CV (5-fold) | Errores (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CalibratedClassifierCV(LinearSVC)`** 🏆 | **0.8148** | **0.8359** | **0.8152** | **0.8167** | **0.8124 ± 0.0170** | **30 / 162** |
| `LogisticRegression` | 0.8148 | 0.8306 | 0.8152 | 0.8157 | 0.8037 ± 0.0235 | 30 / 162 |
| `MultinomialNB` | 0.7654 | 0.8157 | 0.7667 | 0.7654 | 0.7719 ± 0.0246 | 38 / 162 |

## Estructura

```text
chatbot-upq/
├── data/
│   └── corpus_upq.csv
├── docs/
│   ├── CONTRATOS.md
│   └── CRONOGRAMA.md
├── models/
│   ├── modelo.pkl
│   └── vectorizador.pkl
├── notebooks/
│   └── entrenamiento_evaluacion.ipynb
├── reports/
│   ├── calibracion_ood.png
│   ├── confusion_matrices_comparativa.png
│   ├── confusion_matrix_linear_svc_calibrated.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_multinomial_nb.png
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── entrenamiento.py
│   ├── preprocess.py
│   ├── test_app.py
│   └── validate_corpus.py
├── app.py
├── requirements.txt
└── README.md
```

## Instalación y uso

1. Clonar el repositorio y entrar a la carpeta:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd chatbot-upq
   ```

2. Crear y activar el entorno virtual:
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
   ```

5. (Opcional) Re-ejecutar el notebook de entrenamiento:
   ```bash
   jupyter notebook notebooks/entrenamiento_evaluacion.ipynb
   ```

## Documentación

- [Contratos del equipo](docs/CONTRATOS.md)
- [Cronograma de trabajo](docs/CRONOGRAMA.md)
