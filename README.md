# Chatbot FAQ - UPQ (`chatbot-upq`)

Chatbot local para responder preguntas frecuentes de la Universidad Politécnica de Querétaro (UPQ). Utiliza clasificación supervisada con `TF-IDF` y `scikit-learn`, con interfaz en `Streamlit` y sin depender de APIs externas ni LLMs.

## Integrantes

| Integrante | Rol |
| :--- | :--- |
| `[Nombre Completo Integrante 1]` | Datos y corpus (`data/corpus_upq.csv`) |
| `[Nombre Completo Integrante 2]` | Preprocesamiento y modelos (`src/preprocess.py`, `notebooks/`, `models/`) |
| `[Nombre Completo Integrante 3]` | Interfaz y documentación (`app.py`, `docs/`, `README.md`) |

## Cómo funciona

- **Corpus (`data/corpus_upq.csv`):** 375 preguntas distribuidas en 15 intenciones (25 por clase), con columnas `texto`, `intencion` y `respuesta_oficial`.
- **Preprocesamiento (`src/preprocess.py`):** `limpiar_texto` pasa a minúsculas, quita acentos y signos de puntuación, y normaliza espacios.
- **Entrenamiento (`notebooks/entrenamiento_evaluacion.ipynb`):** Partición estratificada 80/20 (`random_state=42`), `TfidfVectorizer(ngram_range=(1, 2))` ajustado solo en entrenamiento y validación cruzada de 5 pliegues.
- **Interfaz (`app.py`):** Carga `models/vectorizador.pkl` y `models/modelo.pkl`, predice la intención y aplica un umbral de confianza de `0.45` para descartar preguntas fuera de tema.

## Resultados

| Modelo | Accuracy (Test) | Precision Macro | Recall Macro | F1 Macro (Test) | F1 Macro CV (5-fold) | Errores (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CalibratedClassifierCV(LinearSVC)`** | **0.9467** | **0.9533** | **0.9467** | **0.9444** | **0.7464 ± 0.0650** | **4 / 75** |
| `LogisticRegression` | 0.9067 | 0.9176 | 0.9067 | 0.9037 | 0.7301 ± 0.0601 | 7 / 75 |
| `MultinomialNB` | 0.8400 | 0.8640 | 0.8400 | 0.8349 | 0.7217 ± 0.0507 | 12 / 75 |

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
│   ├── confusion_matrix_linear_svc_calibrated.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_multinomial_nb.png
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── entrenamiento.py
│   └── preprocess.py
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
