# chatbot-upq

Sistema local de clasificación y recuperación de preguntas frecuentes (FAQ) para la **Universidad Politécnica de Querétaro (UPQ)** basado exclusivamente en aprendizaje supervisado con `scikit-learn` y una interfaz conversacional en `Streamlit`.

## Estructura del proyecto

```text
chatbot-upq/
├── data/
│   ├── corpus_upq.csv        # Corpus etiquetado (pregunta, intencion)
│   └── respuestas_upq.json   # Respuestas oficiales por intención y _fallback
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Normalización y limpieza de texto
│   └── train.py              # Entrenamiento, evaluación y selección del modelo
├── models/                   # Modelo ganador serializado (.joblib)
├── reports/                  # Matrices de confusión (.png) y metrics.json
├── app.py                    # Aplicación web interactiva en Streamlit
├── requirements.txt
└── README.md
```

## Instalación y ejecución

### 1. Crear y activar el entorno virtual

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Entrenar y evaluar los modelos

```bash
python src/train.py
```

Este comando entrenará 3 clasificadores (`LogisticRegression`, `CalibratedClassifierCV(LinearSVC)` y `MultinomialNB`), generará las matrices de confusión y `reports/metrics.json`, y guardará el modelo ganador en `models/modelo_upq.joblib`.

### 4. Iniciar la aplicación de chat

```bash
streamlit run app.py
```

