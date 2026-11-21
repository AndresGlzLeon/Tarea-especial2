# chatbot-upq

Sistema local de clasificación y recuperación de preguntas frecuentes (FAQ) para la **Universidad Politécnica de Querétaro (UPQ)** basado exclusivamente en aprendizaje supervisado con `scikit-learn` y una interfaz conversacional en `Streamlit`.

## Estructura del proyecto

```text
chatbot-upq/
├── data/
│   └── corpus_upq.csv                    # Corpus etiquetado (texto, intencion, respuesta_oficial)
├── docs/
│   ├── CONTRATOS.md                      # Acuerdos técnicos e interfaces entre módulos
│   └── CRONOGRAMA.md                     # Plan de trabajo, hitos y responsables del equipo
├── models/
│   ├── modelo.pkl                        # Clasificador ganador serializado
│   └── vectorizador.pkl                  # Vectorizador TF-IDF entrenado sobre train
├── notebooks/
│   └── entrenamiento_evaluacion.ipynb    # Notebook de entrenamiento, evaluación y exportación
├── reports/                              # Matrices de confusión (.png) y metrics.json
├── src/
│   ├── __init__.py
│   └── preprocess.py                     # Normalización y limpieza de texto (limpiar_texto)
├── app.py                                # Aplicación web interactiva en Streamlit
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

### 3. Entrenar y evaluar los modelos en Jupyter Notebook

Abre y ejecuta todas las celdas del notebook `notebooks/entrenamiento_evaluacion.ipynb`:

```bash
jupyter notebook notebooks/entrenamiento_evaluacion.ipynb
```

El notebook realiza:
- Carga de `data/corpus_upq.csv` (`texto,intencion,respuesta_oficial`).
- Limpieza de texto importando `limpiar_texto` desde `src/preprocess.py`.
- Partición estratificada 80/20 (`train_test_split` con `stratify=y`).
- Ajuste de `TfidfVectorizer` únicamente sobre el conjunto de entrenamiento.
- Comparativa entre `LogisticRegression`, `CalibratedClassifierCV(LinearSVC)` y `MultinomialNB` mediante matrices de confusión y `F1-score macro`.
- Exportación por separado de `models/vectorizador.pkl` y `models/modelo.pkl` usando `joblib.dump`.

### 4. Iniciar la aplicación de chat

```bash
streamlit run app.py
```

## Documentación del equipo

- [Contratos técnicos (`docs/CONTRATOS.md`)](docs/CONTRATOS.md)
- [Cronograma de trabajo (`docs/CRONOGRAMA.md`)](docs/CRONOGRAMA.md)
