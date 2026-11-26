# 🎓 Chatbot FAQ — Universidad Politécnica de Querétaro (`chatbot-upq`)

Sistema local de clasificación supervisada de intenciones y recuperación de respuestas oficiales para preguntas frecuentes (FAQ) de la **Universidad Politécnica de Querétaro (UPQ)**, desarrollado íntegramente en Python con `scikit-learn` y desplegado mediante una interfaz conversacional en `Streamlit`.

> **Restricción cumplida:** El sistema opera **100% de forma local** con algoritmos clásicos de aprendizaje supervisado (`TF-IDF` + clasificadores de `scikit-learn`), sin depender de APIs comerciales, LLMs ni servicios externos.

---

## 👥 Integrantes del Equipo

| Integrante (Nombre Completo) | Rol en el Proyecto |
| :--- | :--- |
| `[Nombre Completo Integrante 1]` | Ingeniería de Datos y Curaduría del Corpus (`data/corpus_upq.csv`, `src/validate_corpus.py`) |
| `[Nombre Completo Integrante 2]` | Preprocesamiento, Experimentación ML y Evaluación (`src/preprocess.py`, `notebooks/`, `models/`) |
| `[Nombre Completo Integrante 3]` | Interfaz Streamlit, Calibración de Fallback, Integración y Documentación (`app.py`, `docs/`, `README.md`) |

*(Actualizar con los nombres completos antes de la entrega final en Classroom.)*

---

## 🏗️ Arquitectura y Metodología de Machine Learning

1. **Corpus unificado y validado (`data/corpus_upq.csv`):**
   - Archivo único en UTF-8 con exactamente 3 columnas: `texto,intencion,respuesta_oficial`.
   - **375 consultas etiquetadas** distribuidas de forma perfectamente balanceada en **15 intenciones oficiales de la UPQ** (25 variantes por clase combinando preguntas formales, lenguaje informal estudiantil, búsquedas tipo *keyword* y sinónimos).
   - Validación automática de integridad mediante `src/validate_corpus.py` (0 nulos, 0 duplicados exactos, 1 única respuesta oficial por intención).
2. **Preprocesamiento determinista (`src/preprocess.py`):**
   - `limpiar_texto(texto)` normaliza minúsculas, elimina acentos mediante Unicode `NFD`, remueve signos de puntuación y colapsa espacios. Se importa de manera idéntica en entrenamiento e inferencia.
3. **Extracción de características y prevención de *data leakage*:**
   - Partición estratificada **80% entrenamiento (300) / 20% prueba (75)** (`random_state=42`).
   - `TfidfVectorizer(ngram_range=(1, 2))` ajustado **exclusivamente** con `X_train` (`fit_transform`) y aplicado sobre `X_test` (`transform`), generando 2,326 características de unigramas y bigramas.
   - Validación cruzada estratificada (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) encapsulando TF-IDF dentro de un `Pipeline` independiente por pliegue.
4. **Comparativa de 3 modelos supervisados y calibración de *fallback*:**
   - Se compararon `LogisticRegression`, `CalibratedClassifierCV(LinearSVC)` y `MultinomialNB` sobre la misma partición.
   - Se evaluaron 12 consultas fuera de dominio (OOD) frente a las 75 consultas válidas de prueba, calibrando el umbral de confianza óptimo en **`0.45`**.
5. **Inferencia desacoplada en Streamlit (`app.py`):**
   - Carga por separado `models/vectorizador.pkl` y `models/modelo.pkl`, construye `{intencion: respuesta_oficial}` leyendo `data/corpus_upq.csv`, maneja entradas vacías/cortas, incluye botones de prueba rápida y expone en modo depuración la confianza máxima con barra de progreso y el **Top-3** de intenciones candidatas.

---

## 📊 Resultados Experimentales

Resultados obtenidos tras ejecutar [`notebooks/entrenamiento_evaluacion.ipynb`](notebooks/entrenamiento_evaluacion.ipynb) sobre el corpus de 375 registros (15 intenciones, split 80/20 estratificado):

| Modelo | Accuracy (Test) | Precision Macro | Recall Macro | F1-score Macro (Test) | F1-score Macro CV (5-fold) | Errores en Test (n=75) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CalibratedClassifierCV(LinearSVC)` (Ganador)** | **0.9467** | **0.9533** | **0.9467** | **0.9444** | **0.7464 ± 0.0650** | **4 / 75** |
| `LogisticRegression` | 0.9067 | 0.9176 | 0.9067 | 0.9037 | 0.7301 ± 0.0601 | 7 / 75 |
| `MultinomialNB` | 0.8400 | 0.8640 | 0.8400 | 0.8349 | 0.7217 ± 0.0507 | 12 / 75 |

- **Umbral de *fallback* calibrado:** **`0.45`** (alcanza **98.21%** de precisión sobre consultas válidas aceptadas y **75.00%** de rechazo correcto en consultas fuera de dominio).

---

## 📁 Estructura del Proyecto

```text
chatbot-upq/
├── data/
│   └── corpus_upq.csv                            # Corpus único UTF-8 (375 filas, 15 intenciones)
├── docs/
│   ├── CONTRATOS.md                              # Contratos técnicos, datos y flujo de Git
│   └── CRONOGRAMA.md                             # Plan de trabajo, responsables y checklist
├── models/
│   ├── modelo.pkl                                # Clasificador ganador serializado (joblib)
│   └── vectorizador.pkl                          # TfidfVectorizer ajustado en train (joblib)
├── notebooks/
│   └── entrenamiento_evaluacion.ipynb            # Entrenamiento, CV, matrices, errores y OOD (con salidas)
├── reports/
│   ├── calibracion_fallback.png                  # Gráfica de calibración In-Domain vs. OOD
│   ├── confusion_matrix_linear_svc_calibrated.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_multinomial_nb.png
│   └── metrics.json                              # Resumen cuantitativo en JSON
├── src/
│   ├── __init__.py
│   ├── preprocess.py                             # Normalización determinista de texto (limpiar_texto)
│   └── validate_corpus.py                        # Validador automático de calidad de datos
├── app.py                                        # Interfaz conversacional en Streamlit
├── requirements.txt                              # Dependencias exactas del proyecto
├── .gitignore                                    # Ignora .venv/__pycache__ y rastrea models/*.pkl
└── README.md
```

---

## 🚀 Instalación y Ejecución Paso a Paso

### 1. Clonar el repositorio e ingresar a la carpeta

```bash
git clone <URL_DEL_REPOSITORIO>
cd chatbot-upq
```

### 2. Crear y activar el entorno virtual

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

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Validar la integridad del corpus

```bash
python src/validate_corpus.py
```

El comando debe finalizar con `RESULTADO: APROBADO (0 errores encontrados)` y código de salida `0`.

### 5. Re-ejecutar el notebook de entrenamiento y evaluación (opcional)

Los artefactos `models/modelo.pkl`, `models/vectorizador.pkl` y las salidas del notebook ya se encuentran generados y versionados en el repositorio. Si modificas `data/corpus_upq.csv` o `src/preprocess.py`, re-ejecuta el notebook:

**Modo interactivo (Jupyter):**
```bash
jupyter notebook notebooks/entrenamiento_evaluacion.ipynb
```

**Modo automático por consola (guardando salidas en el `.ipynb`):**
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/entrenamiento_evaluacion.ipynb
```

### 6. Iniciar la aplicación conversacional en Streamlit

```bash
streamlit run app.py
```

Abre en tu navegador la URL local indicada por Streamlit (generalmente `http://localhost:8501`).

---

## 📚 Documentación del Equipo

- [Contratos Técnicos y Flujo Git (`docs/CONTRATOS.md`)](docs/CONTRATOS.md)
- [Cronograma de Trabajo y Checklist (`docs/CRONOGRAMA.md`)](docs/CRONOGRAMA.md)
