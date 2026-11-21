# Contratos Técnicos del Proyecto (`chatbot-upq`)

Este documento define los acuerdos de interfaz, formatos de datos y artefactos compartidos entre los integrantes del equipo para garantizar la integración limpia entre el corpus, el preprocesamiento, el entrenamiento en Jupyter Notebook y la aplicación en Streamlit.

---

## 1. Contrato de Datos (`data/corpus_upq.csv`)

- **Ruta:** `data/corpus_upq.csv`
- **Codificación:** `UTF-8`
- **Separador:** Coma (`,`), con comillas dobles (`"..."`) en campos que contengan comas.
- **Esquema obligatorio (3 columnas exactas):**
  1. `texto` (`str`): Pregunta o consulta formulada por el usuario/aspirante/estudiante.
  2. `intencion` (`str`): Etiqueta de clase supervisada en formato `snake_case` (`admisiones`, `becas`, `costos`, `servicios_escolares`).
  3. `respuesta_oficial` (`str`): Respuesta institucional asociada a la intención correspondiente.

> **Regla de consistencia:** Todas las filas que compartan la misma `intencion` deben mantener exactamente la misma `respuesta_oficial`, de modo que `app.py` pueda construir el diccionario `{intencion: respuesta_oficial}` directamente desde el CSV sin archivos JSON adicionales.

---

## 2. Contrato de Preprocesamiento (`src/preprocess.py`)

- **Función pública:** `limpiar_texto(texto: str) -> str`
- **Comportamiento acordado:**
  1. Si la entrada no es `str`, devuelve `""`.
  2. Convierte todo el texto a minúsculas.
  3. Elimina acentos y diacríticos mediante normalización Unicode `NFD`.
  4. Elimina signos de puntuación y caracteres no alfanuméricos (`[^a-z0-9\s]`).
  5. Colapsa espacios múltiples y recorta espacios iniciales/finales.
- **Uso obligatorio:** Tanto el notebook `notebooks/entrenamiento_evaluacion.ipynb` como la aplicación `app.py` deben importar y aplicar `limpiar_texto` antes de vectorizar cualquier texto.

---

## 3. Contrato de Entrenamiento y Artefactos (`notebooks/entrenamiento_evaluacion.ipynb`)

- **Partición de datos:** `train_test_split` con proporción **80% entrenamiento / 20% prueba**, `stratify=y` y `random_state=42`.
- **Prevención de fuga de datos (*data leakage*):**
  - `TfidfVectorizer(ngram_range=(1, 2))` se ajusta **únicamente** con `X_train` (`fit_transform`).
  - `X_test` solo se transforma (`transform`).
- **Modelos evaluados:**
  1. `LogisticRegression(max_iter=1000, random_state=42)`
  2. `CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)`
  3. `MultinomialNB()`
- **Criterio de selección:** Mayor `F1-score macro` sobre el conjunto de prueba (20%).
- **Artefactos exportados (con `joblib.dump` y versionados en Git):**
  - `models/vectorizador.pkl`: Instancia entrenada de `TfidfVectorizer`.
  - `models/modelo.pkl`: Clasificador ganador con soporte para `predict_proba` y atributo `classes_`.

---

## 4. Contrato de Inferencia e Interfaz (`app.py`)

- **Entradas requeridas al iniciar:**
  - `models/modelo.pkl`
  - `models/vectorizador.pkl`
  - `data/corpus_upq.csv`
- **Flujo de inferencia por consulta:**
  1. Limpiar la pregunta del usuario con `limpiar_texto(consulta)`.
  2. Transformar el texto limpio con `vectorizador.transform([texto_limpio])`.
  3. Calcular probabilidades por clase con `modelo.predict_proba(X_vectorizado)[0]`.
  4. Determinar la intención con mayor probabilidad (`confianza = max(probabilidades)`).
  5. Si `confianza < umbral_confianza` (por defecto `0.45`), devolver la respuesta `_fallback`; en caso contrario, devolver `respuestas[intencion]`.

