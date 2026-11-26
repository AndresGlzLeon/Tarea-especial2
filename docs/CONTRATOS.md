# Contratos Técnicos del Proyecto (`chatbot-upq`)

Este documento define los acuerdos de interfaz, formatos de datos, artefactos de Machine Learning y reglas de colaboración del equipo para garantizar una integración limpia, reproducible y auditable entre el corpus, el preprocesamiento, el entrenamiento en Jupyter Notebook y la aplicación en Streamlit.

---

## 1. Contrato de Datos (`data/corpus_upq.csv`)

- **Ruta única:** `data/corpus_upq.csv` (prohibido usar archivos JSON o fuentes externas adicionales para las respuestas).
- **Codificación:** `UTF-8` estricta.
- **Separador:** Coma (`,`), con comillas dobles (`"..."`) en campos que contengan comas.
- **Esquema obligatorio (exactamente 3 columnas):**
  1. `texto` (`str`): Pregunta o consulta formulada por el usuario, aspirante o estudiante.
  2. `intencion` (`str`): Etiqueta de clase supervisada en formato `snake_case` estricto (`^[a-z0-9]+(?:_[a-z0-9]+)*$`), sin acentos, sin mayúsculas ni espacios sobrantes.
  3. `respuesta_oficial` (`str`): Respuesta institucional única asociada a la intención correspondiente.
- **Volumen mínimo obligatorio:**
  - Entre **10 y 15 intenciones** oficiales de la UPQ (implementación objetivo: **15 intenciones** sin traslapes semánticos).
  - Al menos **20 variantes únicas** de preguntas por intención (implementación objetivo: **25 variantes por intención**, 375 filas totales).
- **Reglas estrictas de integridad (verificadas automáticamente por `src/validate_corpus.py`):**
  - **0 valores nulos** y **0 cadenas vacías** (o solo espacios) en cualquiera de las 3 columnas.
  - **0 duplicados exactos** en la columna `texto` en todo el corpus.
  - **1 única `respuesta_oficial` por `intencion`**: todas las filas con la misma `intencion` deben contener exactamente la misma cadena en `respuesta_oficial`, para que `app.py` construya el diccionario `{intencion: respuesta_oficial}` directamente desde el CSV.
  - **Sin clase de respaldo en el CSV:** los casos fuera de dominio o de baja confianza se manejan exclusivamente mediante el umbral de probabilidad en `app.py`.

### Catálogo de 15 intenciones oficiales de la UPQ

| # | Intención (`snake_case`) | Alcance semántico |
| :---: | :--- | :--- |
| 1 | `admisiones` | Convocatorias de ingreso, fichas, examen de admisión, registro y requisitos de aspirantes |
| 2 | `oferta_educativa` | Ingenierías, licenciaturas, posgrados, modelo cuatrimestral y planes de estudio |
| 3 | `costos_pagos` | Tabulador de cuotas, líneas de captura, Multipagos, bancos autorizados y recargos |
| 4 | `becas` | Becas institucionales (excelencia, socioeconómica, deportiva, cultural), externas y renovación |
| 5 | `servicios_escolares` | Constancias, kardex, credenciales, reinscripciones, bajas, seguro facultativo IMSS |
| 6 | `calendario_horarios` | Fechas de inicio/fin de cuatrimestre, vacaciones, exámenes y horarios de atención/clases |
| 7 | `estancias_estadias` | Estancia I, Estancia II, Estadía profesional, cartas de presentación y liberación |
| 8 | `servicio_social` | Requisitos de créditos, horas obligatorias, registro, reportes y carta de liberación |
| 9 | `titulacion` | Modalidades de titulación, requisitos académicos, trámite de título y cédula profesional |
| 10 | `idiomas_ingles` | Niveles de inglés obligatorios, examen de colocación, certificaciones y centro de idiomas |
| 11 | `ubicacion_transporte` | Dirección del campus (Carretera Estatal 420 S/N, El Marqués, Qro.), rutas y transporte |
| 12 | `actividades_extracurriculares` | Talleres culturales, equipos deportivos, clubes estudiantiles y actividades formativas |
| 13 | `biblioteca_instalaciones` | Préstamo de libros, salas de estudio, laboratorios, cafetería y servicios del campus |
| 14 | `plataforma_sistemas` | Sistema Integral de Información (SII), correo institucional, aula virtual y recuperación de contraseña |
| 15 | `contacto_directorio` | Teléfonos oficiales, correos de departamentos, portal institucional y redes oficiales |

---

## 2. Contrato de Validación de Datos (`src/validate_corpus.py`)

- **Comando:** `python src/validate_corpus.py`
- **Verificaciones automatizadas:** codificación UTF-8, exactamente 3 columnas requeridas, cero nulos/vacíos, entre 10 y 15 intenciones válidas en `snake_case`, mínimo 20 variantes únicas por intención, cero duplicados exactos en `texto` y consistencia 1 a 1 entre `intencion` y `respuesta_oficial`.
- **Códigos de salida:** `0` si el corpus cumple el 100% del contrato; `1` si existe cualquier incumplimiento.

---

## 3. Contrato de Preprocesamiento (`src/preprocess.py`)

- **Función pública:** `limpiar_texto(texto: str) -> str`
- **Comportamiento acordado:**
  1. Si la entrada no es `str` (por ejemplo `None` o `NaN`) o está vacía, devuelve `""`.
  2. Convierte todo el texto a minúsculas.
  3. Elimina acentos y diacríticos mediante normalización Unicode `NFD`.
  4. Elimina signos de puntuación y caracteres no alfanuméricos (`[^a-z0-9\s]`).
  5. Colapsa espacios múltiples y recorta espacios iniciales y finales.
- **Uso obligatorio:** Tanto `notebooks/entrenamiento_evaluacion.ipynb` como `app.py` deben importar y aplicar `limpiar_texto` antes de vectorizar cualquier texto.

---

## 4. Contrato de Entrenamiento y Artefactos (`notebooks/entrenamiento_evaluacion.ipynb`)

- **Restricción de tecnología:** Algoritmos de aprendizaje supervisado 100% locales en Python (`scikit-learn`). Prohibido el uso de APIs comerciales, LLMs o servicios externos.
- **Partición de datos:** `train_test_split` con proporción **80% entrenamiento / 20% prueba**, `stratify=y` y `random_state=42`.
- **Prevención de fuga de datos (*data leakage*):**
  - `TfidfVectorizer(ngram_range=(1, 2))` se ajusta **únicamente** con `X_train` (`fit_transform`).
  - `X_test` solo se transforma (`transform`).
  - En validación cruzada (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), cada pliegue encapsula su propio `TfidfVectorizer` dentro de un `Pipeline` ajustado exclusivamente en `X_train`.
- **Modelos comparados sobre la misma partición:**
  1. `LogisticRegression(max_iter=1000, random_state=42)`
  2. `CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)`
  3. `MultinomialNB()`
- **Criterio de selección:** Mayor `F1-score macro` en el conjunto de prueba (20%), respaldado por validación cruzada estratificada (5 folds), matriz de confusión legible, análisis automático de errores y justificación escrita en el notebook.
- **Calibración de fallback:** Experimento documentado en el notebook comparando distribuciones de confianza máxima entre preguntas válidas de prueba y consultas fuera de dominio (OOD).
- **Artefactos exportados por separado (con `joblib.dump` y versionados en Git):**
  - `models/vectorizador.pkl`: Instancia entrenada de `TfidfVectorizer`.
  - `models/modelo.pkl`: Clasificador ganador con soporte para `predict_proba` y atributo `classes_`.
  - Salidas y gráficas guardadas dentro de `notebooks/entrenamiento_evaluacion.ipynb`, además de `reports/metrics.json` y matrices de confusión PNG en `reports/`.

---

## 5. Contrato de Inferencia e Interfaz (`app.py`)

- **Entradas requeridas al iniciar:**
  - `models/modelo.pkl`
  - `models/vectorizador.pkl`
  - `data/corpus_upq.csv`
- **Responsabilidad estricta:** `app.py` no entrena modelos ni consulta servicios externos; únicamente carga artefactos, preprocesa, predice, aplica el fallback y presenta la respuesta.
- **Flujo de inferencia por consulta:**
  1. Validar que la consulta limpia no esté vacía ni tenga longitud menor a 2 caracteres.
  2. Limpiar la pregunta del usuario con `limpiar_texto(consulta)`.
  3. Transformar el texto limpio con `vectorizador.transform([texto_limpio])`.
  4. Calcular probabilidades por clase con `modelo.predict_proba(X_vectorizado)[0]`.
  5. Obtener la intención con mayor probabilidad (`confianza = max(probabilidades)`) y el Top-3 de intenciones candidatas.
  6. Si `confianza < umbral_confianza` (valor por defecto calibrado en el notebook), devolver el mensaje de respaldo definido exclusivamente en `app.py` listando brevemente los temas soportados; en caso contrario, devolver `respuestas[intencion]`.

---

## 6. Flujo de Trabajo en Git y Reglas de Integración

- **Protección de `main`:** Prohibido hacer `push` directo a la rama `main`. Todo cambio debe integrarse mediante Pull Request revisado por al menos otro integrante del equipo.
- **Ramas oficiales de trabajo:**
  - `feat/datos`: Expansión, curaduría y validación de `data/corpus_upq.csv` y `src/validate_corpus.py`.
  - `feat/modelo`: Mejoras en `src/preprocess.py`, `notebooks/entrenamiento_evaluacion.ipynb`, artefactos en `models/` y reportes en `reports/`.
  - `feat/interfaz`: Desarrollo y calibración en `app.py`, `requirements.txt`, `README.md` y `docs/`.
- **Regla obligatoria antes de abrir un Pull Request:**
  1. Ejecutar `python src/validate_corpus.py` y confirmar código de salida `0`.
  2. Re-ejecutar todas las celdas de `notebooks/entrenamiento_evaluacion.ipynb` (guardando salidas y gráficas) siempre que cambien `data/corpus_upq.csv` o `src/preprocess.py`.
  3. Regenerar y confirmar que `models/modelo.pkl` y `models/vectorizador.pkl` estén actualizados y rastreados por Git (`!models/*.pkl` en `.gitignore`).
  4. Comprobar sintaxis (`python -m py_compile src/preprocess.py src/validate_corpus.py app.py`) y verificar que `streamlit run app.py` arranque sin errores.
