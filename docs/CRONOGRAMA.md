# Cronograma de Trabajo y Acuerdos del Equipo (`chatbot-upq`)

Este documento establece la planificación, fases, entregables y acuerdos operativos del equipo para el desarrollo y entrega del sistema **Chatbot FAQ - UPQ**.

---

## 1. Distribución de Roles y Responsabilidades

| Rol | Responsabilidad Principal | Entregables Clave |
| :--- | :--- | :--- |
| **Ingeniería de Datos y Corpus** | Recolección, curaduría, etiquetado y validación de preguntas y respuestas oficiales de la UPQ. | `data/corpus_upq.csv` (`texto,intencion,respuesta_oficial`) |
| **Procesamiento de Lenguaje Natural (NLP)** | Diseño e implementación de la función de limpieza y normalización de texto en español. | `src/preprocess.py` (`limpiar_texto`) |
| **Machine Learning y Evaluación** | Experimentación supervisada, ajuste TF-IDF en train, comparativa de modelos y exportación de artefactos. | `notebooks/entrenamiento_evaluacion.ipynb`, `models/vectorizador.pkl`, `models/modelo.pkl`, `reports/` |
| **Desarrollo de Interfaz e Integración** | Construcción de la interfaz conversacional en Streamlit, control de umbral de confianza y documentación. | `app.py`, `README.md`, `docs/CONTRATOS.md`, `docs/CRONOGRAMA.md` |

---

## 2. Cronograma de Actividades por Fase

| Fase | Actividad | Estado | Entregable Verificable |
| :---: | :--- | :---: | :--- |
| **Fase 1** | Definición de intenciones (`admisiones`, `becas`, `costos`, `servicios_escolares`) y acuerdos técnicos. | Completado | `docs/CONTRATOS.md`, `docs/CRONOGRAMA.md` |
| **Fase 2** | Construcción y consolidación del corpus semilla unificado con respuestas oficiales en 3 columnas. | Completado | `data/corpus_upq.csv` |
| **Fase 3** | Implementación del módulo reutilizable de preprocesamiento y limpieza de texto. | Completado | `src/preprocess.py` |
| **Fase 4** | Desarrollo del Jupyter Notebook con split 80/20 estratificado, TF-IDF sin fuga de datos y comparativa de 3 modelos. | Completado | `notebooks/entrenamiento_evaluacion.ipynb` |
| **Fase 5** | Serialización por separado del vectorizador y modelo ganador (`joblib.dump`) y verificación en Git. | Completado | `models/vectorizador.pkl`, `models/modelo.pkl`, `.gitignore` |
| **Fase 6** | Integración de artefactos `.pkl` y lectura dinámica de respuestas desde `data/corpus_upq.csv` en Streamlit. | Completado | `app.py`, `requirements.txt`, `README.md` |

---

## 3. Acuerdos de Calidad y Entrega del Equipo

1. **Fuente única de datos:** No se utilizan archivos JSON externos para las respuestas; toda relación `intencion -> respuesta_oficial` proviene exclusivamente de `data/corpus_upq.csv`.
2. **Reproducibilidad:** Todas las particiones y modelos utilizan `random_state=42` para garantizar resultados idénticos al re-ejecutar `notebooks/entrenamiento_evaluacion.ipynb`.
3. **Versionado de artefactos:** El archivo `.gitignore` permite explícitamente versionar `models/*.pkl` (`models/vectorizador.pkl` y `models/modelo.pkl`) para que la aplicación `app.py` funcione de inmediato al clonar el repositorio.
4. **Validación previa a entrega:** Todo cambio en `src/preprocess.py` o `data/corpus_upq.csv` requiere re-ejecutar el notebook completo y verificar que `app.py` cargue sin errores.

