# Cronograma de Trabajo y Acuerdos del Equipo (`chatbot-upq`)

Planificación oficial del equipo y seguimiento del avance para la **Tarea Especial 2/2: Modelo matemático de ML con aprendizaje supervisado para el manejo y recuperación de información oficial de la UPQ**.

- **Fecha límite de entrega:** Martes 1 de diciembre, 11:00 a.m.
- **Revisión presencial:** Martes 1 de diciembre y jueves 3 de diciembre.

---

## 1. Integrantes y Distribución de Roles

| Rol | Integrante(s) | Responsabilidad Principal | Entregables Clave |
| :--- | :--- | :--- | :--- |
| **Ingeniería de Datos y Corpus** | `[Nombre]` | Verificación institucional en `www.upq.mx`, curaduría de las 15 intenciones oficiales (25 variantes por clase) y validación automática del CSV. | `data/corpus_upq.csv`, `src/validate_corpus.py` |
| **Preprocesamiento y Modelado (ML)** | `[Nombre]` | Mantenimiento de `limpiar_texto`, ejecución del notebook (split 80/20, CV 5-fold, 3 modelos, análisis de errores, calibración de fallback) y exportación de `.pkl`. | `src/preprocess.py`, `notebooks/entrenamiento_evaluacion.ipynb`, `models/*.pkl`, `reports/*` |
| **Interfaz, Integración y Repo** | `[Nombre]` | Desarrollo y pruebas de `app.py` con fallback calibrado, gestión de ramas/PRs en Git, verificación de instalación limpia y documentación final. | `app.py`, `requirements.txt`, `README.md`, `docs/*` |

---

## 2. Cronograma de Actividades por Fase (23 nov – 1 dic)

| Fase | Fechas | Actividad | Responsable | Estado | Entregable Verificable |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Fase 1** | 23 nov | Cierre de contratos técnicos, definición de las 15 intenciones oficiales de la UPQ y creación del script validador de calidad de datos. | `[Nombre]` | `Completado` | `docs/CONTRATOS.md`, `src/validate_corpus.py` |
| **Fase 2** | 23–25 nov | Expansión del corpus a 15 intenciones × 54 variantes (810 filas) y verificación manual de cada respuesta oficial contra `www.upq.mx`. | `[Nombre]` | `Completado` | `data/corpus_upq.csv` validado (`exit code 0`) |
| **Fase 3** | 26 nov | Ejecución completa del Jupyter Notebook (split 80/20 estratificado, TF-IDF solo en train, CV 5-fold y comparativa de 3 modelos). | `[Nombre]` | `Completado` | `notebooks/entrenamiento_evaluacion.ipynb` con salidas visibles |
| **Fase 4** | 27 nov | Revisión del análisis de errores, justificación técnica del modelo ganador, calibración de fallback (OOD) y exportación de `.pkl`. | `[Nombre]` | `Completado` | `models/modelo.pkl`, `models/vectorizador.pkl`, `reports/*` |
| **Fase 5** | 28 nov | Integración final en Streamlit (`app.py`), validación del umbral calibrado, botones de prueba rápida y Top-3 en modo depuración. | `[Nombre]` | `Completado` | `app.py` probado con consultas válidas y OOD |
| **Fase 6** | 29 nov | Asignación de nombres completos del equipo, pulido de `README.md`, revisión de PRs hacia `main` y prueba de instalación limpia desde cero. | `[Nombre]` | `Pendiente` | `README.md`, `requirements.txt` y entorno limpio verificados |
| **Fase 7** | 30 nov | Ensayo de defensa presencial (TF-IDF, prevención de data leakage, CV, métricas, matriz de confusión y fallback) y permisos del repositorio. | `[Nombre]` | `Pendiente` | Repositorio auditado y listo en GitHub/GitLab |
| **Entrega** | 1 dic (< 11:00 a.m.) | Envío de la liga del repositorio en Classroom incluyendo los nombres completos de todos los integrantes. | `[Nombre]` | `Pendiente` | Entrega confirmada en Classroom |

---

## 3. Acuerdos de Calidad y Git

1. **Fuente única de datos:** Toda relación `intencion -> respuesta_oficial` vive exclusivamente en `data/corpus_upq.csv` con exactamente 3 columnas (`texto,intencion,respuesta_oficial`).
2. **Flujo de ramas obligatorio:** Prohibido hacer `push` directo a `main`. Trabajar en `feat/datos`, `feat/modelo` y `feat/interfaz`, e integrar mediante Pull Request revisado.
3. **Versionado de artefactos:** Los archivos `models/modelo.pkl` y `models/vectorizador.pkl` se mantienen versionados en Git mediante `!models/*.pkl` en `.gitignore`.
4. **Regla de validación continua:** Cualquier modificación en `data/corpus_upq.csv` o `src/preprocess.py` exige correr `python src/validate_corpus.py`, re-ejecutar el notebook completo con salidas guardadas y comprobar `streamlit run app.py`.

---

## 4. Checklist de Progreso del Proyecto

| # | Bloque / Área | Tarea / Entregable | Archivo / Ubicación | Responsable | Estado |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | **A. Contratos y Datos** | Actualizar `docs/CONTRATOS.md` y `docs/CRONOGRAMA.md` con reglas de volumen y flujo Git | `docs/` | `[Nombre]` | ✅ `Completado` |
| 2 | **A. Contratos y Datos** | Implementar validador automático de calidad de datos con exit code `0`/`1` | `src/validate_corpus.py` | `[Nombre]` | ✅ `Completado` |
| 3 | **A. Contratos y Datos** | Escalar corpus a 15 intenciones oficiales × 54 variantes (810 filas) sin nulos ni duplicados | `data/corpus_upq.csv` | `[Nombre]` | ✅ `Completado` |
| 4 | **A. Contratos y Datos** | Verificar manualmente cada respuesta oficial contra el portal vigente `www.upq.mx` | `data/corpus_upq.csv` | `[Nombre]` | ✅ `Completado` |
| 5 | **B. Modelado ML** | Auditar `limpiar_texto` con docstrings, type hints y manejo seguro de entradas | `src/preprocess.py` | `[Nombre]` | ✅ `Completado` |
| 6 | **B. Modelado ML** | Ejecutar notebook completo (80/20, TF-IDF en train, 3 modelos, CV 5-fold, matrices, errores, OOD) | `notebooks/entrenamiento_evaluacion.ipynb` | `[Nombre]` | ✅ `Completado` |
| 7 | **B. Modelado ML** | Exportar `models/vectorizador.pkl` y `models/modelo.pkl` y verificar rastreo en Git | `models/*.pkl`, `.gitignore` | `[Nombre]` | ✅ `Completado` |
| 8 | **C. Interfaz Streamlit** | Calibrar umbral de fallback, botones de prueba rápida y Top-3 en modo depuración | `app.py` | `[Nombre]` | ✅ `Completado` |
| 9 | **C. Interfaz Streamlit** | Probar consultas válidas, cortas/vacías, con errores ortográficos y fuera de dominio | `app.py` | `[Nombre]` | ✅ `Completado` |
| 10 | **D. Cierre y Entrega** | Colocar nombres completos de los integrantes en `README.md` y `docs/CRONOGRAMA.md` | `README.md`, `docs/` | `[Nombre]` | ⏳ `Pendiente` |
| 11 | **D. Cierre y Entrega** | Verificar instalación limpia en nuevo `.venv` (`pip install -r requirements.txt`) | `requirements.txt`, `README.md` | `[Nombre]` | ⏳ `Pendiente` |
| 12 | **D. Cierre y Entrega** | Integrar PRs a `main`, ensayar defensa presencial y subir liga a Classroom (< 1 dic, 11:00 a.m.) | GitHub / Classroom | `[Nombre]` | ⏳ `Pendiente` |
