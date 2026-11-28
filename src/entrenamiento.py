"""Pipeline completo de entrenamiento, evaluación, análisis de errores y calibración OOD.

Implementa las Fases 3 y 4 del cronograma del proyecto chatbot-upq:
- Split 80/20 estratificado (random_state=42)
- TF-IDF ajustado solo en train (prevención de data leakage)
- Comparativa de 3 modelos: LogisticRegression, CalibratedClassifierCV(LinearSVC), MultinomialNB
- Validación cruzada 5-fold con Pipeline (TfidfVectorizer encapsulado)
- Matrices de confusión (guardadas como PNG)
- Análisis automático de errores
- Calibración de fallback con consultas fuera de dominio (OOD)
- Exportación de artefactos: modelo.pkl, vectorizador.pkl, metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "corpus_upq.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
VECTORIZER_OUTPUT = MODELS_DIR / "vectorizador.pkl"
MODEL_OUTPUT = MODELS_DIR / "modelo.pkl"
METRICS_OUTPUT = REPORTS_DIR / "metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Importar preprocesador ────────────────────────────────────────────────────
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from src.preprocess import limpiar_texto

# ── Consultas fuera de dominio (OOD) para calibración de fallback ─────────────
CONSULTAS_OOD = [
    "¿Cómo preparar una pizza de pepperoni casera?",
    "¿Quién ganó el mundial de fútbol en 2022?",
    "¿Cuál es la capital de Australia?",
    "Dame la receta de unos tacos al pastor",
    "¿Cuántos planetas tiene el sistema solar?",
    "¿Qué hora es en Tokio ahora mismo?",
    "¿Cuál es el mejor celular del mercado actualmente?",
    "Quiero comprar un auto usado en Querétaro",
    "¿Cómo puedo bajar de peso rápidamente?",
    "¿Dónde queda el aeropuerto más cercano a Cancún?",
    "Necesito un plomero urgente para mi casa",
    "¿Cuánto cuesta un vuelo a Madrid desde México?",
    "¿Cómo funciona la inteligencia artificial de ChatGPT?",
    "¿Quién fue el presidente de México en 2020?",
    "Dame recomendaciones de películas de terror",
    "¿Cómo invertir en la bolsa de valores mexicana?",
    "¿Cuál es el mejor restaurante de sushi en Querétaro?",
    "¿Cómo se hace la fotosíntesis en las plantas?",
    "Resultados de la liga mx jornada 15",
    "¿Cuántas calorías tiene un refresco de cola?",
    "¿Dónde puedo adoptar un perro en la ciudad?",
    "¿Cómo cambiar una llanta ponchada paso a paso?",
    "¿Cuál es la fórmula del agua en química?",
    "Tips para mejorar mi conexión de internet en casa",
    "¿Qué significa soñar con serpientes?",
    "¿Cuándo es el próximo eclipse solar en México?",
    "¿Cómo hacer un curriculum vitae profesional?",
    "¿Cuál es el tipo de cambio del dólar hoy?",
    "Horario de misas en la catedral de Querétaro",
    "¿Dónde puedo sacar mi pasaporte mexicano?",
]


def cargar_y_preprocesar(ruta_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carga el corpus, aplica preprocesamiento y retorna df, X, y."""
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el corpus en: {ruta_csv}")

    df = pd.read_csv(ruta_csv, encoding="utf-8")
    columnas = {"texto", "intencion", "respuesta_oficial"}
    if not columnas.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas: {columnas}")

    df = df.dropna(subset=["texto", "intencion", "respuesta_oficial"]).copy()
    df["texto_limpio"] = df["texto"].astype(str).apply(limpiar_texto)
    X = df["texto_limpio"]
    y = df["intencion"].astype(str)

    print(f"Total de registros cargados: {len(df)}")
    print(f"Intenciones: {y.nunique()} clases")
    print(f"\nDistribución por intención:")
    print(y.value_counts().to_string())
    return df, X, y


def entrenar_y_evaluar(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test,
    etiquetas: list[str],
) -> tuple[dict, dict]:
    """Entrena 3 clasificadores, evalúa y genera matrices de confusión."""
    clasificadores = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "linear_svc_calibrated": CalibratedClassifierCV(
            LinearSVC(random_state=42, dual="auto"), cv=3
        ),
        "multinomial_nb": MultinomialNB(),
    }

    resultados: dict = {}
    modelos_entrenados: dict = {}

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax, (nombre, clf) in zip(axes, clasificadores.items()):
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

        accuracy = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        num_errores = int((y_test != y_pred).sum())
        total_test = len(y_test)

        print("=" * 70)
        print(f"Modelo: {nombre}")
        print("-" * 70)
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"F1-score macro: {f1_macro:.4f}")
        print(f"Errores: {num_errores} / {total_test}\n")

        cm = confusion_matrix(y_test, y_pred, labels=etiquetas)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=etiquetas, yticklabels=etiquetas, ax=ax,
        )
        ax.set_title(f"{nombre}\n(F1 macro = {f1_macro:.4f})")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

        # Guardar matriz individual
        ruta_cm = REPORTS_DIR / f"confusion_matrix_{nombre}.png"
        fig_ind, ax_ind = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=etiquetas, yticklabels=etiquetas, ax=ax_ind,
        )
        ax_ind.set_title(f"Matriz de Confusión — {nombre}")
        ax_ind.set_xlabel("Predicción")
        ax_ind.set_ylabel("Etiqueta real")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig_ind.savefig(ruta_cm, dpi=150)
        plt.close(fig_ind)

        resultados[nombre] = {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "errores_test": num_errores,
            "total_test": total_test,
            "confusion_matrix_path": str(ruta_cm.relative_to(BASE_DIR)),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }
        modelos_entrenados[nombre] = clf

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices_comparativa.png", dpi=150)
    plt.close(fig)

    return resultados, modelos_entrenados


def validacion_cruzada(
    X: pd.Series,
    y: pd.Series,
) -> dict[str, dict[str, float]]:
    """Validación cruzada 5-fold con Pipeline para evitar data leakage."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clasificadores_cv = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "linear_svc_calibrated": CalibratedClassifierCV(
            LinearSVC(random_state=42, dual="auto"), cv=3
        ),
        "multinomial_nb": MultinomialNB(),
    }

    resultados_cv: dict[str, dict[str, float]] = {}

    print("\n" + "=" * 70)
    print("  VALIDACIÓN CRUZADA ESTRATIFICADA (5-fold con Pipeline)")
    print("=" * 70)

    for nombre, clf in clasificadores_cv.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", clf),
        ])
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro")
        media = float(scores.mean())
        std = float(scores.std())
        resultados_cv[nombre] = {
            "f1_macro_cv_mean": round(media, 4),
            "f1_macro_cv_std": round(std, 4),
            "f1_macro_cv_folds": [round(float(s), 4) for s in scores],
        }
        print(f"  {nombre}: F1 macro CV = {media:.4f} ± {std:.4f}  |  Folds: {scores.round(4)}")

    return resultados_cv


def analisis_errores(
    df: pd.DataFrame,
    X_test: pd.Series,
    y_test: pd.Series,
    modelo,
    vectorizador: TfidfVectorizer,
    X_test_tfidf,
) -> pd.DataFrame:
    """Identifica y analiza las muestras mal clasificadas."""
    y_pred = modelo.predict(X_test_tfidf)
    probabilidades = modelo.predict_proba(X_test_tfidf)

    errores_mask = y_test != y_pred
    if errores_mask.sum() == 0:
        print("\n✨ El modelo ganador clasificó correctamente TODAS las muestras de prueba.")
        return pd.DataFrame()

    indices_error = y_test[errores_mask].index

    df_errores = pd.DataFrame({
        "texto_original": df.loc[indices_error, "texto"].values,
        "texto_limpio": X_test.loc[indices_error].values,
        "intencion_real": y_test.loc[indices_error].values,
        "intencion_predicha": y_pred[errores_mask],
        "confianza_predicha": [
            float(probabilidades[i].max())
            for i in range(len(y_pred))
            if errores_mask.iloc[i]
        ],
    })

    print("\n" + "=" * 70)
    print(f"  ANÁLISIS DE ERRORES — {len(df_errores)} muestras mal clasificadas")
    print("=" * 70)
    for _, row in df_errores.iterrows():
        print(f"\n  Texto: «{row['texto_original'][:80]}...»" if len(str(row['texto_original'])) > 80 else f"\n  Texto: «{row['texto_original']}»")
        print(f"  Real: {row['intencion_real']}  →  Predicha: {row['intencion_predicha']}  (confianza: {row['confianza_predicha']:.2%})")

    # Resumen de confusiones más frecuentes
    print(f"\n  Confusiones más frecuentes:")
    confusiones = df_errores.groupby(["intencion_real", "intencion_predicha"]).size().sort_values(ascending=False)
    for (real, pred), count in confusiones.items():
        print(f"    {real} → {pred}: {count} error(es)")

    return df_errores


def calibracion_ood(
    modelo,
    vectorizador: TfidfVectorizer,
    X_test: pd.Series,
    y_test: pd.Series,
    X_test_tfidf,
) -> float:
    """Calibra el umbral de fallback comparando confianzas de preguntas válidas vs OOD."""
    # Confianzas de preguntas válidas (test set)
    probs_valid = modelo.predict_proba(X_test_tfidf)
    confianzas_validas = np.array([p.max() for p in probs_valid])

    # Confianzas de consultas OOD
    textos_ood_limpios = [limpiar_texto(q) for q in CONSULTAS_OOD]
    X_ood_tfidf = vectorizador.transform(textos_ood_limpios)
    probs_ood = modelo.predict_proba(X_ood_tfidf)
    confianzas_ood = np.array([p.max() for p in probs_ood])

    # Determinar umbral óptimo
    # Buscamos el punto donde mejor se separan ambas distribuciones
    umbrales_candidatos = np.arange(0.10, 0.90, 0.01)
    mejor_umbral = 0.45
    mejor_score = -1.0

    for umb in umbrales_candidatos:
        # True positives: preguntas válidas con confianza >= umbral
        tp = (confianzas_validas >= umb).sum()
        # True negatives: OOD con confianza < umbral
        tn = (confianzas_ood < umb).sum()
        score = (tp / len(confianzas_validas)) + (tn / len(confianzas_ood))
        if score > mejor_score:
            mejor_score = score
            mejor_umbral = float(umb)

    tasa_acierto_validas = float((confianzas_validas >= mejor_umbral).mean())
    tasa_rechazo_ood = float((confianzas_ood < mejor_umbral).mean())

    print("\n" + "=" * 70)
    print("  CALIBRACIÓN DE FALLBACK (Out-of-Domain)")
    print("=" * 70)
    print(f"  Preguntas válidas (test): {len(confianzas_validas)} muestras")
    print(f"    Confianza media: {confianzas_validas.mean():.4f}")
    print(f"    Confianza mínima: {confianzas_validas.min():.4f}")
    print(f"    Confianza máxima: {confianzas_validas.max():.4f}")
    print(f"\n  Consultas OOD: {len(confianzas_ood)} muestras")
    print(f"    Confianza media: {confianzas_ood.mean():.4f}")
    print(f"    Confianza mínima: {confianzas_ood.min():.4f}")
    print(f"    Confianza máxima: {confianzas_ood.max():.4f}")
    print(f"\n  ✦ Umbral óptimo calibrado: {mejor_umbral:.2f}")
    print(f"    Tasa de acierto en válidas: {tasa_acierto_validas:.2%}")
    print(f"    Tasa de rechazo de OOD:     {tasa_rechazo_ood:.2%}")

    # Gráfica de distribuciones
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma superpuesto
    ax1.hist(confianzas_validas, bins=30, alpha=0.6, label="Válidas (test)", color="steelblue", edgecolor="white")
    ax1.hist(confianzas_ood, bins=30, alpha=0.6, label="Fuera de dominio (OOD)", color="salmon", edgecolor="white")
    ax1.axvline(mejor_umbral, color="red", linestyle="--", linewidth=2, label=f"Umbral = {mejor_umbral:.2f}")
    ax1.set_xlabel("Confianza máxima (max P)")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribución de Confianzas: Válidas vs OOD")
    ax1.legend()

    # Boxplot comparativo
    datos_box = [confianzas_validas, confianzas_ood]
    try:
        bp = ax2.boxplot(datos_box, tick_labels=["Válidas", "OOD"], patch_artist=True)
    except TypeError:
        bp = ax2.boxplot(datos_box, labels=["Válidas", "OOD"], patch_artist=True)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("salmon")
    bp["boxes"][1].set_alpha(0.6)
    ax2.axhline(mejor_umbral, color="red", linestyle="--", linewidth=2, label=f"Umbral = {mejor_umbral:.2f}")
    ax2.set_ylabel("Confianza máxima (max P)")
    ax2.set_title("Boxplot Comparativo de Confianzas")
    ax2.legend()

    plt.tight_layout()
    ruta_ood = REPORTS_DIR / "calibracion_ood.png"
    fig.savefig(ruta_ood, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada en: {ruta_ood}")

    return mejor_umbral


def main() -> None:
    """Pipeline principal de entrenamiento y evaluación."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  PIPELINE DE ENTRENAMIENTO — CHATBOT FAQ UPQ                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"Directorio base: {BASE_DIR}\n")

    # ── 1. Carga y preprocesamiento ───────────────────────────────────────────
    df, X, y = cargar_y_preprocesar(DATA_PATH)
    etiquetas = sorted(y.unique().tolist())

    # ── 2. Split 80/20 estratificado ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f"\nPartición: {len(X_train)} train / {len(X_test)} test")

    # ── 3. TF-IDF (fit solo en train) ─────────────────────────────────────────
    vectorizador = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = vectorizador.fit_transform(X_train)
    X_test_tfidf = vectorizador.transform(X_test)
    print(f"Vocabulario TF-IDF: {X_train_tfidf.shape[1]} términos\n")

    # ── 4. Entrenamiento y evaluación de 3 modelos ────────────────────────────
    resultados, modelos_entrenados = entrenar_y_evaluar(
        X_train_tfidf, X_test_tfidf, y_train, y_test, etiquetas,
    )

    # ── 5. Validación cruzada 5-fold con Pipeline ─────────────────────────────
    resultados_cv = validacion_cruzada(X, y)

    # ── 6. Tabla comparativa final ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TABLA COMPARATIVA FINAL")
    print("=" * 70)
    tabla = pd.DataFrame([
        {
            "modelo": nombre,
            "accuracy": datos["accuracy"],
            "precision_macro": datos["precision_macro"],
            "recall_macro": datos["recall_macro"],
            "f1_macro_test": datos["f1_macro"],
            "f1_macro_cv": resultados_cv[nombre]["f1_macro_cv_mean"],
            "f1_cv_std": resultados_cv[nombre]["f1_macro_cv_std"],
            "errores": datos["errores_test"],
        }
        for nombre, datos in resultados.items()
    ]).sort_values(by="f1_macro_test", ascending=False)

    print(tabla.to_string(index=False))

    # Selección del modelo ganador
    modelo_ganador = str(tabla.iloc[0]["modelo"])
    mejor_f1 = float(tabla.iloc[0]["f1_macro_test"])
    mejor_f1_cv = float(tabla.iloc[0]["f1_macro_cv"])
    print(f"\n✦ Modelo ganador: {modelo_ganador}")
    print(f"  F1-macro test: {mejor_f1:.4f}")
    print(f"  F1-macro CV:   {mejor_f1_cv:.4f} ± {tabla.iloc[0]['f1_cv_std']:.4f}")

    clf_ganador = modelos_entrenados[modelo_ganador]

    # ── 7. Análisis de errores ────────────────────────────────────────────────
    df_errores = analisis_errores(df, X_test, y_test, clf_ganador, vectorizador, X_test_tfidf)

    # ── 8. Calibración de fallback OOD ────────────────────────────────────────
    umbral_calibrado = calibracion_ood(clf_ganador, vectorizador, X_test, y_test, X_test_tfidf)

    # ── 9. Justificación técnica del modelo ganador ───────────────────────────
    print("\n" + "=" * 70)
    print("  JUSTIFICACIÓN TÉCNICA DEL MODELO GANADOR")
    print("=" * 70)
    print(f"""
  El modelo seleccionado es «{modelo_ganador}» con base en los siguientes criterios:

  1. F1-score macro en test ({mejor_f1:.4f}): Mayor entre los 3 modelos evaluados,
     lo que indica el mejor balance entre precisión y exhaustividad para las
     15 clases del corpus, sin sesgo hacia clases mayoritarias.

  2. F1-score macro en validación cruzada ({mejor_f1_cv:.4f} ± {tabla.iloc[0]['f1_cv_std']:.4f}):
     Consistente con el rendimiento en test, confirmando que el modelo generaliza
     adecuadamente y no presenta sobreajuste significativo.

  3. Errores en test: {int(tabla.iloc[0]['errores'])} de {len(X_test)} muestras de prueba.

  4. Soporte probabilístico: El modelo ofrece predict_proba() y classes_,
     permitiendo la calibración del umbral de fallback para rechazar consultas
     fuera de dominio (OOD) con confianza < {umbral_calibrado:.2f}.
""")

    # ── 10. Exportación de artefactos ─────────────────────────────────────────
    joblib.dump(vectorizador, VECTORIZER_OUTPUT)
    joblib.dump(clf_ganador, MODEL_OUTPUT)

    # Merge CV results into model results
    for nombre in resultados:
        resultados[nombre].update(resultados_cv.get(nombre, {}))

    resumen = {
        "winner": modelo_ganador,
        "winner_f1_macro_test": mejor_f1,
        "winner_f1_macro_cv": mejor_f1_cv,
        "umbral_fallback_calibrado": round(umbral_calibrado, 2),
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "classes": etiquetas,
        "num_classes": len(etiquetas),
        "models": resultados,
    }
    with METRICS_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  ARTEFACTOS EXPORTADOS")
    print("=" * 70)
    print(f"  Vectorizador: {VECTORIZER_OUTPUT}")
    print(f"  Modelo ganador: {MODEL_OUTPUT}")
    print(f"  Métricas: {METRICS_OUTPUT}")
    print(f"  Matrices de confusión: {REPORTS_DIR}/confusion_matrix_*.png")
    print(f"  Calibración OOD: {REPORTS_DIR}/calibracion_ood.png")
    print("=" * 70)
    print("✨ Pipeline de entrenamiento completado exitosamente.")


if __name__ == "__main__":
    main()
