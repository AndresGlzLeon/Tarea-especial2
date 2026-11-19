"""Entrenamiento, evaluación y selección del modelo FAQ de la UPQ."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocess import limpiar_texto

DATA_PATH = BASE_DIR / "data" / "corpus_upq.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_OUTPUT_PATH = MODELS_DIR / "modelo_upq.joblib"
METRICS_OUTPUT_PATH = REPORTS_DIR / "metrics.json"


def construir_modelos() -> dict[str, Pipeline]:
    """Construye los 3 pipelines supervisados con TF-IDF (unigramas y bigramas)."""
    clasificadores = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "linear_svc_calibrated": CalibratedClassifierCV(
            LinearSVC(random_state=42), cv=3
        ),
        "multinomial_nb": MultinomialNB(),
    }
    return {
        nombre: Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("clf", clasificador),
            ]
        )
        for nombre, clasificador in clasificadores.items()
    }


def guardar_matriz_confusion(
    y_true: pd.Series,
    y_pred,
    etiquetas: list[str],
    nombre_modelo: str,
    ruta_salida: Path,
) -> None:
    """Genera y guarda la matriz de confusión como imagen PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusión - {nombre_modelo}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el corpus en: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    if not {"pregunta", "intencion"}.issubset(df.columns):
        raise ValueError("El CSV debe contener las columnas 'pregunta' e 'intencion'.")

    df = df.dropna(subset=["pregunta", "intencion"]).copy()
    X = df["pregunta"].astype(str).apply(limpiar_texto)
    y = df["intencion"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    etiquetas = sorted(y.unique().tolist())
    modelos = construir_modelos()

    resultados: dict[str, dict[str, object]] = {}
    pipelines_entrenados: dict[str, Pipeline] = {}

    for nombre, pipeline in modelos.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision_macro = float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        recall_macro = float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        print("=" * 70)
        print(f"Modelo: {nombre}")
        print("-" * 70)
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"F1-score macro: {f1_macro:.4f}")

        ruta_cm = REPORTS_DIR / f"confusion_matrix_{nombre}.png"
        guardar_matriz_confusion(y_test, y_pred, etiquetas, nombre, ruta_cm)

        resultados[nombre] = {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "confusion_matrix_path": str(ruta_cm.relative_to(BASE_DIR)),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }
        pipelines_entrenados[nombre] = pipeline

    tabla_comparativa = pd.DataFrame(
        [
            {
                "modelo": nombre,
                "accuracy": datos["accuracy"],
                "precision_macro": datos["precision_macro"],
                "recall_macro": datos["recall_macro"],
                "f1_macro": datos["f1_macro"],
            }
            for nombre, datos in resultados.items()
        ]
    ).sort_values(by="f1_macro", ascending=False)

    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA FINAL")
    print("=" * 70)
    print(tabla_comparativa.to_string(index=False))

    modelo_ganador = str(tabla_comparativa.iloc[0]["modelo"])
    mejor_f1 = float(tabla_comparativa.iloc[0]["f1_macro"])
    print(f"\nModelo ganador: {modelo_ganador} (F1-score macro = {mejor_f1:.4f})")

    joblib.dump(pipelines_entrenados[modelo_ganador], MODEL_OUTPUT_PATH)
    print(f"Pipeline ganador guardado en: {MODEL_OUTPUT_PATH}")

    resumen_metricas = {
        "winner": modelo_ganador,
        "winner_f1_macro": mejor_f1,
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "classes": etiquetas,
        "models": resultados,
    }
    with METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as archivo:
        json.dump(resumen_metricas, archivo, ensure_ascii=False, indent=2)
    print(f"Métricas guardadas en: {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

