"""Interfaz conversacional en Streamlit para el chatbot FAQ de la UPQ."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import limpiar_texto

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizador.pkl"
CORPUS_PATH = BASE_DIR / "data" / "corpus_upq.csv"

FALLBACK_MESSAGE = (
    "Lo siento, no estoy seguro de haber entendido tu consulta. "
    "Por favor intenta reformular tu pregunta sobre admisiones, becas, costos "
    "o servicios escolares de la UPQ, o comunícate directamente con los canales "
    "oficiales de la universidad."
)

st.set_page_config(
    page_title="Chatbot FAQ - UPQ",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def cargar_artefactos(ruta_modelo: Path, ruta_vectorizador: Path):
    """Carga el clasificador entrenado y el vectorizador TF-IDF."""
    modelo = joblib.load(ruta_modelo)
    vectorizador = joblib.load(ruta_vectorizador)
    return modelo, vectorizador


@st.cache_data
def cargar_respuestas(ruta_csv: Path) -> dict[str, str]:
    """Construye el diccionario {intencion: respuesta_oficial} leyendo data/corpus_upq.csv."""
    df = pd.read_csv(ruta_csv, encoding="utf-8")
    columnas_requeridas = {"intencion", "respuesta_oficial"}
    if not columnas_requeridas.issubset(df.columns):
        raise ValueError(
            "El archivo data/corpus_upq.csv debe contener las columnas 'intencion' y 'respuesta_oficial'."
        )
    df_unico = df.dropna(subset=["intencion", "respuesta_oficial"]).drop_duplicates(
        subset=["intencion"]
    )
    respuestas = dict(
        zip(
            df_unico["intencion"].astype(str),
            df_unico["respuesta_oficial"].astype(str),
        )
    )
    respuestas["_fallback"] = FALLBACK_MESSAGE
    return respuestas


def responder_consulta(
    consulta: str,
    modelo,
    vectorizador,
    respuestas: dict[str, str],
    umbral: float,
) -> tuple[str, str, float, bool]:
    """Clasifica la consulta y devuelve respuesta, intención, confianza y bandera de fallback."""
    texto_limpio = limpiar_texto(consulta)
    fallback = respuestas.get("_fallback", FALLBACK_MESSAGE)

    if not texto_limpio:
        return fallback, "ninguna", 0.0, True

    X_vectorizado = vectorizador.transform([texto_limpio])
    probabilidades = modelo.predict_proba(X_vectorizado)[0]
    indice_max = int(np.argmax(probabilidades))
    confianza = float(probabilidades[indice_max])
    intencion = str(modelo.classes_[indice_max])

    if confianza < umbral:
        return fallback, intencion, confianza, True

    respuesta = respuestas.get(intencion, fallback)
    return respuesta, intencion, confianza, False


def main() -> None:
    st.title("🎓 Chatbot FAQ - UPQ")
    st.caption(
        "Asistente local de preguntas frecuentes de la Universidad Politécnica de Querétaro."
    )

    with st.sidebar:
        st.header("Configuración")
        umbral_confianza = st.slider(
            "Umbral de confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Si la confianza máxima es menor a este valor, se mostrará la respuesta de respaldo.",
        )
        modo_depuracion = st.toggle(
            "Modo depuración",
            value=False,
            help="Muestra la intención detectada y el porcentaje de confianza en cada respuesta.",
        )
        if st.button("Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        st.warning(
            "Aún no existen los artefactos entrenados (`models/modelo.pkl` y `models/vectorizador.pkl`). "
            "Ejecuta primero el notebook `notebooks/entrenamiento_evaluacion.ipynb` para generarlos."
        )
        st.stop()

    if not CORPUS_PATH.exists():
        st.error(
            f"No se encontró el corpus en `{CORPUS_PATH.relative_to(BASE_DIR)}`."
        )
        st.stop()

    modelo, vectorizador = cargar_artefactos(MODEL_PATH, VECTORIZER_PATH)
    respuestas = cargar_respuestas(CORPUS_PATH)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for mensaje in st.session_state.messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])
            if modo_depuracion and mensaje["role"] == "assistant":
                intencion = mensaje.get("intencion", "desconocida")
                confianza = mensaje.get("confianza", 0.0)
                estado = " (fallback)" if mensaje.get("es_fallback") else ""
                st.caption(
                    f"Intención detectada: `{intencion}`{estado} | "
                    f"Confianza: **{confianza * 100:.1f}%**"
                )

    if prompt := st.chat_input("Escribe tu pregunta sobre la UPQ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        respuesta, intencion, confianza, es_fallback = responder_consulta(
            prompt, modelo, vectorizador, respuestas, umbral_confianza
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": respuesta,
                "intencion": intencion,
                "confianza": confianza,
                "es_fallback": es_fallback,
            }
        )
        with st.chat_message("assistant"):
            st.markdown(respuesta)
            if modo_depuracion:
                estado = " (fallback)" if es_fallback else ""
                st.caption(
                    f"Intención detectada: `{intencion}`{estado} | "
                    f"Confianza: **{confianza * 100:.1f}%**"
                )


if __name__ == "__main__":
    main()
