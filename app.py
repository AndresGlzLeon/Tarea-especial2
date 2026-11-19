"""Interfaz conversacional en Streamlit para el chatbot FAQ de la UPQ."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.preprocess import limpiar_texto

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_upq.joblib"
RESPONSES_PATH = BASE_DIR / "data" / "respuestas_upq.json"

st.set_page_config(
    page_title="Chatbot FAQ - UPQ",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def cargar_modelo(ruta: Path):
    return joblib.load(ruta)


@st.cache_data
def cargar_respuestas(ruta: Path) -> dict[str, str]:
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def responder_consulta(
    consulta: str,
    modelo,
    respuestas: dict[str, str],
    umbral: float,
) -> tuple[str, str, float, bool]:
    """Clasifica la consulta y devuelve respuesta, intención, confianza y bandera de fallback."""
    texto_limpio = limpiar_texto(consulta)
    fallback = respuestas.get(
        "_fallback",
        "Lo siento, no pude entender tu pregunta. Intenta reformularla.",
    )

    if not texto_limpio:
        return fallback, "ninguna", 0.0, True

    probabilidades = modelo.predict_proba([texto_limpio])[0]
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

    if not MODEL_PATH.exists():
        st.warning(
            "Aún no existe un modelo entrenado (`models/modelo_upq.joblib`). "
            "Ejecuta primero el script de entrenamiento desde la raíz del proyecto:\n\n"
            "```bash\npython src/train.py\n```"
        )
        st.stop()

    if not RESPONSES_PATH.exists():
        st.error(
            f"No se encontró el archivo de respuestas en `{RESPONSES_PATH.name}`."
        )
        st.stop()

    modelo = cargar_modelo(MODEL_PATH)
    respuestas = cargar_respuestas(RESPONSES_PATH)

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
            prompt, modelo, respuestas, umbral_confianza
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

