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
    "Lo siento, no estoy seguro de haber entendido tu consulta sobre la UPQ. "
    "Por favor intenta reformular tu pregunta o verificar palabras clave. "
    "Puedo orientarte en temas como: admisiones, oferta educativa e ingenierías, "
    "costos y pagos (colegiaturas), becas, servicios escolares (constancias, kárdex), "
    "calendario y horarios, estancias y estadías, servicio social, titulación, "
    "cursos de idiomas (inglés), transporte y ubicación del campus, actividades "
    "culturales y deportivas, biblioteca y laboratorios, plataforma del SII o "
    "directorio de contacto institucional."
)

CONSULTAS_EJEMPLO = [
    ("🎓 Admisiones", "¿Cuáles son los requisitos y fechas para el examen de admisión?"),
    ("📚 Oferta Educativa", "¿Qué ingenierías y licenciaturas se pueden estudiar en la UPQ?"),
    ("💰 Costos y Pagos", "¿Cuánto cuesta la colegiatura cuatrimestral y cómo genero mi ficha?"),
    ("📝 Servicios Escolares", "¿Cómo tramito una constancia de estudios o mi kárdex oficial?"),
    ("🚌 Transporte", "¿Qué rutas de QroBús llegan a la UPQ y cuál es la dirección?"),
    ("🍕 Fuera de Dominio (OOD)", "¿Cómo preparar una pizza casera paso a paso?"),
]

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
) -> tuple[str, str, float, bool, list[tuple[str, float]]]:
    """Clasifica la consulta y devuelve:
    (respuesta, intencion_ganadora, confianza, es_fallback, top3_candidatas)
    """
    texto_limpio = limpiar_texto(consulta)
    fallback = respuestas.get("_fallback", FALLBACK_MESSAGE)

    # Contrato técnico: longitud mínima de 2 caracteres en el texto limpio
    if not texto_limpio or len(texto_limpio) < 2:
        return fallback, "ninguna", 0.0, True, []

    X_vectorizado = vectorizador.transform([texto_limpio])
    probabilidades = modelo.predict_proba(X_vectorizado)[0]

    # Ordenar probabilidades de mayor a menor para extraer Top-3
    indices_ordenados = np.argsort(probabilidades)[::-1]
    indice_max = int(indices_ordenados[0])
    confianza = float(probabilidades[indice_max])
    intencion = str(modelo.classes_[indice_max])

    top3 = [
        (str(modelo.classes_[i]), float(probabilidades[i]))
        for i in indices_ordenados[:3]
    ]

    # Criterio de fallback por umbral de confianza calibrado
    if confianza < umbral:
        return fallback, intencion, confianza, True, top3

    respuesta = respuestas.get(intencion, fallback)
    return respuesta, intencion, confianza, False, top3


def main() -> None:
    st.title("🎓 Chatbot FAQ - UPQ")
    st.caption(
        "Asistente local inteligente de la Universidad Politécnica de Querétaro. "
        "Clasificación supervisada con TF-IDF y Machine Learning local."
    )

    with st.sidebar:
        st.header("⚙️ Configuración")
        umbral_confianza = st.slider(
            "Umbral de confianza (Fallback)",
            min_value=0.0,
            max_value=1.0,
            value=0.46,
            step=0.01,
            help="Valor óptimo calibrado en la Fase 4 (0.46). Consultas con confianza máxima menor activarán el mensaje de respaldo.",
        )
        modo_depuracion = st.toggle(
            "Modo depuración (Top-3)",
            value=False,
            help="Muestra las 3 intenciones con mayor probabilidad y el porcentaje de confianza en cada consulta.",
        )
        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("⚡ Pruebas Rápidas")
        st.caption("Haz clic en cualquier ejemplo para probar el chatbot:")

        consulta_seleccionada = None
        for etiqueta, texto_ejemplo in CONSULTAS_EJEMPLO:
            if st.button(etiqueta, use_container_width=True, help=texto_ejemplo):
                consulta_seleccionada = texto_ejemplo

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        st.warning(
            "Aún no existen los artefactos entrenados (`models/modelo.pkl` y `models/vectorizador.pkl`). "
            "Ejecuta el script de entrenamiento para generarlos."
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

    # Renderizar historial de mensajes
    for mensaje in st.session_state.messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])
            if modo_depuracion and mensaje["role"] == "assistant":
                intencion = mensaje.get("intencion", "desconocida")
                confianza = mensaje.get("confianza", 0.0)
                estado = " ⚠️ (FALLBACK ACTIVADO)" if mensaje.get("es_fallback") else " ✅ (RESPUESTA OFICIAL)"
                top3 = mensaje.get("top3", [])

                st.caption(
                    f"**Intención:** `{intencion}`{estado} | **Confianza:** **{confianza * 100:.1f}%**"
                )
                if top3:
                    detalles_top3 = " • ".join(
                        [f"`{clase}`: **{prob * 100:.1f}%**" for clase, prob in top3]
                    )
                    st.caption(f"**Top-3 candidatas:** {detalles_top3}")

    # Determinar prompt: si se hizo clic en prueba rápida o se escribió en el input
    prompt = consulta_seleccionada or st.chat_input("Escribe tu pregunta sobre la UPQ...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        respuesta, intencion, confianza, es_fallback, top3 = responder_consulta(
            prompt, modelo, vectorizador, respuestas, umbral_confianza
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": respuesta,
                "intencion": intencion,
                "confianza": confianza,
                "es_fallback": es_fallback,
                "top3": top3,
            }
        )
        with st.chat_message("assistant"):
            st.markdown(respuesta)
            if modo_depuracion:
                estado = " ⚠️ (FALLBACK ACTIVADO)" if es_fallback else " ✅ (RESPUESTA OFICIAL)"
                st.caption(
                    f"**Intención:** `{intencion}`{estado} | **Confianza:** **{confianza * 100:.1f}%**"
                )
                if top3:
                    detalles_top3 = " • ".join(
                        [f"`{clase}`: **{prob * 100:.1f}%**" for clase, prob in top3]
                    )
                    st.caption(f"**Top-3 candidatas:** {detalles_top3}")


if __name__ == "__main__":
    main()
