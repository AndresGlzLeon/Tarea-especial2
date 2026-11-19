"""Funciones de preprocesamiento de texto para el chatbot FAQ de la UPQ."""

import re
import unicodedata


def limpiar_texto(texto: str) -> str:
    """Normaliza un texto en español para clasificación supervisada.

    Pasos aplicados:
    1. Convierte a minúsculas.
    2. Elimina acentos y diacríticos usando normalización Unicode (NFD).
    3. Elimina signos de puntuación y caracteres especiales.
    4. Normaliza espacios en blanco consecutivos.
    """
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto_normalizado = unicodedata.normalize("NFD", texto)
    texto_sin_acentos = "".join(
        caracter
        for caracter in texto_normalizado
        if unicodedata.category(caracter) != "Mn"
    )
    texto_sin_puntuacion = re.sub(r"[^a-z0-9\s]", " ", texto_sin_acentos)
    return re.sub(r"\s+", " ", texto_sin_puntuacion).strip()

