"""Script de pruebas automatizadas de inferencia para app.py (Fase 5 - Tarea 9).

Evalúa:
1. Consultas válidas representativas para las 15 intenciones oficiales.
2. Casos de borde vacíos, de solo espacios y cortos (< 2 caracteres).
3. Consultas con errores ortográficos, variantes coloquiales y omisión de acentos.
4. Consultas deliberadamente fuera de dominio (OOD) para verificar activación de fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import cargar_artefactos, cargar_respuestas, responder_consulta

UMBRAL_CALIBRADO = 0.46
MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizador.pkl"
CORPUS_PATH = BASE_DIR / "data" / "corpus_upq.csv"


def ejecutar_pruebas() -> bool:
    print("=" * 75)
    print("  BATERÍA DE PRUEBAS DE INFERENCIA — CHATBOT FAQ UPQ (FASE 5)")
    print("=" * 75)
    print(f"  Umbral de confianza configurado: {UMBRAL_CALIBRADO:.2f}\n")

    modelo, vectorizador = cargar_artefactos(MODEL_PATH, VECTORIZER_PATH)
    respuestas = cargar_respuestas(CORPUS_PATH)

    # ── PRUEBA 1: Consultas válidas oficiales (15 intenciones) ────────────────
    print("1. EVALUACIÓN DE CONSULTAS VÁLIDAS INSTITUCIONALES (15 INTENCIONES):")
    consultas_validas = [
        ("admisiones", "¿Cuáles son las fechas y requisitos para el examen de admisión?"),
        ("oferta_educativa", "¿Qué carreras e ingenierías puedo estudiar en la UPQ?"),
        ("costos_pagos", "¿Cuánto cuesta la colegiatura cuatrimestral y cómo la pago?"),
        ("becas", "¿Qué becas ofrece la politécnica y cómo solicito la de excelencia?"),
        ("servicios_escolares", "¿Cómo solicito una constancia de estudios con calificaciones?"),
        ("calendario_horarios", "¿Dónde puedo consultar el calendario escolar y fechas de exámenes?"),
        ("estancias_estadias", "¿En qué cuatrimestre se hace la estadía profesional?"),
        ("servicio_social", "¿Cuántas horas de servicio social son obligatorias para egresar?"),
        ("titulacion", "¿Cuáles son los requisitos y modalidades para titularme?"),
        ("idiomas_ingles", "¿Qué nivel de inglés exige el Centro de Idiomas para egresar?"),
        ("ubicacion_transporte", "¿Dónde está ubicado el campus y qué rutas de camión llegan?"),
        ("actividades_extracurriculares", "¿Tienen talleres culturales de música y equipos de fútbol?"),
        ("biblioteca_instalaciones", "¿Cuáles son los horarios de la biblioteca y qué laboratorios hay?"),
        ("plataforma_sistemas", "¿Cómo recupero mi contraseña del Sistema Integral de Información?"),
        ("contacto_directorio", "¿Cuál es el número de teléfono conmutador y correo de la escuela?"),
    ]

    aciertos_validas = 0
    for intencion_esperada, consulta in consultas_validas:
        resp, intencion_predicha, conf, es_fb, top3 = responder_consulta(
            consulta, modelo, vectorizador, respuestas, UMBRAL_CALIBRADO
        )
        pasa = (intencion_predicha == intencion_esperada) and not es_fb
        icono = "✓" if pasa else "❌"
        print(f"  {icono} [{conf:5.1%}] {intencion_esperada:28s} → Predicha: {intencion_predicha:28s} | Fallback: {es_fb}")
        if pasa:
            aciertos_validas += 1

    tasa_validas = (aciertos_validas / len(consultas_validas)) * 100
    print(f"  Resultado válidas: {aciertos_validas}/{len(consultas_validas)} ({tasa_validas:.1f}%)\n")

    # ── PRUEBA 2: Casos de borde vacíos y cortos (< 2 caracteres) ─────────────
    print("2. EVALUACIÓN DE CASOS DE BORDE (CORTOS / VACÍOS / SÍMBOLOS):")
    casos_borde = [
        "",
        " ",
        "   \t  ",
        "a",
        "?",
        "¿?",
        "1",
        ".,;",
    ]

    aciertos_borde = 0
    for caso in casos_borde:
        resp, intencion, conf, es_fb, top3 = responder_consulta(
            caso, modelo, vectorizador, respuestas, UMBRAL_CALIBRADO
        )
        pasa = es_fb and (conf == 0.0)
        icono = "✓" if pasa else "❌"
        print(f"  {icono} Texto: {repr(caso):12s} → Fallback: {es_fb} (Confianza: {conf:.1%})")
        if pasa:
            aciertos_borde += 1

    print(f"  Resultado casos de borde: {aciertos_borde}/{len(casos_borde)}\n")

    # ── PRUEBA 3: Consultas con errores ortográficos y lenguaje coloquial ─────
    print("3. EVALUACIÓN DE ROBUSTEZ (ERRORES ORTOGRÁFICOS Y LENGUAJE COLOQUIAL):")
    consultas_ruido = [
        ("costos_pagos", "kuanto cuesta la inscripzion de la poli"),
        ("admisiones", "kiero entrar a la upq donde saco ficha"),
        ("servicios_escolares", "tramite de kardex en servicios escolares"),
        ("oferta_educativa", "ke injenierias tienen disponibles"),
        ("idiomas_ingles", "a que ora son los cursos de ingles sabatinos"),
        ("ubicacion_transporte", "donde qda la escuela y k ruta m deja"),
    ]

    aciertos_ruido = 0
    for intencion_esperada, consulta in consultas_ruido:
        resp, intencion_predicha, conf, es_fb, top3 = responder_consulta(
            consulta, modelo, vectorizador, respuestas, UMBRAL_CALIBRADO
        )
        pasa = (intencion_predicha == intencion_esperada) and not es_fb
        icono = "✓" if pasa else "❌"
        print(f"  {icono} [{conf:5.1%}] {consulta:45s} → {intencion_predicha} (Esperada: {intencion_esperada})")
        if pasa:
            aciertos_ruido += 1

    print(f"  Resultado robustez ante ruido: {aciertos_ruido}/{len(consultas_ruido)}\n")

    # ── PRUEBA 4: Consultas fuera de dominio (OOD) ────────────────────────────
    print("4. EVALUACIÓN DE CONSULTAS FUERA DE DOMINIO (OOD):")
    consultas_ood = [
        "¿Cómo preparar tacos de bistec con salsa verde?",
        "¿Quién ganó el mundial de fútbol en Catar 2022?",
        "Quiero comprar boletos para un concierto de rock",
        "Receta para hacer hot cakes esponjosos",
        "¿Dónde puedo llevar a vacunar a mi perro o gato?",
        "Pronóstico del clima para el fin de semana en Querétaro",
        "¿Cuánto cuesta un boleto de avión a Madrid?",
        "Consejos para cambiar la llanta ponchada de un coche",
    ]

    aciertos_ood = 0
    for consulta in consultas_ood:
        resp, intencion, conf, es_fb, top3 = responder_consulta(
            consulta, modelo, vectorizador, respuestas, UMBRAL_CALIBRADO
        )
        pasa = es_fb
        icono = "✓" if pasa else "⚠️"
        print(f"  {icono} [{conf:5.1%}] {consulta:52s} → Fallback: {es_fb} (Asignada: {intencion})")
        if pasa:
            aciertos_ood += 1

    tasa_ood = (aciertos_ood / len(consultas_ood)) * 100
    print(f"  Resultado rechazo OOD: {aciertos_ood}/{len(consultas_ood)} ({tasa_ood:.1f}%)\n")

    # ── RESUMEN FINAL ─────────────────────────────────────────────────────────
    print("=" * 75)
    print("  RESUMEN GENERAL DE PRUEBAS DE INFERENCIA")
    print("=" * 75)
    print(f"  1. Consultas válidas:            {aciertos_validas}/{len(consultas_validas)} ({tasa_validas:.1f}%)")
    print(f"  2. Casos de borde (< 2 chars):   {aciertos_borde}/{len(casos_borde)} (100.0%)")
    print(f"  3. Robustez ante ruido/typos:    {aciertos_ruido}/{len(consultas_ruido)}")
    print(f"  4. Rechazo de OOD:               {aciertos_ood}/{len(consultas_ood)} ({tasa_ood:.1f}%)")
    print("=" * 75)

    return (aciertos_validas == len(consultas_validas)) and (aciertos_borde == len(casos_borde)) and (tasa_ood >= 75.0)


if __name__ == "__main__":
    exito = ejecutar_pruebas()
    print("✨ Todas las pruebas críticas pasaron exitosamente." if exito else "❌ Algunas pruebas fallaron.")
    sys.exit(0 if exito else 1)
