"""Script de validación automática para data/corpus_upq.csv según docs/CONTRATOS.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "corpus_upq.csv"

COLUMNAS_ESPERADAS = ["texto", "intencion", "respuesta_oficial"]
INTENCIONES_VALIDAS = {
    "admisiones",
    "oferta_educativa",
    "costos_pagos",
    "becas",
    "servicios_escolares",
    "calendario_horarios",
    "estancias_estadias",
    "servicio_social",
    "titulacion",
    "idiomas_ingles",
    "ubicacion_transporte",
    "actividades_extracurriculares",
    "biblioteca_instalaciones",
    "plataforma_sistemas",
    "contacto_directorio",
}
SNAKE_CASE_REGEX = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
MIN_VARIANTES_POR_INTENCION = 20
TARGET_VARIANTES_POR_INTENCION = 25


def validar_corpus(ruta_csv: Path = CORPUS_PATH) -> bool:
    """Valida la integridad y contratos técnicos de data/corpus_upq.csv."""
    print("=" * 70)
    print("  VALIDACIÓN AUTOMÁTICA DEL CORPUS UPQ")
    print("=" * 70)
    print(f"Archivo objetivo: {ruta_csv}\n")

    errores: list[str] = []

    # 1. Existencia del archivo
    if not ruta_csv.exists():
        print(f"❌ ERROR CRÍTICO: No se encontró el archivo en {ruta_csv}")
        return False

    # 2. Codificación UTF-8 y lectura de CSV
    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8")
        print("✓ Codificación UTF-8 válida.")
    except UnicodeDecodeError as exc:
        errores.append(f"El archivo no tiene codificación UTF-8 válida: {exc}")
        df = pd.DataFrame()
    except Exception as exc:
        errores.append(f"Error al leer el archivo CSV: {exc}")
        df = pd.DataFrame()

    if df.empty and errores:
        for err in errores:
            print(f"❌ {err}")
        return False

    # 3. Validación de columnas
    columnas_actuales = list(df.columns)
    if columnas_actuales != COLUMNAS_ESPERADAS:
        errores.append(
            f"Columnas incorrectas. Esperadas: {COLUMNAS_ESPERADAS}. Encontradas: {columnas_actuales}"
        )
    else:
        print(f"✓ Columnas requeridas presentes: {COLUMNAS_ESPERADAS}")

    # 4. Valores nulos o vacíos
    for col in COLUMNAS_ESPERADAS:
        if col in df.columns:
            nulos = df[col].isna().sum()
            vacios = (df[col].astype(str).str.strip() == "").sum()
            total_invalidos = nulos + vacios
            if total_invalidos > 0:
                errores.append(
                    f"Columna '{col}' contiene {total_invalidos} registros nulos o vacíos (nulos={nulos}, vacíos={vacios})."
                )
    if not any("nulos o vacíos" in e for e in errores):
        print("✓ Cero valores nulos y cero cadenas vacías en todas las columnas.")

    # 5. Duplicados exactos en columna 'texto'
    if "texto" in df.columns:
        duplicados_texto = df[df.duplicated(subset=["texto"], keep=False)]
        if not duplicados_texto.empty:
            num_dup = len(duplicados_texto)
            ejemplos = duplicados_texto["texto"].head(3).tolist()
            errores.append(
                f"Existen {num_dup} registros con texto duplicado exacto. Ejemplos: {ejemplos}"
            )
        else:
            print(f"✓ Cero duplicados exactos en columna 'texto' ({len(df)} textos únicos).")

    # 6. Validación de intenciones (formato snake_case y catálogo)
    if "intencion" in df.columns:
        intenciones_corpus = set(df["intencion"].dropna().unique())
        invalid_format = [
            i for i in intenciones_corpus if not SNAKE_CASE_REGEX.match(str(i))
        ]
        if invalid_format:
            errores.append(
                f"Intenciones con formato no válido (debe ser snake_case): {invalid_format}"
            )

        no_reconocidas = intenciones_corpus - INTENCIONES_VALIDAS
        if no_reconocidas:
            errores.append(
                f"Intenciones fuera del catálogo oficial de 15 intenciones: {no_reconocidas}"
            )

        faltantes = INTENCIONES_VALIDAS - intenciones_corpus
        if faltantes:
            errores.append(
                f"Faltan intenciones del catálogo oficial ({len(faltantes)} faltantes): {faltantes}"
            )

        num_intenciones = len(intenciones_corpus)
        if not (10 <= num_intenciones <= 15):
            errores.append(
                f"Número de intenciones inválido ({num_intenciones}). Debe ser entre 10 y 15."
            )
        else:
            print(f"✓ Total de intenciones presentes: {num_intenciones}/15 del catálogo oficial.")

        # Conteo de variantes por intención
        conteos = df["intencion"].value_counts()
        print("\nDistribución por intención:")
        for intencion, count in conteos.items():
            check = "✓" if count >= MIN_VARIANTES_POR_INTENCION else "❌"
            print(f"  {check} {intencion}: {count} variantes")
            if count < MIN_VARIANTES_POR_INTENCION:
                errores.append(
                    f"Intención '{intencion}' tiene {count} variantes (mínimo requerido: {MIN_VARIANTES_POR_INTENCION})."
                )

    # 7. Consistencia 1 a 1 entre intención y respuesta_oficial
    if "intencion" in df.columns and "respuesta_oficial" in df.columns:
        agrupado = (
            df.groupby("intencion")["respuesta_oficial"].nunique()
        )
        inconsistentes = agrupado[agrupado > 1]
        if not inconsistentes.empty:
            for intencion, num_resp in inconsistentes.items():
                errores.append(
                    f"Intención '{intencion}' tiene {num_resp} respuestas distintas asignadas (debe ser exactamente 1)."
                )
        else:
            print("\n✓ Consistencia 1 a 1 confirmada: 1 única respuesta oficial por intención.")

    # Resumen final
    print("\n" + "=" * 70)
    if errores:
        print("❌ FALLÓ LA VALIDACIÓN DEL CORPUS CON LOS SIGUIENTES ERRORES:")
        for err in errores:
            print(f"  - {err}")
        print("=" * 70)
        return False

    total_filas = len(df)
    print(f"✨ VALIDACIÓN EXITOSA: El corpus cumple al 100% el contrato técnico.")
    print(f"   Total de registros: {total_filas} filas")
    print(f"   Intenciones: {len(df['intencion'].unique())} clases oficiales")
    print("=" * 70)
    return True


if __name__ == "__main__":
    es_valido = validar_corpus()
    sys.exit(0 if es_valido else 1)
