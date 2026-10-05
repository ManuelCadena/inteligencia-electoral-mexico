#!/usr/bin/env python3
"""
build_canonical_dataset.py
==========================
Toma las series tidy generadas por build_series.py y construye un dataset
final, normalizado y listo para modelos de aprendizaje profundo y series
temporales:

  - Limpia valores fantasma, BOM y problemas de codificación.
  - Normaliza nombres de entidad y asigna claves INE/INEGI estables.
  - Calcula variables de outcome (share, participación, margen, ganador,
    alternancia, número efectivo de partidos).
  - Añade variables temporales/estructurales (tipo de elección, incumbencia,
    mid-term, etc.).
  - Integra features exógenas (nacionales y por entidad) con ventanas de
    alineación temporal y advertencias de data leakage.
  - Exporta formatos long (tidy por partido) y wide (MLP) y metadatos JSON.

Uso:
  python scripts/build_canonical_dataset.py \
      --series-dir data/processed/series \
      --features-dir data/processed/features \
      --out-dir data/processed/canonical

Requiere: pandas, numpy.
"""

import argparse
import json
import logging
import pathlib
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("canonical")

# ---------------------------------------------------------------------------
# Catálogos estables
# ---------------------------------------------------------------------------

# Abreviaturas usadas por el INE en los nombres de archivo locales.
STATE_ABBREVS: Dict[str, Dict[str, str]] = {
    "AGS": {"nombre": "AGUASCALIENTES", "id_ine": "01"},
    "BC": {"nombre": "BAJA CALIFORNIA", "id_ine": "02"},
    "BCS": {"nombre": "BAJA CALIFORNIA SUR", "id_ine": "03"},
    "CAMP": {"nombre": "CAMPECHE", "id_ine": "04"},
    "COAH": {"nombre": "COAHUILA", "id_ine": "05"},
    "COL": {"nombre": "COLIMA", "id_ine": "06"},
    "CHIS": {"nombre": "CHIAPAS", "id_ine": "07"},
    "CHIH": {"nombre": "CHIHUAHUA", "id_ine": "08"},
    "CDMX": {"nombre": "CIUDAD DE MEXICO", "id_ine": "09"},
    "DF": {"nombre": "CIUDAD DE MEXICO", "id_ine": "09"},
    "DGO": {"nombre": "DURANGO", "id_ine": "10"},
    "GTO": {"nombre": "GUANAJUATO", "id_ine": "11"},
    "GRO": {"nombre": "GUERRERO", "id_ine": "12"},
    "HGO": {"nombre": "HIDALGO", "id_ine": "13"},
    "JAL": {"nombre": "JALISCO", "id_ine": "14"},
    "MEX": {"nombre": "MEXICO", "id_ine": "15"},
    "MICH": {"nombre": "MICHOACAN", "id_ine": "16"},
    "MOR": {"nombre": "MORELOS", "id_ine": "17"},
    "NAY": {"nombre": "NAYARIT", "id_ine": "18"},
    "NL": {"nombre": "NUEVO LEON", "id_ine": "19"},
    "OAX": {"nombre": "OAXACA", "id_ine": "20"},
    "PUE": {"nombre": "PUEBLA", "id_ine": "21"},
    "QRO": {"nombre": "QUERETARO", "id_ine": "22"},
    "QROO": {"nombre": "QUINTANA ROO", "id_ine": "23"},
    "SLP": {"nombre": "SAN LUIS POTOSI", "id_ine": "24"},
    "SIN": {"nombre": "SINALOA", "id_ine": "25"},
    "SON": {"nombre": "SONORA", "id_ine": "26"},
    "TAB": {"nombre": "TABASCO", "id_ine": "27"},
    "TAMPS": {"nombre": "TAMAULIPAS", "id_ine": "28"},
    "TLAX": {"nombre": "TLAXCALA", "id_ine": "29"},
    "VER": {"nombre": "VERACRUZ", "id_ine": "30"},
    "YUC": {"nombre": "YUCATAN", "id_ine": "31"},
    "ZAC": {"nombre": "ZACATECAS", "id_ine": "32"},
}

# Construye mapa inverso por nombre normalizado.
_NAME_TO_ABBREV: Dict[str, str] = {}
for abbr, info in STATE_ABBREVS.items():
    _NAME_TO_ABBREV[info["nombre"]] = abbr

PRESIDENTIAL_YEARS: List[int] = [1994, 2000, 2006, 2012, 2018, 2024]
MIDTERM_YEARS: List[int] = [1991, 1997, 2003, 2009, 2015, 2021]

BLACKLIST_PARTY_VALUES = {
    "Ï»¿ID_ESTADO",
    "ID_ESTADO",
    "DISTRITOS_LOCALES",
    "MUNICIPIOS",
    "RUTA_ACTA",
    "OBSERVACIONES",
    "",
}

# ---------------------------------------------------------------------------
# Helpers de limpieza
# ---------------------------------------------------------------------------


def _strip_bom(val):
    """Elimina el BOM de strings y nombres de columna."""
    if isinstance(val, str):
        return val.replace("\ufeff", "").replace("\xef\xbb\xbf", "").strip()
    return val


def fix_mojibake(s: str) -> str:
    """
    Corrige el caso común de CSV codificado en latin-1 pero leído como utf-8.
    Detecta la presencia de 'Ã' (U+00C3) seguido de un byte de continuación y,
    de ser posible, decodifica como latin-1 -> utf-8.
    """
    if not isinstance(s, str):
        return s
    if "\u00c3" in s:
        try:
            fixed = s.encode("latin-1").decode("utf-8")
            if "\u00c3" not in fixed:
                return fixed
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return s


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_strip_bom(c) for c in df.columns]
    df.columns = [fix_mojibake(c) for c in df.columns]
    return df


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].apply(_strip_bom).apply(fix_mojibake)
    return df


# ---------------------------------------------------------------------------
# Normalización geográfica
# ---------------------------------------------------------------------------


def _state_abbrev_from_filename(archivo: str) -> Optional[str]:
    """Extrae la abreviatura de entidad de nombres como 2018_SEE_GOB_CDMX_ENT.csv."""
    m = re.search(r"_([A-Z]+)_(?:ENT|MUN|DIS)\.CSV$", str(archivo).upper())
    return m.group(1) if m else None


def normalize_state_name(name: str) -> str:
    """Mapea nombres observados a nombres canónicos del catálogo."""
    name = str(name).strip().upper()
    # Limpiamos espacios múltiples.
    name = re.sub(r"\s+", " ", name)
    # Equivalencias directas.
    if name in ("CIUDAD DE MEXICO", "CIUDAD DE MÉXICO", "DISTRITO FEDERAL"):
        return "CIUDAD DE MEXICO"
    if name in ("MEXICO", "MÉXICO"):
        return "MEXICO"
    if name in ("MICHOACAN", "MICHOACÁN"):
        return "MICHOACAN"
    if name in ("NUEVO LEON", "NUEVO LEÓN"):
        return "NUEVO LEON"
    if name in ("QUERETARO", "QUERÉTARO"):
        return "QUERETARO"
    if name in ("SAN LUIS POTOSI", "SAN LUIS POTOSÍ"):
        return "SAN LUIS POTOSI"
    if name in ("YUCATAN", "YUCATÁN"):
        return "YUCATAN"
    # Sinónimos comunes.
    synonyms = {
        "BAJA CALIFORNIA SUR": "BAJA CALIFORNIA SUR",
        "BAJA CALIFORNIA": "BAJA CALIFORNIA",
    }
    return synonyms.get(name, name)


def add_state_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Añade ENTIDAD_CANONICA e ID_ENTIDAD a partir de NOMBRE_ESTADO o ARCHIVO."""
    from_name = (
        df["NOMBRE_ESTADO"].apply(
            lambda x: normalize_state_name(x) if pd.notna(x) else np.nan
        )
        if "NOMBRE_ESTADO" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    from_file = df["ARCHIVO"].apply(
        lambda x: STATE_ABBREVS.get(_state_abbrev_from_filename(x), {}).get("nombre")
    )
    # Prioridad: nombre del archivo fuente cuando el nombre de estado falta.
    df["ENTIDAD_CANONICA"] = from_name.fillna(from_file)

    def _id(entidad):
        for abbr, info in STATE_ABBREVS.items():
            if info["nombre"] == entidad:
                return info["id_ine"]
        return None

    df["ID_ENTIDAD"] = df["ENTIDAD_CANONICA"].apply(_id)
    return df


# ---------------------------------------------------------------------------
# Carga y estandarización de series
# ---------------------------------------------------------------------------


def _load_series(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df = clean_columns(df)
    df = clean_string_columns(df)
    return df


SERIES_CONFIG = [
    {
        "file": "serie_presidencia_entidad.csv",
        "ambito": "FEDERAL",
        "cargo": "PRESIDENTE",
        "nivel_geo": "ENTIDAD",
    },
    {
        "file": "serie_senado_entidad.csv",
        "ambito": "FEDERAL",
        "cargo": "SENADOR",
        "nivel_geo": "ENTIDAD",
    },
    {
        "file": "serie_diputados_fed_mr_entidad.csv",
        "ambito": "FEDERAL",
        "cargo": "DIPUTADO_FEDERAL_MR",
        "nivel_geo": "ENTIDAD",
    },
    {
        "file": "serie_diputados_fed_rp_entidad.csv",
        "ambito": "FEDERAL",
        "cargo": "DIPUTADO_FEDERAL_RP",
        "nivel_geo": "ENTIDAD",
    },
    {
        "file": "serie_estatal_gobernador.csv",
        "ambito": "LOCAL",
        "cargo": "GOBERNADOR",
        "nivel_geo": "ENTIDAD",
    },
    {
        "file": "serie_estatal_diputados_local.csv",
        "ambito": "LOCAL",
        "cargo": "DIPUTADO_LOCAL",
        "nivel_geo": "DISTRITO",
    },
    {
        "file": "serie_municipal_ayuntamiento.csv",
        "ambito": "LOCAL",
        "cargo": "AYUNTAMIENTO",
        "nivel_geo": "MUNICIPIO",
    },
    {
        "file": "serie_municipal_otros.csv",
        "ambito": "LOCAL",
        "cargo": "OTROS_MUNICIPALES",
        "nivel_geo": "MUNICIPIO",
    },
]


def build_raw_long(series_dir: pathlib.Path) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    for cfg in SERIES_CONFIG:
        path = series_dir / cfg["file"]
        if not path.exists():
            log.warning("No existe %s", path)
            continue
        log.info("Cargando %s", cfg["file"])
        df = _load_series(path)
        df["AMBITO"] = cfg["ambito"]
        df["CARGO"] = cfg["cargo"]
        # Distinguir fórmulas MR/RP dentro del mismo cargo cuando el nombre de
        # archivo lo indica (p. ej. SEN_MR vs SEN_RP, DIP_LOC_MR vs DIP_LOC_RP).
        if cfg["cargo"] in ("SENADOR", "DIPUTADO_LOCAL"):
            is_rp = df["ARCHIVO"].astype(str).str.upper().str.contains(r"_RP_")
            df["CARGO"] = np.where(is_rp, cfg["cargo"] + "_RP", cfg["cargo"] + "_MR")
        df["NIVEL_GEO"] = cfg["nivel_geo"]
        df = add_state_ids(df)
        # Coalesce de columnas geográficas a clave genérica GEO_KEY_2.
        geo_candidates = [
            "MUNICIPIO",
            "NOMBRE_MUNICIPIO",
            "ID_MUNICIPIO",
            "ID_DISTRITO",
            "ID_DISTRITO_LOCAL",
            "DISTRITO",
            "DISTRITO_LOCAL",
            "NOMBRE_DISTRITO",
            "CABECERA_DISTRITAL_LOCAL",
            "SECCION",
        ]
        geo_cols = [c for c in geo_candidates if c in df.columns]
        if geo_cols:
            df["GEO_KEY_2"] = df[geo_cols].bfill(axis=1).iloc[:, 0]
        else:
            df["GEO_KEY_2"] = "_TODOS"
        df["PARTIDO_COALICION"] = df["PARTIDO_COALICION"].astype(str).str.strip().str.upper()
        pieces.append(df)
    long = pd.concat(pieces, ignore_index=True)
    return long


# ---------------------------------------------------------------------------
# Filtrado y cálculo de outcomes
# ---------------------------------------------------------------------------


def filter_valid_rows(long: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas de encabezado/totales y normaliza votos."""
    long["PARTIDO_COALICION"] = long["PARTIDO_COALICION"].astype(str).str.strip().str.upper()
    # Eliminar encabezados/palabras reservadas.
    long = long[~long["PARTIDO_COALICION"].isin(BLACKLIST_PARTY_VALUES)]
    long = long[long["PARTIDO_COALICION"].notna()]
    # Convertir VOTOS a numérico.
    long["VOTOS"] = pd.to_numeric(long["VOTOS"], errors="coerce")
    # Para candidatos independientes y filas técnicas: conservar solo si tienen voto.
    is_tech = long["PARTIDO_COALICION"].str.match(r"^(CAND_IND|RUTA_ACTA|OBSERVACIONES)", na=False)
    long = long[~(is_tech & long["VOTOS"].isna())]
    # Para partidos reales, un voto faltante se interpreta como 0 (no participación).
    real_party_mask = ~long["PARTIDO_COALICION"].str.match(
        r"^(CAND_IND|RUTA_ACTA|OBSERVACIONES|DISTRITOS_LOCALES|MUNICIPIOS|ID_ESTADO)", na=False
    )
    # Filas totalmente vacías (pies de archivo): sin voto ni contexto electoral.
    if "TOTAL_VOTOS" in long.columns:
        long = long[~(long["VOTOS"].isna() & pd.to_numeric(long["TOTAL_VOTOS"], errors="coerce").isna())]
    long.loc[real_party_mask & long["VOTOS"].isna(), "VOTOS"] = 0.0
    long = long[long["VOTOS"].notna() & (long["VOTOS"] >= 0)]
    # Filas sin clave geográfica asignable en niveles subestatales.
    needs_geo = long["NIVEL_GEO"].isin(["DISTRITO", "MUNICIPIO"])
    long = long[~(needs_geo & long["GEO_KEY_2"].isna())]
    # Convertir columnas numéricas de contexto.
    num_cols = ["TOTAL_VOTOS", "LISTA_NOMINAL", "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG"]
    for c in num_cols:
        if c in long.columns:
            long[c] = pd.to_numeric(long[c], errors="coerce")
    # Eliminar filas sin entidad identificada.
    long = long[long["ENTIDAD_CANONICA"].notna()]
    # Consolidar duplicados exactos por clave electoral-geográfica-partido.
    key = [
        "AMBITO", "CARGO", "NIVEL_GEO", "ANIO", "ENTIDAD_CANONICA",
        "ID_ENTIDAD", "GEO_KEY_2", "PARTIDO_COALICION", "ARCHIVO", "NIVEL",
    ]
    key = [c for c in key if c in long.columns]
    if long.duplicated(subset=key).any():
        log.info("Consolidando %s filas duplicadas por clave", long.duplicated(subset=key).sum())
        # Algunos archivos repiten la misma entidad por TERRITORIO (voto en el
        # extranjero vs. nacional): los totales también deben sumarse.
        sum_cols = {
            "VOTOS", "TOTAL_VOTOS", "LISTA_NOMINAL",
            "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG", "VOTOS_VALIDOS",
        }
        agg = {
            c: ("sum" if c in sum_cols else "first")
            for c in long.columns
            if c not in key
        }
        long = long.groupby(key, as_index=False, dropna=False, observed=True).agg(agg)
    return long


def compute_outcomes(long: pd.DataFrame) -> pd.DataFrame:
    """Añade shares, participación, ganador, margen y NEP."""
    # Votos válidos por geografía-año-cargo.
    long["VOTOS_VALIDOS"] = long["TOTAL_VOTOS"] - long["NUM_VOTOS_NULOS"].fillna(0) - long[
        "NUM_VOTOS_CAN_NREG"
    ].fillna(0)
    long["VOTOS_VALIDOS"] = long["VOTOS_VALIDOS"].clip(lower=0)

    group_cols = ["AMBITO", "CARGO", "ANIO", "ENTIDAD_CANONICA", "ID_ENTIDAD", "GEO_KEY_2"]

    # Totales por grupo (se repiten en cada fila).
    group_totals = long.groupby(group_cols, observed=True, as_index=False)[
        ["TOTAL_VOTOS", "LISTA_NOMINAL", "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG", "VOTOS_VALIDOS"]
    ].first()
    long = long.drop(
        columns=["TOTAL_VOTOS", "LISTA_NOMINAL", "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG", "VOTOS_VALIDOS"],
        errors="ignore",
    )
    long = long.merge(group_totals, on=group_cols, how="left", validate="m:1")

    # Shares.
    long["SHARE_TOTAL"] = long["VOTOS"] / long["TOTAL_VOTOS"].replace(0, np.nan)
    long["SHARE_VALIDO"] = long["VOTOS"] / long["VOTOS_VALIDOS"].replace(0, np.nan)

    # Variables de contexto electoral.
    long["PARTICIPACION"] = long["TOTAL_VOTOS"] / long["LISTA_NOMINAL"].replace(0, np.nan)
    long["TASA_NULOS"] = long["NUM_VOTOS_NULOS"] / long["TOTAL_VOTOS"].replace(0, np.nan)
    long["TASA_NREG"] = long["NUM_VOTOS_CAN_NREG"] / long["TOTAL_VOTOS"].replace(0, np.nan)

    # Ranking por grupo (first = desempate por orden de aparición).
    long["RANK"] = long.groupby(group_cols)["VOTOS"].rank(method="first", ascending=False)

    # Orden determinista dentro de cada grupo: más votos primero, desempate por sigla.
    ordered = long.sort_values(
        group_cols + ["VOTOS", "PARTIDO_COALICION"],
        ascending=[True] * len(group_cols) + [False, True],
    )
    # Ganador = primer lugar (una sola fila por grupo, sin empates).
    first = (
        ordered.groupby(group_cols, as_index=False)
        .nth(0)[group_cols + ["PARTIDO_COALICION", "SHARE_VALIDO"]]
        .rename(columns={"PARTIDO_COALICION": "GANADOR", "SHARE_VALIDO": "SHARE_1"})
    )
    # Segundo lugar (una sola fila por grupo).
    second = (
        ordered.groupby(group_cols, as_index=False)
        .nth(1)[group_cols + ["SHARE_VALIDO"]]
        .rename(columns={"SHARE_VALIDO": "SHARE_2"})
    )
    # Número efectivo de partidos (Laakso-Taagepera).
    nep = (
        long.groupby(group_cols)["SHARE_VALIDO"]
        .apply(lambda s: 1.0 / (s.fillna(0).clip(lower=0) ** 2).sum() if s.sum() > 0 else np.nan)
        .reset_index(name="NEP")
    )
    # Número de partidos con votos > 0.
    num_part = long.groupby(group_cols)["VOTOS"].apply(lambda s: (s.fillna(0) > 0).sum()).reset_index(name="NUM_PARTIDOS")

    n_rows = len(long)
    for tabla, nombre in ((first, "ganador"), (second, "segundo"), (nep, "nep"), (num_part, "num_partidos")):
        assert not tabla.duplicated(subset=group_cols).any(), f"Claves duplicadas en tabla {nombre}"
        long = long.merge(tabla, on=group_cols, how="left", validate="m:1")
        assert len(long) == n_rows, f"Merge {nombre} expandió filas: {n_rows} -> {len(long)}"
    long["MARGEN_VICTORIA"] = long["SHARE_1"] - long["SHARE_2"].fillna(0)

    # Alternancia: comparar ganador con el proceso inmediato anterior del mismo grupo.
    long["ALTERNANCIA"] = 0
    group_year_winner = long.groupby(group_cols)["GANADOR"].first().reset_index()
    group_year_winner = group_year_winner.sort_values(group_cols)
    # group_cols ya incluye ANIO; shift dentro del mismo grupo (excluyendo ANIO del índice de grupo).
    group_cols_no_year = [c for c in group_cols if c != "ANIO"]
    group_year_winner["PREV_GANADOR"] = group_year_winner.groupby(group_cols_no_year)["GANADOR"].shift(1)
    group_year_winner["ALTERNANCIA"] = (
        group_year_winner["GANADOR"] != group_year_winner["PREV_GANADOR"]
    ).astype(int)
    n_rows = len(long)
    long = long.merge(
        group_year_winner[group_cols + ["ALTERNANCIA"]],
        on=group_cols,
        how="left",
        suffixes=("", "_calc"),
        validate="m:1",
    )
    assert len(long) == n_rows, f"Merge alternancia expandió filas: {n_rows} -> {len(long)}"
    long["ALTERNANCIA"] = long["ALTERNANCIA_calc"].fillna(0).astype(int)
    long = long.drop(columns=["ALTERNANCIA_calc"])

    # Ganador binario por fila.
    long["ES_GANADOR"] = (long["PARTIDO_COALICION"] == long["GANADOR"]).astype(int)

    # Bandera de consistencia: la suma de SHARE_VALIDO por grupo debe ser ~1.
    # Desviaciones se conservan (elecciones anuladas, archivos con sólo parte
    # de los partidos, o totales inconsistentes en la fuente) pero se marcan.
    share_sum = long.groupby(group_cols)["SHARE_VALIDO"].transform("sum")
    long["FLAG_CONSISTENCIA"] = np.where(
        share_sum.isna() | (share_sum < 0.9) | (share_sum > 1.1), 1, 0
    )
    return long


# ---------------------------------------------------------------------------
# Features temporales / estructurales
# ---------------------------------------------------------------------------


def add_structural_features(long: pd.DataFrame) -> pd.DataFrame:
    long["ES_PRESIDENCIAL"] = long["ANIO"].isin(PRESIDENTIAL_YEARS).astype(int)
    long["ES_INTERMEDIA"] = long["ANIO"].isin(MIDTERM_YEARS).astype(int)
    # Para gobernadores y locales no aplica directamente; se deja como 0.
    long["ES_FEDERAL"] = (long["AMBITO"] == "FEDERAL").astype(int)
    long["ES_LOCAL"] = (long["AMBITO"] == "LOCAL").astype(int)
    # Orden dentro del sexenio presidencial (1-6).
    long["AÑO_SEXENIO"] = ((long["ANIO"] - 1988) % 6).replace(0, 6)
    return long


# ---------------------------------------------------------------------------
# Integración de features exógenas
# ---------------------------------------------------------------------------


def load_features(features_dir: pathlib.Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    national = features_dir / "features_anuales_nacional.csv"
    entidad = features_dir / "features_anuales_entidad.csv"
    if national.exists():
        out["nacional"] = clean_string_columns(clean_columns(pd.read_csv(national)))
    if entidad.exists():
        out["entidad"] = clean_string_columns(clean_columns(pd.read_csv(entidad)))
    return out


def _rename_to_lag(df: pd.DataFrame, year_col: str, lag: int) -> pd.DataFrame:
    df = df.copy()
    df[year_col] = df[year_col] + lag
    return df


def merge_exogenous(
    long: pd.DataFrame,
    features: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Une features nacionales y por entidad, generando versiones contemporánea y rezagada."""
    n_rows = len(long)
    if "nacional" in features:
        nat = features["nacional"].rename(columns={"anio": "ANIO"})
        nat = nat.drop_duplicates(subset="ANIO")
        nat_lag1 = _rename_to_lag(nat.copy(), "ANIO", 1)
        long = long.merge(nat, on="ANIO", how="left", suffixes=("", ""), validate="m:1")
        long = long.merge(
            nat_lag1,
            on="ANIO",
            how="left",
            suffixes=("", "_LAG1_NAC"),
            validate="m:1",
        )
        assert len(long) == n_rows, f"Merge nacional expandió filas: {n_rows} -> {len(long)}"

    if "entidad" in features:
        ent = features["entidad"].rename(
            columns=lambda c: (
                c + "_ENT" if c not in ("anio", "ENTIDAD_CANONICA", "ID_ENTIDAD", "NOMBRE_ESTADO", "ENTIDAD") else c
            )
        )
        # Renombrar año a ANIO para merge uniforme.
        if "anio" in ent.columns:
            ent = ent.rename(columns={"anio": "ANIO"})
        # Normalizar nombre de entidad en features por entidad.
        if "ENTIDAD_CANONICA" in ent.columns:
            ent["ENTIDAD_CANONICA"] = ent["ENTIDAD_CANONICA"].apply(normalize_state_name)
        elif "NOMBRE_ESTADO" in ent.columns:
            ent["ENTIDAD_CANONICA"] = ent["NOMBRE_ESTADO"].apply(normalize_state_name)
            ent = ent.drop(columns=["NOMBRE_ESTADO"])
        elif "ENTIDAD" in ent.columns:
            ent["ENTIDAD_CANONICA"] = ent["ENTIDAD"].apply(normalize_state_name)
            ent = ent.drop(columns=["ENTIDAD"])

        ent_lag1 = _rename_to_lag(ent.copy(), "ANIO", 1)
        merge_cols = ["ANIO"]
        if "ENTIDAD_CANONICA" in long.columns and "ENTIDAD_CANONICA" in ent.columns:
            merge_cols.append("ENTIDAD_CANONICA")
        if "ID_ENTIDAD" in long.columns and "ID_ENTIDAD" in ent.columns:
            merge_cols.append("ID_ENTIDAD")

        ent = ent.drop_duplicates(subset=merge_cols)
        ent_lag1 = ent_lag1.drop_duplicates(subset=merge_cols)
        long = long.merge(ent, on=merge_cols, how="left", suffixes=("", "_ENT"), validate="m:1")
        long = long.merge(
            ent_lag1,
            on=merge_cols,
            how="left",
            suffixes=("", "_LAG1_ENT"),
            validate="m:1",
        )
        assert len(long) == n_rows, f"Merge entidad expandió filas: {n_rows} -> {len(long)}"

    # Eliminar columnas duplicadas de año que pudieran quedar.
    for col in ["anio"]:
        if col in long.columns:
            long = long.drop(columns=[col])
    return long


# ---------------------------------------------------------------------------
# Wide format para modelos tabulares
# ---------------------------------------------------------------------------


def build_wide(long: pd.DataFrame, top_n_parties: int = 12) -> pd.DataFrame:
    """
    Construye una tabla ancha donde cada fila es una geografía-año-cargo y las
    columnas son shares de los N partidos más votados históricamente más
    variables estructurales.
    """
    # Elegir los N partidos/coaliciones más frecuentes en votos totales.
    party_votes = long.groupby("PARTIDO_COALICION")["VOTOS"].sum().sort_values(ascending=False)
    top_parties = party_votes.head(top_n_parties).index.tolist()

    group_cols = ["AMBITO", "CARGO", "ANIO", "ENTIDAD_CANONICA", "ID_ENTIDAD", "GEO_KEY_2"]
    structural = [
        "TOTAL_VOTOS",
        "LISTA_NOMINAL",
        "VOTOS_VALIDOS",
        "NUM_VOTOS_NULOS",
        "NUM_VOTOS_CAN_NREG",
        "PARTICIPACION",
        "TASA_NULOS",
        "TASA_NREG",
        "GANADOR",
        "MARGEN_VICTORIA",
        "NEP",
        "NUM_PARTIDOS",
        "ALTERNANCIA",
        "ES_PRESIDENCIAL",
        "ES_INTERMEDIA",
        "ES_FEDERAL",
        "ES_LOCAL",
        "AÑO_SEXENIO",
    ]
    # Añadir features exógenas si existen.
    feat_cols = [c for c in long.columns if c.startswith(("enco_", "panel_", "ocde_", "fred_", "remesas_"))]

    base = long[group_cols + structural + feat_cols].drop_duplicates(subset=group_cols)

    # Pivotear shares.
    shares = long[long["PARTIDO_COALICION"].isin(top_parties)][
        group_cols + ["PARTIDO_COALICION", "SHARE_VALIDO"]
    ].copy()
    shares["PARTIDO_COALICION"] = "SHARE_" + shares["PARTIDO_COALICION"].astype(str)
    wide_shares = shares.pivot_table(
        index=group_cols,
        columns="PARTIDO_COALICION",
        values="SHARE_VALIDO",
        aggfunc="first",
    ).reset_index()
    wide_shares.columns.name = None

    wide = base.merge(wide_shares, on=group_cols, how="left")

    # Asegurar que los shares sumen 1 (aprox). La diferencia la atribuimos a OTROS.
    share_cols = [c for c in wide.columns if c.startswith("SHARE_")]
    wide["SHARE_OTROS"] = 1.0 - wide[share_cols].sum(axis=1, skipna=True)
    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Construye dataset canónico normalizado.")
    parser.add_argument("--series-dir", default="data/processed/series", type=str)
    parser.add_argument("--features-dir", default="data/processed/features", type=str)
    parser.add_argument("--out-dir", default="data/processed/canonical", type=str)
    parser.add_argument("--top-parties", default=12, type=int)
    args = parser.parse_args(argv)

    series_dir = pathlib.Path(args.series_dir)
    features_dir = pathlib.Path(args.features_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Construyendo dataset canónico...")
    long = build_raw_long(series_dir)
    log.info("Filas crudas: %s | columnas: %s", long.shape[0], long.shape[1])

    long = filter_valid_rows(long)
    log.info("Filas válidas: %s", long.shape[0])

    long = compute_outcomes(long)
    long = add_structural_features(long)

    features = load_features(features_dir)
    if features:
        long = merge_exogenous(long, features)
        log.info("Features exógenas integradas. Columnas totales: %s", long.shape[1])

    # Orden final de columnas para legibilidad.
    first_cols = [
        "AMBITO",
        "CARGO",
        "NIVEL_GEO",
        "ANIO",
        "ENTIDAD_CANONICA",
        "ID_ENTIDAD",
        "GEO_KEY_2",
        "PARTIDO_COALICION",
        "VOTOS",
        "SHARE_TOTAL",
        "SHARE_VALIDO",
        "ES_GANADOR",
    ]
    ordered = first_cols + [c for c in long.columns if c not in first_cols]
    long = long[ordered]

    # Guardar formato long.
    long_path = out_dir / "canonical_long.csv"
    long.to_csv(long_path, index=False, encoding="utf-8-sig")
    log.info("Guardado canonical_long.csv: %s filas", long.shape[0])

    # Wide.
    wide = build_wide(long, top_n_parties=args.top_parties)
    wide_path = out_dir / "canonical_wide.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    log.info("Guardado canonical_wide.csv: %s filas x %s columnas", wide.shape[0], wide.shape[1])

    # Metadatos.
    meta = {
        "filas_long": int(long.shape[0]),
        "filas_wide": int(wide.shape[0]),
        "columnas_long": list(long.columns),
        "columnas_wide": list(wide.columns),
        "top_parties": wide[[c for c in wide.columns if c.startswith("SHARE_")]].columns.tolist(),
        "rangos_anio": {
            "min": int(long["ANIO"].min()),
            "max": int(long["ANIO"].max()),
        },
        "ambitos": long["AMBITO"].value_counts().to_dict(),
        "cargos": long["CARGO"].value_counts().to_dict(),
        "entidades": long["ENTIDAD_CANONICA"].nunique(),
        "niveles_geo": long["NIVEL_GEO"].value_counts().to_dict(),
        "features_exogenas": [c for c in long.columns if c.startswith(("enco_", "panel_", "ocde_", "fred_", "remesas_"))],
        "notas": [
            "SHARE_TOTAL se calcula sobre TOTAL_VOTOS; SHARE_VALIDO sobre votos válidos (total - nulos - can_nreg).",
            "Las variables _LAG1 corresponden al año anterior para reducir data leakage.",
            "Las features exógenas actuales son anuales; se recomienda construir ventanas pre-electorales más finas.",
        ],
    }
    meta_path = out_dir / "canonical_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Metadatos: %s", meta_path)

    # Diccionario de datos reducido.
    dicc = {
        "AMBITO": "Ámbito electoral: FEDERAL o LOCAL",
        "CARGO": "Cargo disputado (PRESIDENTE, SENADOR, DIPUTADO_FEDERAL_MR, ...)",
        "NIVEL_GEO": "Nivel geográfico de agregación (ENTIDAD, DISTRITO, MUNICIPIO)",
        "ANIO": "Año de la elección",
        "ENTIDAD_CANONICA": "Nombre normalizado de la entidad federativa",
        "ID_ENTIDAD": "Clave INE/INEGI de dos dígitos de la entidad",
        "GEO_KEY_2": "Distrito o municipio según el nivel geográfico",
        "PARTIDO_COALICION": "Sigla original del partido o coalición",
        "VOTOS": "Votos obtenidos",
        "SHARE_TOTAL": "Proporción sobre TOTAL_VOTOS",
        "SHARE_VALIDO": "Proporción sobre votos válidos",
        "ES_GANADOR": "1 si este partido ganó la elección en la unidad geográfica",
        "GANADOR": "Sigla del partido ganador",
        "MARGEN_VICTORIA": "Diferencia de share válido entre 1° y 2° lugar",
        "NEP": "Número efectivo de partidos (Laakso-Taagepera)",
        "NUM_PARTIDOS": "Partidos/coaliciones con votos > 0",
        "ALTERNANCIA": "1 si el ganador difiere del proceso anterior en la misma geografía",
        "PARTICIPACION": "TOTAL_VOTOS / LISTA_NOMINAL",
        "TASA_NULOS": "NUM_VOTOS_NULOS / TOTAL_VOTOS",
        "TASA_NREG": "NUM_VOTOS_CAN_NREG / TOTAL_VOTOS",
        "ES_PRESIDENCIAL": "1 si el año es elección presidencial federal",
        "ES_INTERMEDIA": "1 si el año es elección intermedia federal",
        "AÑO_SEXENIO": "Año relativo del sexenio presidencial (1-6)",
    }
    (out_dir / "canonical_data_dictionary.json").write_text(
        json.dumps(dicc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
