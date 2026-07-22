#!/usr/bin/env python3
"""NOSA drug screening tool — filter and rank approved small molecules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from structure_match import recompute_dose_feasible_nosa, sanitize_dose_columns, sync_unified_dose

ROOT = Path(__file__).resolve().parent
DATABASE_CSV = ROOT / "nosa_drug_database.csv"
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"

PRIMARY = "#1E6464"
ACCENT = "#00BFBF"
TEXT = "#FFFFFF"
SURFACE = "#174F4F"
DISABLED = "#3A6B6B"
MEMANTINE_COLOR = "#FFD700"
MANUAL_PASS_COLOR = "#2ECC71"
MANUAL_FAIL_COLOR = "#C0392B"

MANUAL_SCORE_FILTER_OPTIONS = (
    "All",
    "Pass only (1)",
    "No pass (0)",
    "Not yet evaluated (blank)",
)

BASE_IDEAL = {
    "molecular_weight": 179.31,
    "logP": 2.69,
    "PSA": 26.02,
    "H_donors": 1.0,
    "rotatable_bonds": 0.0,
}

PUBCHEM_IDEAL = {
    "vapor_pressure_mmhg": 0.04,
    "melting_point_c": 130.0,
}

BASE_SLIDER_RANGES = {
    "molecular_weight": (0, 600),
    "logP": (-2, 8),
    "PSA": (0, 200),
    "H_donors": (0, 10),
    "rotatable_bonds": (0, 20),
}

PUBCHEM_SLIDER_RANGES = {
    "vapor_pressure_mmhg": (0, 100),
    "melting_point_c": (-100, 400),
}

CLINICAL_SLIDER_RANGES = {
    "pka_predicted": (-5, 14),
    "logD_pH6": (-5, 8),
    "max_dose_mg": (0, 500),
}

PUBCHEM_COLUMNS = list(PUBCHEM_SLIDER_RANGES.keys())
CLINICAL_COLUMNS = list(CLINICAL_SLIDER_RANGES.keys())

FILTER_LABELS = {
    "molecular_weight": "Molecular weight (Da)",
    "logP": "logP",
    "PSA": "PSA (Å²)",
    "H_donors": "H-bond donors",
    "rotatable_bonds": "Rotatable bonds",
    "vapor_pressure_mmhg": "Vapor pressure (mmHg)",
    "melting_point_c": "Melting point (°C)",
    "pka_predicted": "pKa (predicted)",
    "logD_pH6": "logD at nasal pH 6.0",
    "max_dose_mg": "Max clinical dose (mg) — source: ChEMBL or ClinicalTrials.gov",
}

FILTER_STEPS = {
    "molecular_weight": 1.0,
    "logP": 0.1,
    "PSA": 1.0,
    "H_donors": 1.0,
    "rotatable_bonds": 1.0,
    "vapor_pressure_mmhg": 0.01,
    "melting_point_c": 1.0,
    "pka_predicted": 0.1,
    "logD_pH6": 0.1,
    "max_dose_mg": 1.0,
}

CLINICAL_DEFAULTS = {
    "pka_predicted": (3.0, 10.0),
    "logD_pH6": (-1.0, 5.0),
    "max_dose_mg": (0.0, 100.0),
}

NOSA_OPTIMAL_PRESET: dict[str, tuple[float, float] | None] = {
    "molecular_weight": (100, 300),
    "logP": (1.0, 4.0),
    "PSA": (0, 100),
    "H_donors": (0, 3),
    "rotatable_bonds": (0, 7),
    "pka_predicted": (3, 10),
    "logD_pH6": (-1, 5),
    "melting_point_c": (0, 300),
    "max_dose_mg": (0, 100),
    "vapor_pressure_mmhg": (0, 100),
}

ATC_LEVEL1_NAMES = {
    "A": "Alimentary / metabolism",
    "B": "Blood / hematology",
    "C": "Cardiovascular",
    "D": "Dermatologicals",
    "G": "Genito-urinary / hormones",
    "H": "Systemic hormonal",
    "J": "Anti-infectives",
    "L": "Antineoplastic / immunomod",
    "M": "Musculoskeletal",
    "N": "Nervous system",
    "P": "Antiparasitic / insecticides",
    "R": "Respiratory",
    "S": "Sensory organs",
    "V": "Various",
}

ROUTE_LABELS = {
    "oral": "Oral",
    "parenteral": "Parenteral",
    "topical": "Topical",
}

DISPLAY_ROW_LIMIT = 500


def database_path() -> Path:
    """Prefer enriched CSV when it exists and has at least as many rows as master."""
    if not DATABASE_CSV.exists() and not ENRICHED_CSV.exists():
        return DATABASE_CSV
    if ENRICHED_CSV.exists():
        if not DATABASE_CSV.exists():
            return ENRICHED_CSV
        with ENRICHED_CSV.open("rb") as handle:
            enriched_rows = sum(1 for _ in handle) - 1
        with DATABASE_CSV.open("rb") as handle:
            master_rows = sum(1 for _ in handle) - 1
        if enriched_rows >= master_rows:
            return ENRICHED_CSV
    return DATABASE_CSV


def has_pubchem_data(df: pd.DataFrame) -> bool:
    return "vapor_pressure_mmhg" in df.columns and "melting_point_c" in df.columns


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {PRIMARY};
            color: {TEXT};
        }}
        [data-testid="stSidebar"] {{
            background-color: {SURFACE};
        }}
        [data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}
        h1, h2, h3, h4, p, label, span {{
            color: {TEXT} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {ACCENT} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT} !important;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {ACCENT};
            border-radius: 8px;
        }}
        .stDownloadButton button {{
            background-color: {ACCENT} !important;
            color: {PRIMARY} !important;
            border: none;
            font-weight: 600;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            background-color: {ACCENT} !important;
            color: {PRIMARY} !important;
            border: none;
            font-weight: 600;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background-color: {PRIMARY} !important;
            color: {TEXT} !important;
            border: none !important;
            font-weight: 600;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
            background-color: {SURFACE} !important;
            color: {TEXT} !important;
            border: 1px solid {ACCENT} !important;
            font-weight: 600;
        }}
        [data-testid="stNumberInput"] button {{
            color: {ACCENT} !important;
            border-color: {ACCENT} !important;
        }}
        [data-testid="stNumberInput"] input {{
            color: {TEXT} !important;
            background-color: {SURFACE} !important;
            border-color: {ACCENT} !important;
        }}
        .stSlider [data-baseweb="slider"] div {{
            color: {ACCENT} !important;
        }}
        .nosa-note {{
            color: {ACCENT} !important;
            font-size: 0.85rem;
            margin-top: -0.5rem;
        }}
        .nosa-warning {{
            color: #FFB84D !important;
            font-size: 0.85rem;
            margin-top: -0.5rem;
        }}
        .nosa-disabled {{
            color: {DISABLED} !important;
            font-size: 0.85rem;
        }}
        [data-testid="stPills"] button {{
            border-radius: 8px !important;
            font-weight: 600 !important;
        }}
        [data-testid="stPills"] [aria-pressed="true"] {{
            background-color: {ACCENT} !important;
            color: {PRIMARY} !important;
            border-color: {ACCENT} !important;
        }}
        [data-testid="stPills"] [aria-pressed="false"] {{
            background-color: {SURFACE} !important;
            color: {TEXT} !important;
            border-color: {DISABLED} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def has_clinical_data(df: pd.DataFrame) -> bool:
    return "pka_predicted" in df.columns and "logD_pH6" in df.columns


def _split_pipe_values(value: Any) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


@st.cache_data
def load_database(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, low_memory=False)
    numeric_cols = list(BASE_IDEAL.keys()) + PUBCHEM_COLUMNS + CLINICAL_COLUMNS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "nosa_candidate" in df.columns:
        df["nosa_candidate"] = df["nosa_candidate"].fillna(False).astype(bool)
    if "dose_feasible_nosa" in df.columns:
        df["dose_feasible_nosa"] = df["dose_feasible_nosa"].map(
            lambda v: True if v is True or str(v).lower() == "true"
            else False if v is False or str(v).lower() == "false"
            else None
        )
    if "max_dose_mg" not in df.columns and "max_clinical_dose_mg" in df.columns:
        df["max_dose_mg"] = df["max_clinical_dose_mg"]
        df["max_dose_source"] = df["max_clinical_dose_mg"].apply(
            lambda v: "chembl" if pd.notna(v) else None
        )
    if "cns_target" in df.columns:
        df["cns_target"] = df["cns_target"].astype("boolean").fillna(False).astype(bool)
    if "nasal_cyp_risk" in df.columns:
        df["nasal_cyp_risk"] = df["nasal_cyp_risk"].astype("boolean").fillna(False).astype(bool)
    if "pgp_substrate" in df.columns:
        df["pgp_substrate"] = df["pgp_substrate"].astype("boolean").fillna(False).astype(bool)
    if "big_pharma_owned" in df.columns:
        df["big_pharma_owned"] = df["big_pharma_owned"].astype("boolean").fillna(False).astype(bool)
    if "manual_score" in df.columns:
        df["manual_score"] = pd.to_numeric(df["manual_score"], errors="coerce").astype("Int64")
    if "manual_score_comment" in df.columns:
        df["manual_score_comment"] = df["manual_score_comment"].astype("string")
    if "manual_eval_set" in df.columns:
        df["manual_eval_set"] = df["manual_eval_set"].astype("boolean").fillna(False).astype(bool)
    if "route_of_administration" in df.columns:
        df["_route_tokens"] = df["route_of_administration"].map(
            lambda v: frozenset(_split_pipe_values(v))
        )
    if "atc_level1" in df.columns:
        df["_atc_tokens"] = df["atc_level1"].map(
            lambda v: frozenset(_split_pipe_values(v))
        )
    df = sanitize_dose_columns(df)
    if "max_clinical_dose_mg" in df.columns or "ct_max_dose_mg" in df.columns:
        df = df.apply(sync_unified_dose, axis=1)
    df = recompute_dose_feasible_nosa(df)
    return df


def active_score_config(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    ideal = dict(BASE_IDEAL)
    ranges = dict(BASE_SLIDER_RANGES)
    if has_pubchem_data(df):
        ideal.update(PUBCHEM_IDEAL)
        ranges.update(PUBCHEM_SLIDER_RANGES)
    if has_clinical_data(df):
        ideal.update(
            {
                "pka_predicted": 6.5,
                "logD_pH6": 2.5,
                "max_dose_mg": 50.0,
            }
        )
        ranges.update(CLINICAL_SLIDER_RANGES)
    return ideal, ranges


def property_score(series: pd.Series, ideal: float, span: float) -> pd.Series:
    distance = (series - ideal).abs()
    return (100.0 * (1.0 - distance / span)).clip(lower=0.0, upper=100.0).fillna(0.0)


def composite_score(
    df: pd.DataFrame,
    ideal: dict[str, float],
    ranges: dict[str, tuple[float, float]],
) -> pd.Series:
    score_cols = [c for c in ideal if c in df.columns and c != "max_dose_mg"]
    if not score_cols:
        total = pd.Series(0.0, index=df.index)
    else:
        scores = []
        for col in score_cols:
            lo, hi = ranges[col]
            scores.append(property_score(df[col], ideal[col], hi - lo))

        if "max_dose_mg" in df.columns:
            dose_score = df["max_dose_mg"].map(
                lambda dose: 100.0
                if pd.notna(dose) and dose <= 100
                else 0.0
                if pd.notna(dose)
                else 50.0
            )
            scores.append(dose_score)

        total = sum(scores) / len(scores)
        if "max_dose_mg" in df.columns:
            completeness_bonus = df["max_dose_mg"].notna().astype(float) * 3.0
            total = total + completeness_bonus

    if "cns_target" in df.columns:
        total = total + df["cns_target"].fillna(False).astype(float) * 5.0
    if "nasal_cyp_risk" in df.columns:
        total = total - df["nasal_cyp_risk"].fillna(False).astype(float) * 3.0
    if "pgp_substrate" in df.columns:
        total = total - df["pgp_substrate"].fillna(False).astype(float) * 2.0
    return total.clip(upper=100.0)


def apply_filters(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    default_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    pubchem_cols = set(PUBCHEM_COLUMNS)
    sparse_optional = {"max_dose_mg", "pka_predicted", "logD_pH6"}
    defaults = default_bounds or bounds
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        in_range = df[col].between(lo, hi)
        def_lo, def_hi = defaults.get(col, (lo, hi))
        narrowed = lo > def_lo or hi < def_hi
        if col in pubchem_cols or col in sparse_optional:
            if narrowed:
                mask &= df[col].notna() & in_range
            else:
                mask &= df[col].isna() | in_range
        else:
            mask &= df[col].notna() & in_range
    return df.loc[mask]


def apply_categorical_filters(
    df: pd.DataFrame,
    routes: list[str] | None,
    atc_levels: list[str] | None,
    all_routes: list[str],
    all_atc_levels: list[str],
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if routes is not None and len(routes) < len(all_routes) and "route_of_administration" in df.columns:
        selected = frozenset(routes)
        if "_route_tokens" in df.columns:
            mask &= df["_route_tokens"].map(lambda tokens: bool(tokens & selected))
        else:
            mask &= df["route_of_administration"].map(
                lambda v: bool(set(_split_pipe_values(v)) & selected)
            )

    if (
        atc_levels is not None
        and len(atc_levels) < len(all_atc_levels)
        and "atc_level1" in df.columns
    ):
        selected = frozenset(atc_levels)
        if "_atc_tokens" in df.columns:
            mask &= df["_atc_tokens"].map(lambda tokens: bool(tokens & selected))
        else:
            mask &= df["atc_level1"].map(
                lambda v: bool(set(_split_pipe_values(v)) & selected)
            )

    return df.loc[mask]


def apply_applicant_filters(
    df: pd.DataFrame,
    selected_companies: list[str],
    big_pharma_only: bool,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if selected_companies and "original_applicant_normalized" in df.columns:
        mask &= df["original_applicant_normalized"].isin(selected_companies)

    if big_pharma_only and "big_pharma_owned" in df.columns:
        mask &= df["big_pharma_owned"].fillna(False).astype(bool)

    return df.loc[mask]


def apply_metadata_filters(
    df: pd.DataFrame,
    hide_phases: set[int],
    hide_with_patent: bool,
    hide_without_patent: bool,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if hide_phases and "chembl_max_phase" in df.columns:
        phase = pd.to_numeric(df["chembl_max_phase"], errors="coerce")
        # Rows without a phase (e.g. manual volatiles) stay visible
        mask &= phase.isna() | ~phase.isin(hide_phases)

    if "earliest_patent_expiry" in df.columns:
        has_patent = df["earliest_patent_expiry"].notna()
        if hide_with_patent:
            mask &= ~has_patent
        if hide_without_patent:
            mask &= has_patent

    return df.loc[mask]


def phase_counts(df: pd.DataFrame) -> dict[int, int]:
    if "chembl_max_phase" not in df.columns:
        return {}
    phase = pd.to_numeric(df["chembl_max_phase"], errors="coerce")
    return {int(p): int((phase == p).sum()) for p in phase.dropna().unique()}


def render_metadata_filters(df: pd.DataFrame) -> tuple[set[int], bool, bool]:
    """Sidebar checkboxes — checked means hide that group."""
    st.subheader("Hide by status")
    st.caption("Check a box to hide matching drugs from results.")

    counts = phase_counts(df)
    no_phase = int(df["chembl_max_phase"].isna().sum()) if "chembl_max_phase" in df.columns else 0
    hide_phases: set[int] = set()
    for phase in (1, 2, 3, 4):
        n = counts.get(phase, 0)
        label = f"Hide phase {phase} ({n:,})" if n else f"Hide phase {phase} (0)"
        if st.checkbox(label, key=f"hide_phase_{phase}"):
            hide_phases.add(phase)
    if no_phase:
        st.caption(f"{no_phase:,} drugs have no phase (manual/other) — always shown")

    has_patent = int(df["earliest_patent_expiry"].notna().sum()) if "earliest_patent_expiry" in df.columns else 0
    no_patent = len(df) - has_patent
    hide_with_patent = st.checkbox(f"Hide drugs with patent ({has_patent:,})", key="hide_patent_yes")
    hide_without_patent = st.checkbox(
        f"Hide drugs without patent ({no_patent:,})", key="hide_patent_no"
    )

    return hide_phases, hide_with_patent, hide_without_patent


def render_applicant_filters(df: pd.DataFrame) -> tuple[list[str], bool]:
    if "original_applicant_normalized" not in df.columns:
        return [], False

    st.subheader("Patent holder")
    companies = sorted(df["original_applicant_normalized"].dropna().unique())
    if not companies:
        st.caption("Rebuild database to enable patent holder filters.")
        return [], False

    counts = df["original_applicant_normalized"].value_counts()
    options = [f"{company} ({counts.get(company, 0):,})" for company in companies]
    display_to_value = dict(zip(options, companies))

    selected_display = st.multiselect(
        "Filter by patent holder",
        options=options,
        default=[],
        key="filter_applicants",
        placeholder="All companies",
    )
    selected = [display_to_value[item] for item in selected_display if item in display_to_value]

    if selected:
        n = int(df["original_applicant_normalized"].isin(selected).sum())
        st.caption(f"{n:,} drugs from selected companies")
    else:
        st.caption("No companies selected — all patent holders shown")

    big_pharma_only = False
    if "big_pharma_owned" in df.columns:
        n_big = int(df["big_pharma_owned"].fillna(False).astype(bool).sum())
        big_pharma_only = st.checkbox(
            f"Show only big pharma-originated drugs ({n_big:,})",
            key="filter_big_pharma_only",
        )

    return selected, big_pharma_only


def has_drugbank_annotations(df: pd.DataFrame) -> bool:
    return "cns_target" in df.columns and "primary_target" in df.columns


def render_drugbank_filters(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    if not has_drugbank_annotations(df):
        return False, False, False

    st.subheader("DrugBank ADME filters")
    cns_count = int(df["cns_target"].fillna(False).astype(bool).sum())
    cyp_count = int(df["nasal_cyp_risk"].fillna(False).astype(bool).sum())
    pgp_count = int(df["pgp_substrate"].fillna(False).astype(bool).sum())

    cns_only = st.checkbox(
        f"Show only CNS-active drugs ({cns_count:,})",
        key="filter_cns_only",
    )
    hide_nasal_cyp = st.checkbox(
        f"Hide drugs at risk of nasal CYP metabolism ({cyp_count:,})",
        key="filter_hide_nasal_cyp",
    )
    hide_pgp = st.checkbox(
        f"Hide P-gp efflux substrates ({pgp_count:,})",
        key="filter_hide_pgp",
    )
    return cns_only, hide_nasal_cyp, hide_pgp


def apply_manual_score_filter(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter by manually curated evaluation. Blanks stay blank (never coerced to 0)."""
    if mode == "All" or "manual_score" not in df.columns:
        return df
    score = df["manual_score"]
    if mode == "Pass only (1)":
        return df.loc[score == 1]
    if mode == "No pass (0)":
        return df.loc[score == 0]
    if mode == "Not yet evaluated (blank)":
        # Prefer the curated evaluation batch when present; otherwise all NaN scores.
        if "manual_eval_set" in df.columns:
            in_set = df["manual_eval_set"].fillna(False).astype(bool)
            return df.loc[in_set & score.isna()]
        return df.loc[score.isna()]
    return df


def render_manual_score_filter(df: pd.DataFrame) -> str:
    st.subheader("Manual evaluation")
    if "manual_score" not in df.columns:
        st.caption("Run `python merge_manual_scores.py` to enable this filter.")
        return "All"

    score = df["manual_score"]
    n_pass = int((score == 1).sum())
    n_fail = int((score == 0).sum())
    if "manual_eval_set" in df.columns:
        n_blank = int((df["manual_eval_set"].fillna(False) & score.isna()).sum())
        n_eval = n_pass + n_fail
        n_set = int(df["manual_eval_set"].fillna(False).sum())
        st.caption(
            f"Eval set {n_set:,} · Evaluated {n_eval:,} · Pass {n_pass:,} · "
            f"No pass {n_fail:,} · Pending {n_blank:,}"
        )
    else:
        n_blank = int(score.isna().sum())
        n_eval = n_pass + n_fail
        st.caption(f"Evaluated {n_eval:,} · Pass {n_pass:,} · No pass {n_fail:,} · Pending {n_blank:,}")
    return st.radio(
        "Manual evaluation",
        options=list(MANUAL_SCORE_FILTER_OPTIONS),
        index=0,
        key="filter_manual_score",
        label_visibility="collapsed",
    )


def apply_drugbank_filters(
    df: pd.DataFrame,
    cns_only: bool,
    hide_nasal_cyp: bool,
    hide_pgp: bool,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if cns_only and "cns_target" in df.columns:
        mask &= df["cns_target"].fillna(False).astype(bool)
    if hide_nasal_cyp and "nasal_cyp_risk" in df.columns:
        mask &= ~df["nasal_cyp_risk"].fillna(False).astype(bool)
    if hide_pgp and "pgp_substrate" in df.columns:
        mask &= ~df["pgp_substrate"].fillna(False).astype(bool)
    return df.loc[mask]


def parse_field_sources(value: Any) -> dict[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def source_badge(field: str, field_sources: dict[str, str]) -> str:
    source = field_sources.get(field, "")
    if source == "drugbank":
        return "DB"
    if source == "clinicaltrials":
        return "CT"
    if source == "chembl":
        return "C"
    return ""


def dose_field_sources(row: pd.Series) -> dict[str, str]:
    sources = parse_field_sources(row.get("field_sources"))
    source = row.get("max_dose_source")
    if pd.notna(source) and str(source).strip():
        sources["max_dose_mg"] = str(source)
    return sources


def format_value_with_badge(
    value: Any,
    field: str,
    field_sources: dict[str, str],
    *,
    precision: int | None = None,
    conflict: bool = False,
    alt_value: Any = None,
    alt_label: str = "ChEMBL",
) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    text = f"{value:.{precision}f}" if precision is not None else str(value)
    badge = source_badge(field, field_sources)
    if badge:
        text = f"{text} [{badge}]"
    if conflict:
        alt_text = str(alt_value)[:80] + ("…" if alt_value and len(str(alt_value)) > 80 else "")
        text = f"⚠️ {text}"
        if alt_value is not None and not (isinstance(alt_value, float) and pd.isna(alt_value)):
            text = f"{text} ({alt_label}: {alt_text})"
    return text


def prepare_results_table(df: pd.DataFrame, limit: int = DISPLAY_ROW_LIMIT) -> tuple[pd.DataFrame, bool]:
    """Build display-ready string table (capped for performance)."""
    display_cols = [
        "name",
        "manual_score",
        "chembl_id",
        "chembl_max_phase",
        "composite_score",
        "manual_score_comment",
        "cns_target",
        "primary_target",
        "nasal_cyp_risk",
        "molecular_weight",
        "logP",
        "PSA",
        "H_donors",
        "rotatable_bonds",
        "vapor_pressure_mmhg",
        "melting_point_c",
        "pka_predicted",
        "logD_pH6",
        "logD_pH74",
        "max_dose_mg",
        "dose_feasible_nosa",
        "original_applicant_normalized",
        "earliest_patent_expiry",
        "atc_code",
        "mesh_heading",
        "mechanism_of_action",
        "target_name",
        "physical_state",
        "half_life",
        "dosage_form",
        "synonyms",
        "nosa_candidate",
        "indication_class",
        "earliest_patent_expiry",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    truncated = len(df) > limit
    work = df.head(limit).reset_index(drop=True)

    if work.empty:
        order = [c for c in display_cols if c != "nosa_candidate"]
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in order})
        empty["nosa_candidate"] = pd.Series(dtype="bool")
        return empty, truncated

    if "field_sources" in work.columns:
        fs_series = work["field_sources"].map(parse_field_sources)
    else:
        fs_series = pd.Series([{}] * len(work), index=work.index)

    def field_sources_for(row: pd.Series) -> dict[str, str]:
        fs = dict(fs_series.get(row.name, {}))
        fs.update({k: v for k, v in dose_field_sources(row).items() if k not in fs or k == "max_dose_mg"})
        return fs

    def mol_weight_cell(row: pd.Series) -> str:
        if pd.isna(row.get("molecular_weight")):
            return "—"
        return format_value_with_badge(
            round(float(row["molecular_weight"]), 2),
            "molecular_weight",
            field_sources_for(row),
            conflict=bool(row.get("mw_conflict")),
            alt_value=row.get("molecular_weight_db"),
            alt_label="DrugBank",
        )

    def numeric_badge_cell(row: pd.Series, col: str) -> str:
        if pd.isna(row.get(col)):
            return "—"
        return format_value_with_badge(row[col], col, field_sources_for(row), precision=2)

    def text_badge_cell(row: pd.Series, col: str) -> str:
        if pd.isna(row.get(col)):
            return "—"
        val = str(row[col])
        if len(val) > 50:
            val = val[:47] + "…"
        return format_value_with_badge(val, col, field_sources_for(row))

    def conflict_text_cell(row: pd.Series, col: str, conflict_col: str, alt_col: str) -> str:
        if pd.isna(row.get(col)):
            return "—"
        text = str(row[col])
        if len(text) > 60:
            text = text[:57] + "…"
        return format_value_with_badge(
            text,
            col,
            field_sources_for(row),
            conflict=bool(row.get(conflict_col)),
            alt_value=row.get(alt_col),
        )

    conflict_cols = {
        "mechanism_of_action": ("mechanism_of_action_conflict", "mechanism_of_action_chembl"),
        "indication_class": ("indication_class_conflict", "indication_class_chembl"),
    }

    columns: dict[str, list[str]] = {}
    for col in display_cols:
        if col == "nosa_candidate":
            continue
        if col == "composite_score":
            values = work[col].round(1).astype(str)
        elif col == "manual_score":
            values = work[col].map(
                lambda v: "1" if pd.notna(v) and int(v) == 1
                else "0" if pd.notna(v) and int(v) == 0
                else ""
            )
        elif col == "manual_score_comment":
            values = work[col].map(
                lambda v: ""
                if pd.isna(v) or str(v).strip() == ""
                else (str(v)[:77] + "…") if len(str(v)) > 80 else str(v)
            )
        elif col == "chembl_max_phase":
            values = work[col].apply(lambda v: str(int(v)) if pd.notna(v) else "—")
        elif col == "dose_feasible_nosa":
            values = work[col].map(lambda v: "✓" if v is True else "✗" if v is False else "?")
        elif col == "cns_target":
            values = work[col].map(lambda v: "🧠" if v is True else "")
        elif col == "nasal_cyp_risk":
            values = work[col].map(lambda v: "⚠️" if v is True else "")
        elif col in ("PSA", "vapor_pressure_mmhg", "melting_point_c"):
            values = work[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        elif col == "primary_target":
            values = work[col].apply(
                lambda v: (str(v)[:47] + "…") if pd.notna(v) and len(str(v)) > 50 else (str(v) if pd.notna(v) else "—")
            )
        elif col == "molecular_weight":
            values = work.apply(mol_weight_cell, axis=1)
        elif col in ("logP", "pka_predicted", "logD_pH6", "logD_pH74", "max_dose_mg"):
            values = work.apply(lambda row, c=col: numeric_badge_cell(row, c), axis=1)
        elif col in ("physical_state", "half_life"):
            values = work.apply(lambda row, c=col: text_badge_cell(row, c), axis=1)
        elif col in conflict_cols:
            cc, ac = conflict_cols[col]
            values = work.apply(lambda row, c=col, cc=cc, ac=ac: conflict_text_cell(row, c, cc, ac), axis=1)
        else:
            values = work[col].apply(lambda v: str(v) if pd.notna(v) else "—")
        columns[col] = list(values)

    display_order = [c for c in display_cols if c != "nosa_candidate" and c in columns]
    shown = pd.DataFrame({c: columns[c] for c in display_order}, index=work.index, dtype="object")
    shown["nosa_candidate"] = (
        work["nosa_candidate"].astype(bool) if "nosa_candidate" in work.columns else False
    )
    return shown, truncated


def style_results_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    shown, _ = prepare_results_table(df)
    nosa = shown.pop("nosa_candidate").astype(bool)

    def highlight_nosa(row: pd.Series) -> list[str]:
        if nosa.iloc[row.name]:
            return [f"background-color: {ACCENT}; color: {PRIMARY}; font-weight: 600"] * len(row)
        return [f"background-color: {SURFACE}; color: {TEXT}"] * len(row)

    def badge_style(row: pd.Series) -> list[str]:
        base = highlight_nosa(row)
        if "dose_feasible_nosa" in row.index:
            badge = row["dose_feasible_nosa"]
            dose_idx = list(row.index).index("dose_feasible_nosa")
            if badge == "✓":
                base[dose_idx] = f"background-color: #2E8B57; color: {TEXT}; font-weight: 700"
            elif badge == "✗":
                base[dose_idx] = f"background-color: #C0392B; color: {TEXT}; font-weight: 700"
            elif badge == "?":
                base[dose_idx] = f"background-color: {DISABLED}; color: {TEXT}; font-weight: 700"
        if "manual_score" in row.index:
            score = row["manual_score"]
            score_idx = list(row.index).index("manual_score")
            if score == "1":
                base[score_idx] = (
                    f"background-color: {MANUAL_PASS_COLOR}; color: {PRIMARY}; font-weight: 700"
                )
            elif score == "0":
                base[score_idx] = (
                    f"background-color: {MANUAL_FAIL_COLOR}; color: {TEXT}; font-weight: 700"
                )
        return base

    return shown.style.apply(badge_style, axis=1)


def render_drug_detail_panel(filtered: pd.DataFrame) -> None:
    if filtered.empty:
        return

    with st.expander("Drug detail & data sources", expanded=False):
        st.caption("C = ChEMBL · DB = DrugBank · ⚠️ = conflicting values between sources")
        names = filtered["name"].tolist()
        selected = st.selectbox("Select drug", names, key="detail_drug_select")
        row = filtered.loc[filtered["name"] == selected].iloc[0]
        sources = parse_field_sources(row.get("field_sources"))

        st.markdown(f"**{selected}**")
        if row.get("drugbank_id"):
            st.markdown(f"DrugBank ID: `{row['drugbank_id']}`")
        if row.get("chembl_id"):
            st.markdown(f"ChEMBL ID: `{row['chembl_id']}`")

        if "manual_score" in row.index:
            score = row.get("manual_score")
            if pd.isna(score):
                score_label = "Not yet evaluated"
            elif int(score) == 1:
                score_label = "Pass (1)"
            else:
                score_label = "No pass (0)"
            st.markdown(f"**Manual evaluation:** {score_label}")
            comment = row.get("manual_score_comment")
            if pd.notna(comment) and str(comment).strip():
                st.markdown(f"**Score comment:** {comment}")

        st.markdown("**Data sources**")
        if sources:
            source_rows = [{"Field": k, "Source": v.upper() if v == "drugbank" else v.title()} for k, v in sorted(sources.items())]
            st.dataframe(pd.DataFrame(source_rows), width="stretch", hide_index=True)
        else:
            st.info("No source mapping recorded for this compound.")

        conflicts: list[str] = []
        if row.get("mechanism_of_action_conflict"):
            conflicts.append(
                f"**Mechanism** — shown: DrugBank · alternate ChEMBL: {row.get('mechanism_of_action_chembl', '—')}"
            )
        if row.get("indication_class_conflict"):
            conflicts.append(
                f"**Indication** — shown: DrugBank · alternate ChEMBL: {row.get('indication_class_chembl', '—')}"
            )
        if row.get("mw_conflict"):
            conflicts.append(
                f"**Molecular weight** — ChEMBL: {row.get('molecular_weight', '—')} Da · "
                f"DrugBank: {row.get('molecular_weight_db', '—')} Da"
            )
        if conflicts:
            st.markdown("**⚠️ Source conflicts**")
            for note in conflicts:
                st.markdown(note)

        dose_lines: list[str] = []
        if pd.notna(row.get("max_dose_mg")):
            src = row.get("max_dose_source", "—")
            src_label = "ChEMBL" if src == "chembl" else "ClinicalTrials.gov" if src == "clinicaltrials" else str(src)
            dose_lines.append(f"**Unified max dose:** {row['max_dose_mg']:.1f} mg ({src_label})")
        if pd.notna(row.get("max_clinical_dose_mg")):
            dose_lines.append(f"ChEMBL max dose: {row['max_clinical_dose_mg']:.1f} mg")
        if pd.notna(row.get("ct_max_dose_mg")):
            dose_lines.append(f"ClinicalTrials.gov max dose: {row['ct_max_dose_mg']:.1f} mg")
        if pd.notna(row.get("ct_n_studies")):
            dose_lines.append(f"ClinicalTrials.gov studies: {int(row['ct_n_studies'])}")
        if pd.notna(row.get("ct_max_dose_raw")):
            dose_lines.append(f"CT dose snippet: _{row['ct_max_dose_raw']}_")
        if dose_lines:
            st.markdown("**Dose data**")
            for line in dose_lines:
                st.markdown(line)

        ownership_lines: list[str] = []
        if pd.notna(row.get("original_applicant_normalized")):
            ownership_lines.append(f"**Original holder:** {row['original_applicant_normalized']}")
        if pd.notna(row.get("original_applicant")) and row.get("original_applicant") != row.get(
            "original_applicant_normalized"
        ):
            ownership_lines.append(f"**Original applicant (Orange Book):** {row['original_applicant']}")
        if pd.notna(row.get("all_applicants")):
            ownership_lines.append(
                f"**All applicants:** {str(row['all_applicants']).replace('|', ' · ')}"
            )
        if pd.notna(row.get("applicant_count")):
            ownership_lines.append(f"**Applicant count:** {int(row['applicant_count'])}")
        if pd.notna(row.get("earliest_patent_expiry")):
            ownership_lines.append(f"**Earliest patent expiry:** {row['earliest_patent_expiry']}")
        if ownership_lines:
            st.markdown("**Patent & ownership**")
            for line in ownership_lines:
                st.markdown(line)

        if pd.notna(row.get("target_names")):
            st.markdown("**Targets (DrugBank)**")
            st.markdown(str(row["target_names"]).replace("|", " · "))
        if pd.notna(row.get("metabolizing_enzymes")) or pd.notna(row.get("cyp_enzymes")):
            st.markdown("**Metabolizing enzymes**")
            enzymes = str(row.get("metabolizing_enzymes") or "")
            if pd.notna(row.get("cyp_enzymes")):
                cyp_set = set(_split_pipe_values(row.get("cyp_enzymes")))
                parts = []
                for part in _split_pipe_values(enzymes):
                    if part in cyp_set or "CYP" in part.upper() or "Cytochrome" in part:
                        parts.append(f"**{part}**")
                    else:
                        parts.append(part)
                st.markdown(" · ".join(parts) if parts else enzymes)
            else:
                st.markdown(enzymes.replace("|", " · "))
        if pd.notna(row.get("transporters")):
            st.markdown("**Transporters**")
            trans_parts = []
            for part in _split_pipe_values(row.get("transporters")):
                if re.search(r"P-glycoprotein|P-gp|ABCB1|MDR1", part, re.IGNORECASE):
                    trans_parts.append(f"**{part}** (P-gp)")
                else:
                    trans_parts.append(part)
            st.markdown(" · ".join(trans_parts))

        pk_fields = [
            "absorption",
            "half_life",
            "protein_binding",
            "metabolism",
            "route_of_elimination",
            "volume_of_distribution",
            "clearance",
            "toxicity",
        ]
        pk_data = {f: row.get(f) for f in pk_fields if f in row.index and pd.notna(row.get(f))}
        if pk_data:
            st.markdown("**Pharmacokinetics (DrugBank)**")
            st.json(pk_data)


def build_scatter(df: pd.DataFrame, ideal: dict[str, float]) -> go.Figure:
    plot_df = df.dropna(subset=["logP", "molecular_weight"]).copy()
    if "manual_score" in plot_df.columns:
        plot_df["series"] = plot_df.apply(
            lambda row: "Manual pass (1)"
            if pd.notna(row.get("manual_score")) and int(row["manual_score"]) == 1
            else ("NOSA candidate" if bool(row.get("nosa_candidate")) else "Other"),
            axis=1,
        )
    else:
        plot_df["series"] = plot_df["nosa_candidate"].map(
            {True: "NOSA candidate", False: "Other"}
        )

    fig = px.scatter(
        plot_df,
        x="molecular_weight",
        y="logP",
        color="series",
        symbol="series",
        hover_name="name",
        hover_data={
            "composite_score": ":.1f",
            "PSA": ":.1f",
            "molecular_weight": ":.1f",
            "logP": ":.2f",
            "series": False,
        },
        color_discrete_map={
            "Manual pass (1)": MANUAL_PASS_COLOR,
            "NOSA candidate": ACCENT,
            "Other": "rgba(255,255,255,0.45)",
        },
        symbol_map={
            "Manual pass (1)": "triangle-up",
            "NOSA candidate": "star",
            "Other": "circle",
        },
        labels={
            "molecular_weight": "Molecular weight (Da)",
            "logP": "logP",
        },
    )

    fig.add_trace(
        go.Scatter(
            x=[ideal["molecular_weight"]],
            y=[ideal["logP"]],
            mode="markers+text",
            name="Memantine (ideal)",
            marker=dict(color=MEMANTINE_COLOR, size=16, symbol="diamond"),
            text=["Memantine"],
            textposition="top center",
            textfont=dict(color=TEXT, size=11),
            hovertemplate=(
                "Memantine benchmark<br>MW: %{x:.1f} Da<br>logP: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=PRIMARY,
        plot_bgcolor=SURFACE,
        font_color=TEXT,
        legend=dict(bgcolor=SURFACE, bordercolor=ACCENT),
        xaxis=dict(gridcolor="rgba(0,191,191,0.25)", zerolinecolor=ACCENT),
        yaxis=dict(gridcolor="rgba(0,191,191,0.25)", zerolinecolor=ACCENT),
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _init_filter_bounds(col: str, default: tuple[float, float]) -> None:
    lo_key = f"filter_{col}_lo"
    hi_key = f"filter_{col}_hi"
    if lo_key not in st.session_state:
        st.session_state[lo_key] = float(default[0])
    if hi_key not in st.session_state:
        st.session_state[hi_key] = float(default[1])


def render_range_filter(
    col: str,
    label: str,
    lo_bound: float,
    hi_bound: float,
    default: tuple[float, float],
    step: float,
    *,
    disabled: bool = False,
) -> tuple[float, float]:
    """Range filter with keyboard min/max inputs and a synced slider."""
    lo_bound = float(lo_bound)
    hi_bound = float(hi_bound)
    default = (float(default[0]), float(default[1]))
    step = float(step)

    _init_filter_bounds(col, default)
    lo_key = f"filter_{col}_lo"
    hi_key = f"filter_{col}_hi"
    slider_key = f"filter_{col}_slider"
    fmt = "%.2f" if step < 1 else "%.0f"

    st.markdown(f"**{label}**")
    c1, c2 = st.columns(2)
    c1.number_input(
        "Min",
        min_value=lo_bound,
        max_value=hi_bound,
        step=step,
        format=fmt,
        key=lo_key,
        disabled=disabled,
    )
    c2.number_input(
        "Max",
        min_value=lo_bound,
        max_value=hi_bound,
        step=step,
        format=fmt,
        key=hi_key,
        disabled=disabled,
    )

    lo = float(min(st.session_state[lo_key], st.session_state[hi_key]))
    hi = float(max(st.session_state[lo_key], st.session_state[hi_key]))

    if not disabled:
        if slider_key not in st.session_state:
            st.session_state[slider_key] = (lo, hi)
        elif (lo, hi) != tuple(st.session_state[slider_key]):
            # Number inputs changed — update slider state before the widget is drawn
            st.session_state[slider_key] = (lo, hi)

        def _sync_slider() -> None:
            s_lo, s_hi = st.session_state[slider_key]
            st.session_state[lo_key] = float(s_lo)
            st.session_state[hi_key] = float(s_hi)

        st.slider(
            label,
            lo_bound,
            hi_bound,
            step=step,
            label_visibility="collapsed",
            key=slider_key,
            on_change=_sync_slider,
        )

    return (
        float(min(st.session_state[lo_key], st.session_state[hi_key])),
        float(max(st.session_state[lo_key], st.session_state[hi_key])),
    )


def optimal_filter_defaults(slider_ranges: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Default bounds for sparse-optional filter logic (NOSA optimal preset)."""
    defaults = dict(slider_ranges)
    for col, preset in NOSA_OPTIMAL_PRESET.items():
        if preset is not None:
            defaults[col] = preset
    return defaults


def filter_default(col: str, slider_ranges: dict[str, tuple[float, float]]) -> tuple[float, float]:
    preset = NOSA_OPTIMAL_PRESET.get(col)
    if preset is not None:
        return preset
    return slider_ranges[col]


def set_slider_bounds(col: str, lo: float, hi: float) -> None:
    st.session_state[f"filter_{col}_lo"] = float(lo)
    st.session_state[f"filter_{col}_hi"] = float(hi)
    st.session_state[f"filter_{col}_slider"] = (float(lo), float(hi))


def apply_nosa_optimal_filters(slider_ranges: dict[str, tuple[float, float]]) -> None:
    """Apply NOSA optimal numeric slider preset (categorical toggles unchanged)."""
    for col, (lo, hi) in optimal_filter_defaults(slider_ranges).items():
        set_slider_bounds(col, lo, hi)


def reset_sliders_to_full_range(slider_ranges: dict[str, tuple[float, float]]) -> None:
    """Return all numeric sliders to their widest setting."""
    for col, (lo, hi) in slider_ranges.items():
        set_slider_bounds(col, lo, hi)


def reset_all_filters(slider_ranges: dict[str, tuple[float, float]]) -> None:
    reset_sliders_to_full_range(slider_ranges)
    for phase in (1, 2, 3, 4):
        st.session_state[f"hide_phase_{phase}"] = False
    st.session_state["hide_patent_yes"] = False
    st.session_state["hide_patent_no"] = False
    st.session_state["filter_cns_only"] = False
    st.session_state["filter_hide_nasal_cyp"] = False
    st.session_state["filter_hide_pgp"] = False
    st.session_state["filter_applicants"] = []
    st.session_state["filter_big_pharma_only"] = False
    for key in ("filter_routes_pills", "filter_atc_pills"):
        if key in st.session_state:
            del st.session_state[key]


def render_pubchem_filters(
    pubchem_available: bool,
    slider_ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    st.subheader("Platform feasibility")
    if not pubchem_available:
        st.markdown(
            '<p class="nosa-disabled">PubChem properties unavailable — run '
            "<code>python enrich_pubchem.py</code> to enable.</p>",
            unsafe_allow_html=True,
        )
        for col in PUBCHEM_SLIDER_RANGES:
            render_range_filter(
                col,
                FILTER_LABELS[col],
                *slider_ranges[col],
                filter_default(col, slider_ranges),
                FILTER_STEPS[col],
                disabled=True,
            )
        return bounds

    bounds["vapor_pressure_mmhg"] = render_range_filter(
        "vapor_pressure_mmhg",
        FILTER_LABELS["vapor_pressure_mmhg"],
        *slider_ranges["vapor_pressure_mmhg"],
        filter_default("vapor_pressure_mmhg", slider_ranges),
        FILTER_STEPS["vapor_pressure_mmhg"],
    )
    st.markdown(
        '<p class="nosa-note">Higher = more volatile = releases better from polymer</p>',
        unsafe_allow_html=True,
    )
    bounds["melting_point_c"] = render_range_filter(
        "melting_point_c",
        FILTER_LABELS["melting_point_c"],
        *slider_ranges["melting_point_c"],
        filter_default("melting_point_c", slider_ranges),
        FILTER_STEPS["melting_point_c"],
    )
    st.markdown(
        '<p class="nosa-note">Narrowing VP or MP excludes drugs missing that property.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="nosa-warning">Injection molding reaches ~150–250°C — drugs '
        "melting below 150°C are thermal stability risks</p>",
        unsafe_allow_html=True,
    )
    return bounds


def unique_pipe_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    for item in series.dropna():
        values.extend(_split_pipe_values(item))
    return sorted(set(values))


def token_counts(df: pd.DataFrame, token_col: str, options: list[str]) -> dict[str, int]:
    counts = {opt: 0 for opt in options}
    if token_col not in df.columns:
        return counts
    for opt in options:
        counts[opt] = int(df[token_col].map(lambda tokens: opt in tokens).sum())
    return counts


def render_pills_filter(
    label: str,
    options: list[str],
    counts: dict[str, int],
    labels: dict[str, str],
    session_key: str,
) -> list[str]:
    """Multi-select pill toggles with drug counts."""
    if not options:
        return []

    st.markdown(f"**{label}**")
    format_opts = [f"{labels.get(opt, opt)} ({counts.get(opt, 0):,})" for opt in options]
    display_to_value = dict(zip(format_opts, options))
    pills_key = f"{session_key}_pills"

    if pills_key not in st.session_state:
        st.session_state[pills_key] = format_opts

    selected_display = st.pills(
        label,
        options=format_opts,
        selection_mode="multi",
        key=pills_key,
        label_visibility="collapsed",
    )
    if selected_display is None:
        selected_display = []

    selected = [display_to_value[item] for item in selected_display if item in display_to_value]

    n_sel = len(selected)
    n_all = len(options)
    if n_sel == n_all:
        st.caption(f"All {n_all} selected")
    elif n_sel == 0:
        st.caption("None selected — no drugs will match")
    else:
        st.caption(f"{n_sel} of {n_all} selected")

    btn1, btn2 = st.columns(2)
    if btn1.button("Select all", key=f"{session_key}_all", use_container_width=True):
        st.session_state[pills_key] = format_opts
        st.rerun()
    if btn2.button("Clear all", key=f"{session_key}_clear", use_container_width=True):
        st.session_state[pills_key] = []
        st.rerun()

    return selected


def render_clinical_filters(
    df: pd.DataFrame,
    slider_ranges: dict[str, tuple[float, float]],
) -> tuple[dict[str, tuple[float, float]], list[str], list[str]]:
    bounds: dict[str, tuple[float, float]] = {}
    if not has_clinical_data(df):
        st.subheader("Clinical / ADME filters")
        st.markdown(
            '<p class="nosa-disabled">Rebuild database to enable pKa, logD, and dose filters.</p>',
            unsafe_allow_html=True,
        )
        return bounds, [], []

    st.subheader("Clinical / ADME filters")
    for col in CLINICAL_COLUMNS:
        bounds[col] = render_range_filter(
            col,
            FILTER_LABELS[col],
            *slider_ranges[col],
            filter_default(col, slider_ranges),
            FILTER_STEPS[col],
        )

    st.markdown("**Route & therapeutic area**")
    st.caption("Click pills to toggle. Drugs missing route/ATC data are hidden when filtering.")

    all_routes = unique_pipe_values(df["route_of_administration"]) if "route_of_administration" in df.columns else []
    all_atc = unique_pipe_values(df["atc_level1"]) if "atc_level1" in df.columns else []

    selected_routes = all_routes
    selected_atc = all_atc
    if all_routes:
        route_counts = token_counts(df, "_route_tokens", all_routes)
        selected_routes = render_pills_filter(
            "Route of administration",
            all_routes,
            route_counts,
            ROUTE_LABELS,
            "filter_routes",
        )
    if all_atc:
        atc_counts = token_counts(df, "_atc_tokens", all_atc)
        selected_atc = render_pills_filter(
            "ATC level 1 (therapeutic area)",
            all_atc,
            atc_counts,
            ATC_LEVEL1_NAMES,
            "filter_atc",
        )

    return bounds, selected_routes, selected_atc


def benchmark_text(ideal: dict[str, float], n_criteria: int) -> str:
    parts = [
        f"MW {ideal['molecular_weight']:.1f} Da",
        f"logP {ideal['logP']:.2f}",
        f"PSA {ideal['PSA']:.1f} Å²",
        f"H-donors {ideal['H_donors']:.0f}",
        f"Rot. bonds {ideal['rotatable_bonds']:.0f}",
    ]
    if "vapor_pressure_mmhg" in ideal:
        parts.append(f"VP {ideal['vapor_pressure_mmhg']:.2f} mmHg")
    if "melting_point_c" in ideal:
        parts.append(f"MP {ideal['melting_point_c']:.0f} °C")
    if "pka_predicted" in ideal:
        parts.append(f"pKa {ideal['pka_predicted']:.1f}")
    if "logD_pH6" in ideal:
        parts.append(f"logD₆ {ideal['logD_pH6']:.1f}")
    if "max_dose_mg" in ideal:
        parts.append(f"Dose {ideal['max_dose_mg']:.0f} mg")
    return " · ".join(parts) + f" ({n_criteria} criteria)"


def main() -> None:
    st.set_page_config(
        page_title="NOSA Drug Screener",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    path = database_path()
    df = load_database(str(path))
    pubchem_available = has_pubchem_data(df)
    ideal, slider_ranges = active_score_config(df)
    default_bounds = optimal_filter_defaults(slider_ranges)
    n_criteria = len(ideal) + (1 if "max_dose_mg" in df.columns else 0)

    st.title("NOSA Drug Screener")
    st.caption(
        f"Screen FDA-approved small molecules for nasal vapor delivery. "
        f"Composite score ranks similarity to memantine across {n_criteria} "
        f"physicochemical properties. "
        f"Data source: `{path.name}`"
    )

    total_drugs = len(df)

    with st.sidebar:
        st.header("Property filters")
        st.caption(
            "Optimal preset keeps memantine and all validated candidates in view while narrowing the field."
        )
        btn_opt, btn_full = st.columns(2)
        if btn_opt.button("🎯 Apply NOSA Optimal Filters", key="apply_nosa_optimal", type="primary"):
            apply_nosa_optimal_filters(slider_ranges)
            st.rerun()
        if btn_full.button("↺ Reset to Full Range", key="reset_full_range", type="secondary"):
            reset_sliders_to_full_range(slider_ranges)
            st.rerun()
        st.caption("Type min/max values or drag the slider.")
        bounds: dict[str, tuple[float, float]] = {}
        for col in BASE_SLIDER_RANGES:
            bounds[col] = render_range_filter(
                col,
                FILTER_LABELS[col],
                *slider_ranges[col],
                filter_default(col, slider_ranges),
                FILTER_STEPS[col],
            )

        st.divider()
        bounds.update(render_pubchem_filters(pubchem_available, slider_ranges))

        st.divider()
        clinical_bounds, selected_routes, selected_atc = render_clinical_filters(df, slider_ranges)
        bounds.update(clinical_bounds)
        all_routes = unique_pipe_values(df["route_of_administration"]) if "route_of_administration" in df.columns else []
        all_atc = unique_pipe_values(df["atc_level1"]) if "atc_level1" in df.columns else []

        st.divider()
        hide_phases, hide_with_patent, hide_without_patent = render_metadata_filters(df)

        st.divider()
        selected_applicants, big_pharma_only = render_applicant_filters(df)

        st.divider()
        manual_score_mode = render_manual_score_filter(df)

        st.divider()
        cns_only, hide_nasal_cyp, hide_pgp = render_drugbank_filters(df)

        st.divider()
        st.markdown("**Memantine benchmark**")
        st.markdown(benchmark_text(ideal, n_criteria))

        preview = apply_filters(df, bounds, default_bounds)
        preview = apply_categorical_filters(
            preview, selected_routes, selected_atc, all_routes, all_atc
        )
        preview = apply_metadata_filters(
            preview, hide_phases, hide_with_patent, hide_without_patent
        )
        preview = apply_applicant_filters(preview, selected_applicants, big_pharma_only)
        preview = apply_manual_score_filter(preview, manual_score_mode)
        preview = apply_drugbank_filters(preview, cns_only, hide_nasal_cyp, hide_pgp)
        preview_nosa = int(preview["nosa_candidate"].fillna(False).sum()) if len(preview) else 0
        st.caption(
            f"**{len(preview):,}** drugs match current filters · "
            f"**{preview_nosa}** NOSA candidates in view"
        )

    filtered = apply_filters(df, bounds, default_bounds)
    filtered = apply_categorical_filters(
        filtered, selected_routes, selected_atc, all_routes, all_atc
    )
    filtered = apply_metadata_filters(
        filtered, hide_phases, hide_with_patent, hide_without_patent
    )
    filtered = apply_applicant_filters(filtered, selected_applicants, big_pharma_only)
    filtered = apply_manual_score_filter(filtered, manual_score_mode)
    filtered = apply_drugbank_filters(filtered, cns_only, hide_nasal_cyp, hide_pgp)
    filtered["composite_score"] = composite_score(filtered, ideal, slider_ranges)
    filtered = filtered.sort_values("composite_score", ascending=False)

    passing = len(filtered)
    nosa_in_view = int(filtered["nosa_candidate"].fillna(False).sum())

    if "manual_score" in df.columns:
        eval_count = int(df["manual_score"].notna().sum())
        pass_count = int((df["manual_score"] == 1).sum())
        if "manual_eval_set" in df.columns:
            pending_count = int(
                (df["manual_eval_set"].fillna(False) & df["manual_score"].isna()).sum()
            )
        else:
            pending_count = int(df["manual_score"].isna().sum())
    else:
        eval_count = pass_count = pending_count = 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total in database", f"{total_drugs:,}")
    m2.metric("Evaluated", f"{eval_count:,}")
    m3.metric("Pass (1)", f"{pass_count:,}")
    m4.metric("Pending", f"{pending_count:,}")

    f1, f2, f3 = st.columns(3)
    f1.metric("Drugs passing filters", f"{passing:,}")
    f2.metric("NOSA candidates in view", nosa_in_view)
    f3.metric(
        "Manual pass in view",
        int((filtered["manual_score"] == 1).sum()) if "manual_score" in filtered.columns else 0,
    )

    col_table, col_chart = st.columns([1.2, 1])

    with col_table:
        st.subheader("Ranked results")
        if passing == 0:
            st.warning("No drugs match the current filters. Widen the sliders.")
        else:
            if passing > DISPLAY_ROW_LIMIT:
                st.caption(
                    f"Showing top {DISPLAY_ROW_LIMIT:,} of {passing:,} results "
                    f"(download CSV for the full filtered list)."
                )
            st.dataframe(
                style_results_table(filtered),
                width="stretch",
                height=480,
                column_config={
                    "manual_score": st.column_config.TextColumn(
                        "Score",
                        help="1 = pass · 0 = no pass · blank = not yet evaluated",
                        width="small",
                    ),
                    "manual_score_comment": st.column_config.TextColumn(
                        "Score comment",
                        help="Full comment is in Drug detail below",
                        width="large",
                    ),
                },
            )
        render_drug_detail_panel(filtered)

    with col_chart:
        st.subheader("logP vs molecular weight")
        if passing > 0:
            st.plotly_chart(build_scatter(filtered, ideal), width="stretch")
        else:
            st.info("Scatter plot appears when at least one drug passes filters.")

    st.divider()
    export_df = filtered.drop(columns=["name_key"], errors="ignore")
    st.download_button(
        label="Download filtered results (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="nosa_screening_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
