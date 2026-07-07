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

ROOT = Path(__file__).resolve().parent
DATABASE_CSV = ROOT / "nosa_drug_database.csv"
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"

PRIMARY = "#1E6464"
ACCENT = "#00BFBF"
TEXT = "#FFFFFF"
SURFACE = "#174F4F"
DISABLED = "#3A6B6B"
MEMANTINE_COLOR = "#FFD700"

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
    "pka_predicted": "pKa (predicted) — Stanko filter: 3–10",
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
    "logD_pH6": (1.0, 4.0),
    "max_dose_mg": (0.0, 100.0),
}


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
        </style>
        """,
        unsafe_allow_html=True,
    )


def has_clinical_data(df: pd.DataFrame) -> bool:
    return "pka_predicted" in df.columns and "logD_pH6" in df.columns


@st.cache_data
def load_database(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
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
    return df.loc[mask].copy()


def _split_pipe_values(value: Any) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def apply_categorical_filters(
    df: pd.DataFrame,
    routes: list[str] | None,
    atc_levels: list[str] | None,
    all_routes: list[str],
    all_atc_levels: list[str],
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if routes is not None and len(routes) < len(all_routes) and "route_of_administration" in df.columns:
        selected = set(routes)

        def route_match(value: Any) -> bool:
            parts = _split_pipe_values(value)
            return bool(parts) and any(part in selected for part in parts)

        mask &= df["route_of_administration"].map(route_match)

    if (
        atc_levels is not None
        and len(atc_levels) < len(all_atc_levels)
        and "atc_level1" in df.columns
    ):
        selected = set(atc_levels)

        def atc_match(value: Any) -> bool:
            parts = _split_pipe_values(value)
            return bool(parts) and any(part in selected for part in parts)

        mask &= df["atc_level1"].map(atc_match)

    return df.loc[mask].copy()


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

    return df.loc[mask].copy()


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
    return df.loc[mask].copy()


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


def style_results_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def highlight_nosa(row: pd.Series) -> list[str]:
        if row.get("nosa_candidate"):
            return [f"background-color: {ACCENT}; color: {PRIMARY}; font-weight: 600"] * len(
                row
            )
        return [f"background-color: {SURFACE}; color: {TEXT}"] * len(row)

    display_cols = [
        "name",
        "chembl_id",
        "chembl_max_phase",
        "composite_score",
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
    shown = df[display_cols].copy()
    shown["composite_score"] = shown["composite_score"].round(1)

    field_sources_list = df["field_sources"].map(parse_field_sources) if "field_sources" in df.columns else [{}] * len(df)
    field_sources_list = list(field_sources_list)

    if "chembl_max_phase" in shown.columns:
        shown["chembl_max_phase"] = shown["chembl_max_phase"].apply(
            lambda v: int(v) if pd.notna(v) else "—"
        )
    if "dose_feasible_nosa" in shown.columns:
        shown["dose_feasible_nosa"] = shown["dose_feasible_nosa"].map(
            lambda v: "✓" if v is True else "✗" if v is False else "?"
        )
    if "cns_target" in shown.columns:
        shown["cns_target"] = shown["cns_target"].map(
            lambda v: "🧠" if v is True else ""
        )
    if "nasal_cyp_risk" in shown.columns:
        shown["nasal_cyp_risk"] = shown["nasal_cyp_risk"].map(
            lambda v: "⚠️" if v is True else ""
        )

    for idx, row in df.iterrows():
        pos = df.index.get_loc(idx)
        fs = field_sources_list[pos] if pos < len(field_sources_list) else {}
        fs = {**fs, **{k: v for k, v in dose_field_sources(row).items() if k not in fs or k == "max_dose_mg"}}
        if "molecular_weight" in shown.columns and pd.notna(row.get("molecular_weight")):
            shown.at[idx, "molecular_weight"] = format_value_with_badge(
                round(float(row["molecular_weight"]), 2),
                "molecular_weight",
                fs,
                conflict=bool(row.get("mw_conflict")),
                alt_value=row.get("molecular_weight_db"),
                alt_label="DrugBank",
            )
        if "logP" in shown.columns and pd.notna(row.get("logP")):
            shown.at[idx, "logP"] = format_value_with_badge(row["logP"], "logP", fs, precision=2)
        if "mechanism_of_action" in shown.columns and pd.notna(row.get("mechanism_of_action")):
            mech = str(row["mechanism_of_action"])
            if len(mech) > 60:
                mech = mech[:57] + "…"
            shown.at[idx, "mechanism_of_action"] = format_value_with_badge(
                mech,
                "mechanism_of_action",
                fs,
                conflict=bool(row.get("mechanism_of_action_conflict")),
                alt_value=row.get("mechanism_of_action_chembl"),
            )
        if "indication_class" in shown.columns and pd.notna(row.get("indication_class")):
            ind = str(row["indication_class"])
            if len(ind) > 60:
                ind = ind[:57] + "…"
            shown.at[idx, "indication_class"] = format_value_with_badge(
                ind,
                "indication_class",
                fs,
                conflict=bool(row.get("indication_class_conflict")),
                alt_value=row.get("indication_class_chembl"),
            )
        if "primary_target" in shown.columns and pd.notna(row.get("primary_target")):
            pt = str(row["primary_target"])
            if len(pt) > 50:
                pt = pt[:47] + "…"
            shown.at[idx, "primary_target"] = pt

        for col in ["pka_predicted", "logD_pH6", "logD_pH74", "max_dose_mg"]:
            if col in shown.columns and pd.notna(row.get(col)):
                shown.at[idx, col] = format_value_with_badge(row[col], col, fs, precision=2)
        for col in ["physical_state", "half_life"]:
            if col in shown.columns and pd.notna(row.get(col)):
                val = str(row[col])
                if len(val) > 50:
                    val = val[:47] + "…"
                shown.at[idx, col] = format_value_with_badge(val, col, fs)

    for col in ["PSA", "vapor_pressure_mmhg", "melting_point_c"]:
        if col in shown.columns:
            shown[col] = shown[col].round(2)

    def dose_badge_style(row: pd.Series) -> list[str]:
        base = highlight_nosa(row)
        if "dose_feasible_nosa" not in row.index:
            return base
        badge = row["dose_feasible_nosa"]
        dose_idx = list(row.index).index("dose_feasible_nosa")
        if badge == "✓":
            base[dose_idx] = f"background-color: #2E8B57; color: {TEXT}; font-weight: 700"
        elif badge == "✗":
            base[dose_idx] = f"background-color: #C0392B; color: {TEXT}; font-weight: 700"
        elif badge == "?":
            base[dose_idx] = f"background-color: {DISABLED}; color: {TEXT}; font-weight: 700"
        return base

    return shown.style.apply(dose_badge_style, axis=1).format(
        {"composite_score": "{:.1f}"},
        na_rep="—",
    )


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

        st.markdown("**Data sources**")
        if sources:
            source_rows = [{"Field": k, "Source": v.upper() if v == "drugbank" else v.title()} for k, v in sorted(sources.items())]
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
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
    plot_df["series"] = plot_df["nosa_candidate"].map(
        {True: "NOSA candidate", False: "Other"}
    )

    fig = px.scatter(
        plot_df,
        x="molecular_weight",
        y="logP",
        color="series",
        hover_name="name",
        hover_data={
            "composite_score": ":.1f",
            "PSA": ":.1f",
            "molecular_weight": ":.1f",
            "logP": ":.2f",
            "series": False,
        },
        color_discrete_map={
            "NOSA candidate": ACCENT,
            "Other": "rgba(255,255,255,0.45)",
        },
        symbol_map={"NOSA candidate": "star", "Other": "circle"},
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


def reset_all_filters(slider_ranges: dict[str, tuple[float, float]]) -> None:
    for col, default in slider_ranges.items():
        st.session_state[f"filter_{col}_lo"] = float(default[0])
        st.session_state[f"filter_{col}_hi"] = float(default[1])
        st.session_state[f"filter_{col}_slider"] = (float(default[0]), float(default[1]))
    for col, default in CLINICAL_DEFAULTS.items():
        if col in slider_ranges:
            st.session_state[f"filter_{col}_lo"] = float(default[0])
            st.session_state[f"filter_{col}_hi"] = float(default[1])
            st.session_state[f"filter_{col}_slider"] = (float(default[0]), float(default[1]))
    for phase in (1, 2, 3, 4):
        st.session_state[f"hide_phase_{phase}"] = False
    st.session_state["hide_patent_yes"] = False
    st.session_state["hide_patent_no"] = False


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
                slider_ranges[col],
                FILTER_STEPS[col],
                disabled=True,
            )
        return bounds

    bounds["vapor_pressure_mmhg"] = render_range_filter(
        "vapor_pressure_mmhg",
        FILTER_LABELS["vapor_pressure_mmhg"],
        *slider_ranges["vapor_pressure_mmhg"],
        slider_ranges["vapor_pressure_mmhg"],
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
        slider_ranges["melting_point_c"],
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
        default = CLINICAL_DEFAULTS.get(col, slider_ranges[col])
        bounds[col] = render_range_filter(
            col,
            FILTER_LABELS[col],
            *slider_ranges[col],
            default,
            FILTER_STEPS[col],
        )

    all_routes = unique_pipe_values(df["route_of_administration"]) if "route_of_administration" in df.columns else []
    all_atc = unique_pipe_values(df["atc_level1"]) if "atc_level1" in df.columns else []

    selected_routes = all_routes
    selected_atc = all_atc
    if all_routes:
        selected_routes = st.multiselect(
            "Route of administration",
            options=all_routes,
            default=all_routes,
        )
    if all_atc:
        selected_atc = st.multiselect(
            "ATC level 1 (therapeutic area)",
            options=all_atc,
            default=all_atc,
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
    default_bounds = {**BASE_SLIDER_RANGES, **PUBCHEM_SLIDER_RANGES, **CLINICAL_SLIDER_RANGES}
    for col, bounds in CLINICAL_DEFAULTS.items():
        if col in slider_ranges:
            default_bounds[col] = bounds
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
        st.caption("Type min/max values or drag the slider.")
        if st.button("Reset all filters"):
            reset_all_filters(slider_ranges)
            st.rerun()
        bounds: dict[str, tuple[float, float]] = {}
        for col in BASE_SLIDER_RANGES:
            bounds[col] = render_range_filter(
                col,
                FILTER_LABELS[col],
                *slider_ranges[col],
                slider_ranges[col],
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
        cns_only, hide_nasal_cyp, hide_pgp = render_drugbank_filters(df)

        st.divider()
        st.markdown("**Memantine benchmark**")
        st.markdown(benchmark_text(ideal, n_criteria))

    filtered = apply_filters(df, bounds, default_bounds)
    filtered = apply_categorical_filters(
        filtered, selected_routes, selected_atc, all_routes, all_atc
    )
    filtered = apply_metadata_filters(
        filtered, hide_phases, hide_with_patent, hide_without_patent
    )
    filtered = apply_drugbank_filters(filtered, cns_only, hide_nasal_cyp, hide_pgp)
    filtered["composite_score"] = composite_score(filtered, ideal, slider_ranges)
    filtered = filtered.sort_values("composite_score", ascending=False)

    passing = len(filtered)
    nosa_in_view = int(filtered["nosa_candidate"].fillna(False).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Drugs passing filters", f"{passing:,}")
    m2.metric("Total in database", f"{total_drugs:,}")
    m3.metric("NOSA candidates in view", nosa_in_view)

    col_table, col_chart = st.columns([1.2, 1])

    with col_table:
        st.subheader("Ranked results")
        if passing == 0:
            st.warning("No drugs match the current filters. Widen the sliders.")
        else:
            st.dataframe(
                style_results_table(filtered),
                use_container_width=True,
                height=480,
            )
        render_drug_detail_panel(filtered)

    with col_chart:
        st.subheader("logP vs molecular weight")
        if passing > 0:
            st.plotly_chart(build_scatter(filtered, ideal), use_container_width=True)
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
