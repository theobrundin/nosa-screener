#!/usr/bin/env python3
"""NOSA drug screening tool — filter and rank approved small molecules."""

from __future__ import annotations

from pathlib import Path

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

PUBCHEM_COLUMNS = list(PUBCHEM_SLIDER_RANGES.keys())

FILTER_LABELS = {
    "molecular_weight": "Molecular weight (Da)",
    "logP": "logP",
    "PSA": "PSA (Å²)",
    "H_donors": "H-bond donors",
    "rotatable_bonds": "Rotatable bonds",
    "vapor_pressure_mmhg": "Vapor pressure (mmHg)",
    "melting_point_c": "Melting point (°C)",
}

FILTER_STEPS = {
    "molecular_weight": 1.0,
    "logP": 0.1,
    "PSA": 1.0,
    "H_donors": 1.0,
    "rotatable_bonds": 1.0,
    "vapor_pressure_mmhg": 0.01,
    "melting_point_c": 1.0,
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


@st.cache_data
def load_database(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    numeric_cols = list(BASE_IDEAL.keys()) + PUBCHEM_COLUMNS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "nosa_candidate" in df.columns:
        df["nosa_candidate"] = df["nosa_candidate"].fillna(False).astype(bool)
    return df


def active_score_config(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    ideal = dict(BASE_IDEAL)
    ranges = dict(BASE_SLIDER_RANGES)
    if has_pubchem_data(df):
        ideal.update(PUBCHEM_IDEAL)
        ranges.update(PUBCHEM_SLIDER_RANGES)
    return ideal, ranges


def property_score(series: pd.Series, ideal: float, span: float) -> pd.Series:
    distance = (series - ideal).abs()
    return (100.0 * (1.0 - distance / span)).clip(lower=0.0, upper=100.0).fillna(0.0)


def composite_score(
    df: pd.DataFrame,
    ideal: dict[str, float],
    ranges: dict[str, tuple[float, float]],
) -> pd.Series:
    score_cols = [c for c in ideal if c in df.columns]
    if not score_cols:
        return pd.Series(0.0, index=df.index)
    scores = []
    for col in score_cols:
        lo, hi = ranges[col]
        scores.append(property_score(df[col], ideal[col], hi - lo))
    return sum(scores) / len(scores)


def apply_filters(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    pubchem_cols = set(PUBCHEM_COLUMNS)
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        in_range = df[col].between(lo, hi)
        if col in pubchem_cols:
            # Sparse PubChem data: missing values pass but score 0 in composite
            mask &= df[col].isna() | in_range
        else:
            mask &= df[col].notna() & in_range
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
        "molecular_weight",
        "logP",
        "PSA",
        "H_donors",
        "rotatable_bonds",
        "vapor_pressure_mmhg",
        "melting_point_c",
        "nosa_candidate",
        "indication_class",
        "earliest_patent_expiry",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    shown = df[display_cols].copy()
    shown["composite_score"] = shown["composite_score"].round(1)
    if "chembl_max_phase" in shown.columns:
        shown["chembl_max_phase"] = shown["chembl_max_phase"].apply(
            lambda v: int(v) if pd.notna(v) else "—"
        )
    for col in ["molecular_weight", "logP", "PSA", "vapor_pressure_mmhg", "melting_point_c"]:
        if col in shown.columns:
            shown[col] = shown[col].round(2)
    return shown.style.apply(highlight_nosa, axis=1).format(
        {"composite_score": "{:.1f}"},
        na_rep="—",
    )


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

    return (lo, hi)


def reset_all_filters(slider_ranges: dict[str, tuple[float, float]]) -> None:
    for col, default in slider_ranges.items():
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
        '<p class="nosa-warning">Injection molding reaches ~150–250°C — drugs '
        "melting below 150°C are thermal stability risks</p>",
        unsafe_allow_html=True,
    )
    return bounds


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
    n_criteria = len(ideal)

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
        hide_phases, hide_with_patent, hide_without_patent = render_metadata_filters(df)

        st.divider()
        st.markdown("**Memantine benchmark**")
        st.markdown(benchmark_text(ideal, n_criteria))

    filtered = apply_filters(df, bounds)
    filtered = apply_metadata_filters(
        filtered, hide_phases, hide_with_patent, hide_without_patent
    )
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
