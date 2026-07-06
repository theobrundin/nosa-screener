#!/usr/bin/env python3
"""One-off sanity check of Adrian's NOSA filter parameters against the enriched DB."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"

FILTERS = {
    "melting_point_c": (-100, 180),
    "logP": (1, 4),
    "vapor_pressure_mmhg": (0.015, 0.375),
    "PSA": (0, 9),
    "molecular_weight": (0, 900),
    "H_donors": (0, 3),
}

PSA_ALT = (0, 90)

REFERENCE_COMPOUNDS = ["memantine", "nicotine", "melatonin", "menthol"]

NOSA_CANDIDATES = {
    "memantine", "nicotine", "melatonin", "zolmitriptan", "brivaracetam",
    "vortioxetine", "morphine", "propranolol", "valproic acid", "dimethyl fumarate",
}


def load_db() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_CSV)
    numeric_cols = list(FILTERS) + ["H_acceptors", "rotatable_bonds"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def find_compound(df: pd.DataFrame, name: str) -> pd.Series | None:
    match = df[df["name"].str.lower().str.strip() == name.lower()]
    if match.empty:
        return None
    return match.iloc[0]


def passes_filter(series: pd.Series, lo: float, hi: float, *, allow_missing: bool = False) -> pd.Series:
    in_range = series.between(lo, hi)
    if allow_missing:
        return series.isna() | in_range
    return series.notna() & in_range


def format_filter_label(col: str, lo: float, hi: float) -> str:
    labels = {
        "melting_point_c": f"{lo:.0f} to {hi:.0f}",
        "logP": f"{lo:.0f} to {hi:.0f}",
        "vapor_pressure_mmhg": f"{lo:.3f}-{hi:.3f}",
        "PSA": f"{lo:.0f} to {hi:.0f}",
        "molecular_weight": f"{lo:.0f} to {hi:.0f}",
        "H_donors": f"{lo:.0f} to {hi:.0f}",
    }
    return labels.get(col, f"{lo} to {hi}")


def compound_status(row: pd.Series | None, col: str, lo: float, hi: float) -> str:
    if row is None:
        return "not in DB"
    val = row.get(col)
    if pd.isna(val):
        return "no data"
    if lo <= val <= hi:
        if col in ("molecular_weight", "melting_point_c", "H_donors"):
            return f"{val:.0f} ✓"
        if col == "logP":
            return f"{val:.2f} ✓"
        if col == "PSA":
            return f"{val:.2f} ✓"
        if col == "vapor_pressure_mmhg":
            return f"{val:.3f} ✓"
        return "✓"
    if col in ("molecular_weight", "melting_point_c", "H_donors"):
        return f"{val:.0f} ❌"
    if col == "logP":
        return f"{val:.2f} ❌"
    if col == "PSA":
        return f"{val:.2f} ❌"
    if col == "vapor_pressure_mmhg":
        return f"{val:.3f} ❌"
    return "❌"


def print_filter_row(
    label: str,
    filter_label: str,
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    memantine: pd.Series | None,
    *,
    warn: str = "",
) -> None:
    allow_missing = False
    mask = passes_filter(df[col], lo, hi, allow_missing=allow_missing)
    n_pass = int(mask.sum())
    pct = 100.0 * n_pass / len(df) if len(df) else 0.0
    mem = compound_status(memantine, col, lo, hi)
    warn_str = f"  {warn}" if warn else ""
    print(f"{label:<22}  {filter_label:<14}  {n_pass:>5,}  {pct:>5.1f}%   {mem}{warn_str}")


def combined_mask(df: pd.DataFrame, filters: dict[str, tuple[float, float]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in filters.items():
        mask &= passes_filter(df[col], lo, hi, allow_missing=False)
    return mask


def fmt_val(val: float | None, fmt: str = ".0f") -> str:
    if val is None or pd.isna(val):
        return "null"
    return format(val, fmt)


def main() -> None:
    df = load_db()
    total = len(df)
    memantine = find_compound(df, "memantine")

    print("── Filter Sanity Check ──────────────────────────")
    print(f"{'Parameter':<22}  {'Filter':<14}  {'Pass':>5}  {'%':>6}   Memantine")
    print("─" * 62)

    print_filter_row(
        "melting_point_c",
        format_filter_label("melting_point_c", *FILTERS["melting_point_c"]),
        df, "melting_point_c", *FILTERS["melting_point_c"], memantine,
    )
    print_filter_row(
        "logP",
        format_filter_label("logP", *FILTERS["logP"]),
        df, "logP", *FILTERS["logP"], memantine,
    )
    print_filter_row(
        "vapor_pressure_mmhg",
        format_filter_label("vapor_pressure_mmhg", *FILTERS["vapor_pressure_mmhg"]),
        df, "vapor_pressure_mmhg", *FILTERS["vapor_pressure_mmhg"], memantine,
    )

    psa_pass = int(passes_filter(df["PSA"], *FILTERS["PSA"]).sum())
    psa_pct = 100.0 * psa_pass / total
    psa_warn = ""
    if psa_pct < 5.0:
        psa_warn = "⚠️ LIKELY TYPO"
    print_filter_row(
        "PSA",
        format_filter_label("PSA", *FILTERS["PSA"]),
        df, "PSA", *FILTERS["PSA"], memantine,
        warn=psa_warn,
    )
    print_filter_row(
        "PSA (alt: 0 to 90)",
        format_filter_label("PSA", *PSA_ALT),
        df, "PSA", *PSA_ALT, memantine,
    )
    print_filter_row(
        "molecular_weight",
        format_filter_label("molecular_weight", *FILTERS["molecular_weight"]),
        df, "molecular_weight", *FILTERS["molecular_weight"], memantine,
    )
    print_filter_row(
        "H_donors",
        format_filter_label("H_donors", *FILTERS["H_donors"]),
        df, "H_donors", *FILTERS["H_donors"], memantine,
    )

    print()
    print("── Reference compounds ──────────────────────────")
    print(f"{'':22}  {'MW':>5}  {'logP':>5}  {'PSA':>5}  {'MP°C':>6}  {'VP_mmHg':>8}")
    for name in REFERENCE_COMPOUNDS:
        row = find_compound(df, name)
        if row is None:
            print(f"{name.capitalize():<22}  {'—':>5}  {'—':>5}  {'—':>5}  {'—':>6}  {'—':>8}")
            continue
        print(
            f"{name.capitalize():<22}  "
            f"{fmt_val(row.get('molecular_weight'), '.0f'):>5}  "
            f"{fmt_val(row.get('logP'), '.2f'):>5}  "
            f"{fmt_val(row.get('PSA'), '.0f'):>5}  "
            f"{fmt_val(row.get('melting_point_c'), '.0f'):>6}  "
            f"{fmt_val(row.get('vapor_pressure_mmhg'), '.1f' if pd.notna(row.get('vapor_pressure_mmhg')) else '') :>8}"
        )

    print()
    print("── Combined filter ──────────────────────────────")
    combined_filters = dict(FILTERS)
    combined_filters["PSA"] = PSA_ALT
    all_pass = combined_mask(df, combined_filters)
    n_all = int(all_pass.sum())
    pct_all = 100.0 * n_all / total

    nosa_mask = df["name"].str.lower().str.strip().isin(NOSA_CANDIDATES) | df.get(
        "nosa_candidate", pd.Series(False, index=df.index)
    ).fillna(False)
    nosa_total = int(nosa_mask.sum())
    nosa_pass = int((all_pass & nosa_mask).sum())

    print(f"Drugs passing ALL filters (PSA <90 variant): {n_all:,} ({pct_all:.1f}%)")
    print(f"NOSA candidates passing: {nosa_pass} / {nosa_total}")

    print()
    print(
        "Enthalpy of vaporization not in current DB. Adrian's filter value 'under 5' has "
        "no unit specified — typical values for volatile drugs are 30-60 kJ/mol, so "
        "'under 5' likely refers to a different unit or scale (possibly log scale, or "
        "kcal/mol but still unusually low). Flag for clarification."
    )


if __name__ == "__main__":
    main()
