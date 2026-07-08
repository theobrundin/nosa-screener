#!/usr/bin/env python3
"""Sanity check for NOSA-DB data, coverage numbers, and screener filter logic."""

from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd
from pathlib import Path

from app import (
    BASE_SLIDER_RANGES,
    CLINICAL_DEFAULTS,
    CLINICAL_SLIDER_RANGES,
    PUBCHEM_SLIDER_RANGES,
    active_score_config,
    apply_categorical_filters,
    apply_drugbank_filters,
    apply_filters,
    apply_metadata_filters,
    composite_score,
    database_path,
    has_pubchem_data,
    load_database,
    unique_pipe_values,
)

NOSA_CANDIDATES = {
    "memantine", "nicotine", "melatonin", "zolmitriptan", "brivaracetam",
    "vortioxetine", "morphine", "propranolol", "valproic acid", "dimethyl fumarate",
}

KNOWN_DRUGS = ["MEMANTINE", "NICOTINE", "ASPIRIN", "METFORMIN", "MORPHINE"]


def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.lower().strip()
    return re.sub(r"\s+", " ", s)


def default_app_bounds(slider_ranges: dict) -> dict:
    bounds = {**BASE_SLIDER_RANGES, **PUBCHEM_SLIDER_RANGES, **CLINICAL_SLIDER_RANGES}
    for col, clinical in CLINICAL_DEFAULTS.items():
        if col in slider_ranges:
            bounds[col] = clinical
    return bounds


def simulate_default_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    ideal, slider_ranges = active_score_config(df)
    default_bounds = default_app_bounds(slider_ranges)
    filtered = apply_filters(df, default_bounds, default_bounds)
    all_routes = unique_pipe_values(df["route_of_administration"]) if "route_of_administration" in df.columns else []
    all_atc = unique_pipe_values(df["atc_level1"]) if "atc_level1" in df.columns else []
    filtered = apply_categorical_filters(filtered, all_routes, all_atc, all_routes, all_atc)
    filtered = apply_metadata_filters(filtered, set(), False, False)
    filtered = apply_drugbank_filters(filtered, False, False, False)
    filtered = filtered.copy()
    filtered["composite_score"] = composite_score(filtered, ideal, slider_ranges)
    return filtered


def check_missing_data_passes(df: pd.DataFrame, issues: list[str]) -> None:
    """Sparse filters must not exclude rows solely for missing values at default bounds."""
    default_bounds = default_app_bounds(active_score_config(df)[1])
    for col in ("vapor_pressure_mmhg", "melting_point_c", "max_dose_mg", "pka_predicted", "logD_pH6"):
        if col not in df.columns:
            continue
        lo, hi = default_bounds[col]
        missing = df[col].isna()
        if not missing.any():
            continue
        subset = df.loc[missing, [col]].copy()
        passed = apply_filters(subset, {col: (lo, hi)}, default_bounds)
        if len(passed) != missing.sum():
            issues.append(
                f"{col}: {missing.sum() - len(passed)} drugs with missing data wrongly excluded at default bounds"
            )


def check_random_sample(df: pd.DataFrame, warnings: list[str], n: int = 80) -> None:
    sample = df.sample(n, random_state=42)
    problems = 0
    for _, row in sample.iterrows():
        flags: list[str] = []
        mw = row.get("molecular_weight")
        if pd.notna(mw) and (mw < 50 or mw > 1500):
            flags.append(f"MW={mw}")
        if pd.notna(row.get("target_count")) and pd.isna(row.get("target_names")):
            flags.append("target_count without names")
        if pd.notna(row.get("max_dose_mg")) and pd.isna(row.get("max_dose_source")):
            flags.append("dose without source")
        if flags:
            problems += 1
    if problems:
        warnings.append(f"Random sample: {problems}/{n} rows with minor data quirks")


def main() -> None:
    issues: list[str] = []
    warnings: list[str] = []

    print("=" * 70)
    print("NOSA-DB SANITY CHECK")
    print("=" * 70)

    path = database_path()
    print(f"\n[1] App data source: {path.name}")
    df = load_database(str(path))
    n = len(df)
    print(f"    Rows loaded: {n:,}")

    expected_cols = [
        "name", "chembl_id", "chembl_max_phase", "molecular_weight", "logP", "PSA",
        "H_donors", "H_acceptors", "rotatable_bonds", "SMILES", "nosa_candidate",
        "pubchem_cid", "vapor_pressure_mmhg", "melting_point_c",
        "pka_predicted", "logD_pH6", "max_dose_mg", "dose_feasible_nosa",
        "cns_target", "primary_target", "nasal_cyp_risk", "pgp_substrate",
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    else:
        print("[2] Schema: all expected columns present")

    print("[3] Coverage highlights:")
    for col, label in [
        ("molecular_weight", "MW"),
        ("SMILES", "SMILES"),
        ("pubchem_cid", "PubChem CID"),
        ("vapor_pressure_mmhg", "Vapor pressure"),
        ("melting_point_c", "Melting point"),
        ("pka_predicted", "pKa predicted"),
        ("logD_pH6", "logD pH6"),
        ("max_dose_mg", "Unified max dose"),
        ("target_names", "DrugBank target names"),
        ("cns_target", "CNS flag (any)"),
    ]:
        if col in df.columns:
            if col == "cns_target":
                true_n = int((df[col] == True).sum())
                print(f"    {label}: {true_n:,} True / {n:,} rows")
            else:
                nn = int(df[col].notna().sum())
                print(f"    {label}: {nn:,} ({100 * nn / n:.1f}%)")

    print("[4] ChEMBL max phase:")
    for p in [4, 3, 2]:
        print(f"    Phase {p}: {(df['chembl_max_phase'] == p).sum():,}")
    print(f"    Manual: {(df['source'] == 'manual').sum()}")

    df["_name_key"] = df["name"].map(normalize_name)
    nosa = df[df["nosa_candidate"] == True]
    print(f"[5] NOSA candidates: {len(nosa)} rows")
    for c in sorted(NOSA_CANDIDATES):
        match = df[(df["name"].str.lower().str.strip() == c) | (df["_name_key"] == c)]
        flagged = bool(match["nosa_candidate"].any()) if len(match) else False
        if not flagged:
            warnings.append(f"NOSA candidate not flagged: {c}")

    print("[6] Known drug spot checks:")
    for kn in KNOWN_DRUGS:
        r = df[df["name"].str.upper() == kn]
        if r.empty:
            warnings.append(f"{kn} not found in database")
            continue
        r = r.iloc[0]
        print(
            f"    {kn}: MW={r.get('molecular_weight', '—')} logP={r.get('logP', '—')} "
            f"cns={r.get('cns_target', '—')} dose={r.get('max_dose_mg', '—')}"
        )

    check_missing_data_passes(df, issues)
    check_random_sample(df, warnings)

    print("[7] Default screener pipeline:")
    filtered = simulate_default_pipeline(df)
    print(f"    Drugs passing default filters: {len(filtered):,} / {n:,} ({100 * len(filtered) / n:.1f}%)")

    nosa_pass = filtered[filtered["nosa_candidate"] == True]
    print(f"    NOSA candidates in view: {len(nosa_pass)} / {len(nosa)}")
    for _, r in nosa.iterrows():
        in_view = r["name"] in set(filtered["name"])
        mark = "✓" if in_view else "✗"
        print(f"      {mark} {r['name']}")

    mem = filtered[filtered["_name_key"] == "memantine"]
    if len(mem):
        mem_score = float(mem["composite_score"].max())
        rank = int((filtered["composite_score"] > mem_score).sum()) + 1
        print(f"[8] Memantine score: {mem_score:.1f}, rank {rank}/{len(filtered)}")
    else:
        warnings.append(
            "Memantine excluded from default view (likely logD outside 1–4, not missing data)"
        )

    pubchem_ok = has_pubchem_data(df)
    if pubchem_ok:
        bounds = default_app_bounds(active_score_config(df)[1])
        tight_vp = apply_filters(df, {**bounds, "vapor_pressure_mmhg": (0.01, 1.0)}, bounds)
        print(f"[9] VP narrowed to 0.01–1.0: {len(tight_vp):,} (default: {len(filtered):,})")

    structural = apply_filters(df, BASE_SLIDER_RANGES, BASE_SLIDER_RANGES)
    print(f"[10] Structural filters only: {len(structural):,} ({100 * len(structural) / n:.1f}%)")
    print(f"     Missing MW always excluded from structural filters: {int(df['molecular_weight'].isna().sum()):,}")

    print("\n" + "=" * 70)
    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  ✗ {i}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    if not issues:
        print("All critical checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
