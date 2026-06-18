#!/usr/bin/env python3
"""Sanity check for NOSA-DB data and app logic."""

import re

import pandas as pd
from pathlib import Path

from app import (
    BASE_SLIDER_RANGES,
    PUBCHEM_SLIDER_RANGES,
    active_score_config,
    apply_filters,
    composite_score,
    database_path,
    has_pubchem_data,
    load_database,
)


NOSA_CANDIDATES = {
    "memantine", "nicotine", "melatonin", "zolmitriptan", "brivaracetam",
    "vortioxetine", "morphine", "propranolol", "valproic acid", "dimethyl fumarate",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.lower().strip()
    return re.sub(r"\s+", " ", s)


def main() -> None:
    issues: list[str] = []
    warnings: list[str] = []

    print("=" * 60)
    print("NOSA-DB SANITY CHECK")
    print("=" * 60)

    path = database_path()
    print(f"\n[1] App data source: {path.name}")
    if not Path("nosa_drug_database_enriched.csv").exists():
        print("    (no enriched CSV — using master with carried-forward PubChem cols)")

    df = load_database(str(path))
    print(f"    Rows loaded: {len(df):,}")

    expected_cols = [
        "name", "chembl_id", "chembl_max_phase", "molecular_weight", "logP", "PSA",
        "H_donors", "H_acceptors", "rotatable_bonds", "SMILES", "smiles_source",
        "source", "sources", "nosa_candidate", "ema_approved", "pmda_approved",
        "pubchem_cid", "vapor_pressure_mmhg", "melting_point_c",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    else:
        print("[2] Schema: all expected columns present")

    df["_name_key"] = df["name"].map(normalize_name)
    dup_names = df["name"].duplicated().sum()
    dup_keys = df["_name_key"].duplicated().sum()
    print(f"[3] Duplicate names: {dup_names} | duplicate name_keys: {dup_keys}")
    if dup_keys > 0:
        warnings.append(f"{dup_keys} duplicate name_key rows")

    print("[4] ChEMBL max phase:")
    for p in [4, 3, 2]:
        print(f"    Phase {p}: {(df['chembl_max_phase'] == p).sum():,}")
    print(f"    Manual: {(df['source'] == 'manual').sum()}")

    print("[5] Field coverage:")
    for col, label in [
        ("SMILES", "SMILES"),
        ("molecular_weight", "MW"),
        ("ingredient", "Orange Book"),
        ("earliest_patent_expiry", "Patent expiry"),
        ("pubchem_cid", "PubChem CID"),
        ("vapor_pressure_mmhg", "Vapor pressure"),
        ("melting_point_c", "Melting point"),
        ("indication_class", "Indication class"),
    ]:
        n = df[col].notna().sum()
        print(f"    {label}: {n:,} ({100 * n / len(df):.1f}%)")

    has_smiles = df["SMILES"].notna()
    no_source = has_smiles & df["smiles_source"].isna()
    print(
        f"[6] smiles_source: chembl={(df['smiles_source'] == 'chembl').sum()}, "
        f"drugcentral={(df['smiles_source'] == 'drugcentral').sum()}, "
        f"missing={no_source.sum()}"
    )
    if no_source.sum() > 0:
        warnings.append(f"{no_source.sum()} rows have SMILES but no smiles_source")

    print(f"[7] EMA/PMDA flags: {df['ema_approved'].sum()} each")

    nosa = df[df["nosa_candidate"] == True]
    print(f"[8] NOSA candidates: {len(nosa)} rows")
    for c in sorted(NOSA_CANDIDATES):
        match = df[(df["name"].str.lower().str.strip() == c) | (df["_name_key"] == c)]
        flagged = bool(match["nosa_candidate"].any()) if len(match) else False
        print(f"    {c}: {len(match)} row(s), flagged={flagged}")
        if not flagged:
            warnings.append(f"NOSA candidate not flagged: {c}")

    pubchem_ok = has_pubchem_data(df)
    ideal, ranges = active_score_config(df)
    print(f"[9] PubChem sliders: {pubchem_ok} | score criteria: {len(ideal)}")

    bounds = {**BASE_SLIDER_RANGES, **(PUBCHEM_SLIDER_RANGES if pubchem_ok else {})}
    filtered = apply_filters(df, bounds)
    filtered["composite_score"] = composite_score(filtered, ideal, ranges)
    mem = filtered[filtered["_name_key"] == "memantine"]
    if len(mem):
        mem_score = mem["composite_score"].max()
        rank = int((filtered["composite_score"] > mem_score).sum()) + 1
        print(f"[10] Memantine score: {mem_score:.1f}, rank {rank}/{len(filtered)}")
        # MP ideal (130°C) is a platform target, not memantine's measured MP (~290°C)
        if mem_score < 70:
            warnings.append(f"Memantine score unexpectedly low: {mem_score:.1f}")
    else:
        issues.append("Memantine missing from filtered set")

    if pubchem_ok:
        tight = apply_filters(df, {**bounds, "vapor_pressure_mmhg": (0.01, 1.0)})
        print(f"[11] VP filter 0.01-1.0: {len(tight)} drugs (default: {len(filtered)})")

    print("[12] Sources:", df["source"].value_counts().to_dict())

    manual_df = df[df["source"] == "manual"]
    if len(manual_df):
        print(f"[13] Manual ({len(manual_df)}):")
        missing_props = 0
        for _, r in manual_df.iterrows():
            sm = "yes" if pd.notna(r["SMILES"]) else "NO"
            print(f"    {r['name']}: MW={r['molecular_weight']}, SMILES={sm}")
            if pd.isna(r["molecular_weight"]) or pd.isna(r["SMILES"]):
                missing_props += 1
        if missing_props:
            issues.append(f"{missing_props} manual compounds missing MW or SMILES")

    print("\n" + "=" * 60)
    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  ✗ {i}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    if not issues and not warnings:
        print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
