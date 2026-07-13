#!/usr/bin/env python3
"""Advanced read-only audit of the NOSA database pipeline and enriched output."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app import (
    BASE_SLIDER_RANGES,
    CLINICAL_DEFAULTS,
    CLINICAL_SLIDER_RANGES,
    PUBCHEM_COLUMNS,
    PUBCHEM_SLIDER_RANGES,
    active_score_config,
    apply_applicant_filters,
    apply_categorical_filters,
    apply_drugbank_filters,
    apply_filters,
    apply_metadata_filters,
    composite_score,
    unique_pipe_values,
)
from build_database import calculate_logd, normalize_name

ROOT = Path(__file__).resolve().parent
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"
MASTER_CSV = ROOT / "nosa_drug_database.csv"
DRUGBANK_CSV = ROOT / "kaggle-drugbank" / "drugbank_clean.csv"
ORANGE_BOOK_PRODUCTS = ROOT / "EOBZIP_2026_04" / "products.txt"

EXPECTED_ROWS = 10_545
EXPECTED_DEFAULT_PASS = 3_683
DEFAULT_PASS_TOLERANCE = 150

NOSA_CANDIDATES = {
    "memantine",
    "nicotine",
    "melatonin",
    "zolmitriptan",
    "brivaracetam",
    "vortioxetine",
    "morphine",
    "propranolol",
    "valproic acid",
    "dimethyl fumarate",
}

MEMANTINE_BENCHMARKS = {
    "molecular_weight": (170.0, 185.0),
    "logP": (2.3, 3.1),
    "PSA": (20.0, 35.0),
    "H_donors": (1.0, 1.0),
    "pka_predicted": (8.5, 10.5),
    "logD_pH6": (-2.0, 0.5),
}

SALT_COUNTERION_MASSES = {
    "HCl": 36.5,
    "hydrochloride": 36.5,
    "sodium": 23.0,
    "sulfate": 98.1,
    "mesylate": 94.1,
    "maleate": 116.1,
    "tartrate": 150.1,
    "phosphate": 98.0,
}

CORE_REQUIRED_FIELDS = {"molecular_weight", "logP", "PSA", "H_donors", "rotatable_bonds"}
SPARSE_OPTIONAL_FIELDS = {
    "vapor_pressure_mmhg",
    "melting_point_c",
    "max_dose_mg",
    "pka_predicted",
    "logD_pH6",
}

Status = str  # "PASS" | "WARN" | "FAIL"


@dataclass
class SectionResult:
    title: str
    status: Status = "PASS"
    lines: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def pass_(self, msg: str) -> None:
        self.lines.append(f"  ✓ {msg}")

    def warn(self, msg: str, details: list[str] | None = None) -> None:
        self._bump("WARN")
        self.lines.append(f"  ⚠ {msg}")
        for detail in details or []:
            self.lines.append(f"      {detail}")

    def fail(self, msg: str, details: list[str] | None = None) -> None:
        self._bump("FAIL")
        self.lines.append(f"  ✗ {msg}")
        for detail in details or []:
            self.lines.append(f"      {detail}")

    def info(self, msg: str) -> None:
        self.lines.append(f"  · {msg}")

    def _bump(self, level: Status) -> None:
        if level == "FAIL" or (level == "WARN" and self.status == "PASS"):
            self.status = level


@dataclass
class AuditContext:
    df: pd.DataFrame
    n: int
    today: date
    drugbank_source: pd.DataFrame | None = None
    orange_book_ingredients: int | None = None


def pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def coverage(df: pd.DataFrame, col: str) -> tuple[int, float]:
    if col not in df.columns:
        return 0, 0.0
    nn = int(df[col].notna().sum())
    return nn, 100.0 * nn / len(df) if len(df) else 0.0


def default_app_bounds(slider_ranges: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
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
    filtered = apply_applicant_filters(filtered, [], False)
    filtered = apply_drugbank_filters(filtered, False, False, False)
    filtered = filtered.copy()
    filtered["composite_score"] = composite_score(filtered, ideal, slider_ranges)
    return filtered


def source_verdict(rate: float, full_threshold: float = 85.0, partial_threshold: float = 40.0) -> str:
    if rate >= full_threshold:
        return f"fully utilized ({rate:.1f}%)"
    if rate >= partial_threshold:
        return f"partially utilized ({rate:.1f}% of available data)"
    return f"under-utilized — investigate ({rate:.1f}%)"


def load_audit_data() -> AuditContext:
    path = ENRICHED_CSV if ENRICHED_CSV.exists() else MASTER_CSV
    if not path.exists():
        raise FileNotFoundError("No database CSV found (expected nosa_drug_database_enriched.csv)")

    df = pd.read_csv(path, low_memory=False)
    if "name_key" not in df.columns:
        df["name_key"] = df["name"].map(normalize_name)

    drugbank_source = None
    if DRUGBANK_CSV.exists():
        drugbank_source = pd.read_csv(DRUGBANK_CSV, low_memory=False, usecols=lambda c: c in {"name", "drugbank_id"})
        drugbank_source["name_key"] = drugbank_source["name"].map(normalize_name)
        drugbank_source = drugbank_source[drugbank_source["name_key"] != ""].drop_duplicates("name_key")

    orange_book_ingredients = None
    if ORANGE_BOOK_PRODUCTS.exists():
        products = pd.read_csv(ORANGE_BOOK_PRODUCTS, sep="~", dtype=str, usecols=["Ingredient"])
        keys: set[str] = set()
        for ingredient in products["Ingredient"].dropna():
            for part in str(ingredient).split(";"):
                part = part.strip()
                if part:
                    keys.add(normalize_name(part))
        orange_book_ingredients = len(keys)

    return AuditContext(df=df, n=len(df), today=date.today(), drugbank_source=drugbank_source, orange_book_ingredients=orange_book_ingredients)


def section_1_structural(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 1 — Structural integrity")
    df, n = ctx.df, ctx.n

    sec.info(f"Primary file: {ENRICHED_CSV.name if ENRICHED_CSV.exists() else MASTER_CSV.name}")
    sec.info(f"Total rows: {n:,} (expected {EXPECTED_ROWS:,})")
    if n == EXPECTED_ROWS:
        sec.pass_(f"Row count matches last build summary ({EXPECTED_ROWS:,})")
    elif abs(n - EXPECTED_ROWS) <= 10:
        sec.warn(f"Row count {n:,} differs slightly from expected {EXPECTED_ROWS:,}")
    else:
        sec.fail(f"Row count {n:,} differs from expected {EXPECTED_ROWS:,}")

    full_dupes = int(df.duplicated().sum())
    if full_dupes == 0:
        sec.pass_("No fully duplicate rows")
    else:
        sec.fail(f"{full_dupes} fully duplicate rows", df[df.duplicated()].head(5)["name"].astype(str).tolist())

    name_key_dupes = df[df["name_key"].duplicated(keep=False)]
    if name_key_dupes.empty:
        sec.pass_("No duplicate name_key values")
    else:
        sec.fail(
            f"{len(name_key_dupes)} rows share duplicate name_key values",
            name_key_dupes.groupby("name_key")["name"].apply(lambda s: " | ".join(s.head(3))).head(5).tolist(),
        )

    null_names = int(df["name"].isna().sum()) + int((df["name"].astype(str).str.strip() == "").sum())
    null_keys = int(df["name_key"].isna().sum()) + int((df["name_key"].astype(str).str.strip() == "").sum())
    if null_names == 0 and null_keys == 0:
        sec.pass_("Every row has non-null name and name_key")
    else:
        sec.fail(f"Null/empty names: {null_names}, name_key: {null_keys}")

    sec.info(f"Columns present: {len(df.columns)}")
    dead_cols = [c for c in df.columns if df[c].isna().all()]
    if dead_cols:
        sec.warn(f"{len(dead_cols)} column(s) are 100% null (dead columns)", dead_cols[:20])
    else:
        sec.pass_("No 100%-null dead columns")

    if len(df.columns) == len(set(df.columns)):
        sec.pass_("No duplicated column names")
    else:
        sec.fail("Duplicated column names detected")

    return sec


def section_2_source_utilization(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 2 — Source utilization audit")
    df, n = ctx.df, ctx.n

    chembl_id_n, chembl_id_pct = coverage(df, "chembl_id")
    sec.info(f"ChEMBL ID coverage: {chembl_id_n:,} ({chembl_id_pct:.1f}%)")
    chembl_fields = {
        "molecular_weight": "MW",
        "logP": "logP",
        "PSA": "PSA",
        "mechanism_of_action": "mechanism",
        "atc_code": "ATC",
        "mesh_heading": "MeSH",
    }
    chembl_low: list[str] = []
    for col, label in chembl_fields.items():
        nn, p = coverage(df, col)
        sec.info(f"  ChEMBL field {label}: {nn:,} ({p:.1f}%)")
        if chembl_id_n and nn < chembl_id_n * 0.5:
            chembl_low.append(f"{label} ({p:.1f}% vs chembl_id {chembl_id_pct:.1f}%)")
    if chembl_low:
        sec.warn("ChEMBL fields far below chembl_id coverage", chembl_low)
    else:
        sec.pass_("ChEMBL field coverage consistent with chembl_id presence")
    sec.info(f"ChEMBL verdict: {source_verdict(min(chembl_id_pct, 95.0))}")

    db_matched = int(df["drugbank_id"].notna().sum()) if "drugbank_id" in df.columns else 0
    db_pct = 100.0 * db_matched / n
    source_n = len(ctx.drugbank_source) if ctx.drugbank_source is not None else 0
    source_keys = set(ctx.drugbank_source["name_key"]) if ctx.drugbank_source is not None else set()
    db_keys_in_db = set(df.loc[df["drugbank_id"].notna(), "name_key"])
    match_rate = 100.0 * len(db_keys_in_db) / source_n if source_n else 0.0
    sec.info(f"DrugBank matched rows: {db_matched:,} ({db_pct:.1f}% of database)")
    sec.info(f"DrugBank source indexed (dedup name_key): {source_n:,}")
    sec.info(f"Match rate vs DrugBank index: {match_rate:.1f}%")
    if ctx.drugbank_source is not None:
        unmatched = ctx.drugbank_source[~ctx.drugbank_source["name_key"].isin(db_keys_in_db)]
        sample = unmatched["name"].dropna().head(10).tolist()
        sec.info(f"Sample unmatched DrugBank names ({len(unmatched):,} total):")
        for name in sample:
            sec.info(f"    - {name}")
    sec.info(f"DrugBank verdict: {source_verdict(match_rate, full_threshold=30.0, partial_threshold=15.0)}")

    cid_n, cid_pct = coverage(df, "pubchem_cid")
    mp_n, mp_pct = coverage(df, "melting_point_c")
    vp_n, vp_pct = coverage(df, "vapor_pressure_mmhg")
    bp_n, bp_pct = coverage(df, "boiling_point_c")
    sec.info(f"PubChem CID: {cid_n:,} ({cid_pct:.1f}%)")
    sec.info(f"PubChem melting point: {mp_n:,} ({mp_pct:.1f}%)")
    sec.info(f"PubChem vapor pressure: {vp_n:,} ({vp_pct:.1f}%)")
    sec.info(f"PubChem boiling point: {bp_n:,} ({bp_pct:.1f}%)")
    if cid_n and mp_n < cid_n * 0.05:
        sec.warn("Melting point coverage unusually low relative to CID resolution")
    sec.info(f"PubChem verdict: {source_verdict(cid_pct, full_threshold=90.0, partial_threshold=50.0)}")

    patent_n, patent_pct = coverage(df, "earliest_patent_expiry")
    applicant_n, applicant_pct = coverage(df, "original_applicant")
    sec.info(f"Orange Book patent expiry: {patent_n:,} ({patent_pct:.1f}%)")
    sec.info(f"Orange Book applicant data: {applicant_n:,} ({applicant_pct:.1f}%)")
    chembl_without_patent = int((df["chembl_id"].notna() & df["earliest_patent_expiry"].isna()).sum())
    sec.info(f"ChEMBL rows without patent data: {chembl_without_patent:,} (expected gap for non-FDA drugs)")
    if ctx.orange_book_ingredients:
        ob_proxy = int((df["earliest_patent_expiry"].notna() | df["original_applicant"].notna()).sum())
        sec.info(f"Rows with any Orange Book signal: {ob_proxy:,} vs ~{ctx.orange_book_ingredients:,} OB ingredients")
    sec.info(f"Orange Book verdict: {source_verdict(applicant_pct, full_threshold=20.0, partial_threshold=10.0)}")

    dose_n, dose_pct = coverage(df, "max_dose_mg")
    ct_n, ct_pct = coverage(df, "ct_max_dose_mg")
    sec.info(f"Unified max dose: {dose_n:,} ({dose_pct:.1f}%)")
    sec.info(f"ClinicalTrials.gov dose: {ct_n:,} ({ct_pct:.1f}%)")
    sec.info(f"ClinicalTrials verdict: {source_verdict(ct_pct, full_threshold=25.0, partial_threshold=10.0)}")

    return sec


def section_3_join_quality(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 3 — Join quality and name-matching gaps")
    df, n = ctx.df, ctx.n

    db_matched = int(df["drugbank_id"].notna().sum())
    pubchem_matched = int(df["pubchem_cid"].notna().sum())
    ob_matched = int((df["earliest_patent_expiry"].notna() | df["original_applicant"].notna()).sum())
    sec.info(f"DrugBank join rate: {db_matched:,}/{n:,} ({pct(db_matched, n)})")
    sec.info(f"PubChem join rate: {pubchem_matched:,}/{n:,} ({pct(pubchem_matched, n)})")
    sec.info(f"Orange Book join rate: {ob_matched:,}/{n:,} ({pct(ob_matched, n)})")

    phase = pd.to_numeric(df["chembl_max_phase"], errors="coerce")
    high_phase = df[(phase >= 3) & df["chembl_id"].notna() & df["drugbank_id"].isna()]
    if len(high_phase) > 0:
        sample = high_phase.nlargest(10, "chembl_max_phase")[["name", "chembl_max_phase", "chembl_id"]]
        sec.warn(
            f"{len(high_phase):,} phase ≥3 drugs with ChEMBL ID but no DrugBank match",
            [f"{r['name']} (phase {int(r['chembl_max_phase'])})" for _, r in sample.iterrows()],
        )
    else:
        sec.pass_("No phase ≥3 ChEMBL drugs missing DrugBank match")

    nosa_missing_db = df[
        df["name"].str.lower().str.strip().isin(NOSA_CANDIDATES) & df["drugbank_id"].isna()
    ]
    if nosa_missing_db.empty:
        sec.pass_("All NOSA candidates matched to DrugBank")
    else:
        sec.fail(
            "NOSA candidates missing DrugBank match",
            nosa_missing_db["name"].tolist(),
        )

    keys = sorted(set(df["name_key"].dropna()))
    salt_pairs: list[str] = []
    by_stem: dict[str, list[str]] = {}
    for key in keys:
        stem = key.split()[0] if key else key
        by_stem.setdefault(stem, []).append(key)
    for group in by_stem.values():
        group = sorted(group, key=len)
        for i, short in enumerate(group):
            for long in group[i + 1 :]:
                if long.startswith(short + " "):
                    salt_pairs.append(f"{short} ↔ {long}")
                if len(salt_pairs) >= 50:
                    break
            if len(salt_pairs) >= 50:
                break
        if len(salt_pairs) >= 50:
            break
    if len(salt_pairs) >= 5:
        sec.warn(f"Potential salt-form fragmentation: {len(salt_pairs)}+ prefix-related name_key pairs", salt_pairs[:10])
        sec.recommendations.append(
            f"Salt-form deduplication: {len(salt_pairs)}+ prefix-related name_key pairs suggest over-splitting"
        )
    else:
        sec.pass_(f"Limited salt-form fragmentation ({len(salt_pairs)} prefix-related pairs)")

    smiles_no_db = df[df["SMILES"].notna() & df["drugbank_id"].isna()]
    smiles_no_pubchem = df[df["SMILES"].notna() & df["pubchem_cid"].isna()]
    sec.info(f"Rows with SMILES but no DrugBank match: {len(smiles_no_db):,}")
    sec.info(f"Rows with SMILES but no PubChem CID: {len(smiles_no_pubchem):,}")
    sec.info(
        "InChIKey re-match estimate: could recover DrugBank data for "
        f"~{len(smiles_no_db):,} rows and PubChem for ~{len(smiles_no_pubchem):,} rows currently missing name joins"
    )
    if len(smiles_no_db) > 500:
        sec.recommendations.append(
            f"InChIKey re-match: could recover DrugBank data for ~{len(smiles_no_db):,} rows with SMILES but no DrugBank match"
        )

    return sec


def section_4_calculations(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 4 — Calculation correctness")
    df = ctx.df

    sample = df.dropna(subset=["logP", "pka_predicted", "logD_pH6", "ionization_class"]).sample(
        min(20, len(df)), random_state=42
    )
    mismatches: list[str] = []
    for _, row in sample.iterrows():
        expected = calculate_logd(
            float(row["logP"]),
            float(row["pka_predicted"]),
            str(row["ionization_class"]),
            6.0,
        )
        if expected is None:
            continue
        if abs(expected - float(row["logD_pH6"])) > 0.05:
            mismatches.append(
                f"{row['name']}: stored={float(row['logD_pH6']):.3f} recomputed={expected:.3f}"
            )
    if mismatches:
        sec.fail(f"logD_pH6 mismatches in random sample ({len(mismatches)}/20)", mismatches[:10])
    else:
        sec.pass_("logD_pH6 matches acid/base recomputation in random sample (±0.05)")

    both = df.dropna(subset=["logD_pH6", "logD_pH74", "ionization_class"])
    direction_errors: list[str] = []
    for _, row in both.iterrows():
        ion = str(row["ionization_class"])
        ld6 = float(row["logD_pH6"])
        ld74 = float(row["logD_pH74"])
        if ion in {"base", "zwitterion"} and ld6 > ld74 + 0.01:
            direction_errors.append(f"{row['name']}: base but logD_pH6 ({ld6:.2f}) > logD_pH74 ({ld74:.2f})")
        if ion == "acid" and ld74 > ld6 + 0.01:
            direction_errors.append(f"{row['name']}: acid but logD_pH74 ({ld74:.2f}) > logD_pH6 ({ld6:.2f})")
    if direction_errors:
        sec.fail(f"logD pH direction errors ({len(direction_errors)})", direction_errors[:10])
    else:
        sec.pass_("logD_pH6 vs logD_pH74 direction correct for acid/base classes")

    mem_rows = df[df["name"].str.lower().str.strip() == "memantine"]
    if mem_rows.empty:
        sec.fail("Memantine not found — critical anchor compound missing")
    else:
        mem = mem_rows.iloc[0]
        mem_fails: list[str] = []
        for field, (lo, hi) in MEMANTINE_BENCHMARKS.items():
            val = mem.get(field)
            if pd.isna(val) or not (lo <= float(val) <= hi):
                mem_fails.append(f"{field}={val} (expected {lo}–{hi})")
        if mem_fails:
            sec.fail("Memantine benchmark deviation — pipeline may miscompute anchor", mem_fails)
        else:
            sec.pass_("Memantine benchmark values within expected ranges")

    dose_logic_errors: list[str] = []
    if "max_dose_mg" in df.columns and "dose_feasible_nosa" in df.columns:
        with_dose = df[df["max_dose_mg"].notna()]
        bad_low = with_dose[(with_dose["max_dose_mg"] <= 100) & (with_dose["dose_feasible_nosa"] != True)]
        bad_high = with_dose[(with_dose["max_dose_mg"] > 100) & (with_dose["dose_feasible_nosa"] != False)]
        missing_feasible = df[df["max_dose_mg"].isna() & df["dose_feasible_nosa"].notna()]
        for _, row in pd.concat([bad_low, bad_high, missing_feasible]).head(20).iterrows():
            dose_logic_errors.append(
                f"{row['name']}: max_dose_mg={row.get('max_dose_mg')} feasible={row.get('dose_feasible_nosa')}"
            )
        total_bad = len(bad_low) + len(bad_high) + len(missing_feasible)
        if total_bad:
            ct_null = bad_low[bad_low["max_dose_source"] == "clinicaltrials"] if "max_dose_source" in bad_low.columns else bad_low.iloc[0:0]
            detail_msg = f"dose_feasible_nosa logic violations ({total_bad})"
            if len(ct_null):
                detail_msg += f" — {len(ct_null)} CT-sourced doses ≤100 mg lack feasible flag (sync issue)"
            sec.fail(detail_msg, dose_logic_errors[:10])
            sec.recommendations.append(
                f"Sync dose_feasible_nosa with unified max_dose_mg: {total_bad:,} rows inconsistent"
            )
        else:
            sec.pass_("dose_feasible_nosa logic consistent with max_dose_mg ≤100 rule")

    ideal, ranges = active_score_config(df)
    scored = df.copy()
    scored["composite_score"] = composite_score(scored, ideal, ranges)
    out_of_range = scored[(scored["composite_score"] < 0) | (scored["composite_score"] > 100)]
    if len(out_of_range):
        sec.fail(f"{len(out_of_range)} composite scores outside 0–100", out_of_range[["name", "composite_score"]].head(5).astype(str).values.tolist())
    else:
        sec.pass_("All composite scores within 0–100")

    filtered = simulate_default_pipeline(df)
    mem = filtered[filtered["name"].str.lower().str.strip() == "memantine"]
    if len(mem):
        rank = int((filtered["composite_score"] > float(mem["composite_score"].max())).sum()) + 1
        pct_rank = 100.0 * rank / len(filtered)
        if pct_rank <= 5:
            sec.pass_(f"Memantine ranks {rank}/{len(filtered)} ({pct_rank:.1f}%) under default filters")
        else:
            sec.warn(f"Memantine rank {rank}/{len(filtered)} ({pct_rank:.1f}%) — below top ~1% expectation")
    else:
        sec.warn("Memantine excluded from default-filter view (logD default 1–4 excludes it)")

    if "cns_target" in df.columns:
        cns = df[df["cns_target"].fillna(False).astype(bool)]
        non = df[~df["cns_target"].fillna(False).astype(bool)]
        cns_mean = float(scored.loc[cns.index, "composite_score"].mean()) if len(cns) else 0.0
        non_mean = float(scored.loc[non.index, "composite_score"].mean()) if len(non) else 0.0
        if cns_mean > non_mean:
            sec.pass_(f"CNS drugs average higher composite score ({cns_mean:.1f} vs {non_mean:.1f})")
        else:
            sec.warn(f"CNS drugs do not average higher than non-CNS ({cns_mean:.1f} vs {non_mean:.1f})")

    if "max_dose_mg" in df.columns:
        doses = df["max_dose_mg"].dropna()
        sec.info(f"max_dose_mg distribution: min={doses.min():.2f}, median={doses.median():.2f}, max={doses.max():.2f}")
        huge = df[df["max_dose_mg"] > 5000]
        tiny = df[(df["max_dose_mg"] > 0) & (df["max_dose_mg"] < 0.001)]
        if len(huge):
            sec.fail(
                f"{len(huge)} doses > 5000 mg (likely unit/parsing error)",
                huge[["name", "max_dose_mg", "max_dose_source"]].head(10).astype(str).values.tolist(),
            )
            sec.recommendations.append(
                f"Review {len(huge)} mega-dose values >5000 mg for mg/µg unit confusion"
            )
        else:
            sec.pass_("No suspicious mega-doses > 5000 mg")
        if len(tiny):
            sec.warn(f"{len(tiny)} suspiciously tiny doses < 0.001 mg", tiny[["name", "max_dose_mg"]].head(5).astype(str).values.tolist())
        sec.info("Unit check: values should be milligrams; flag if many look like µg or grams")

    return sec


def section_5_missing_data(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 5 — Missing-data handling verification")
    df = ctx.df
    ideal, slider_ranges = active_score_config(df)
    default_bounds = default_app_bounds(slider_ranges)

    for col in sorted(SPARSE_OPTIONAL_FIELDS):
        if col not in df.columns:
            continue
        lo, hi = default_bounds[col]
        missing = df[col].isna()
        if not missing.any():
            sec.pass_(f"{col}: no missing values to test")
            continue
        subset = df.loc[missing, [col]]
        passed = apply_filters(subset, {col: (lo, hi)}, default_bounds)
        if len(passed) == missing.sum():
            sec.pass_(f"{col}: all {missing.sum():,} missing rows pass at default bounds")
        else:
            sec.fail(
                f"{col}: {missing.sum() - len(passed):,} missing rows wrongly excluded at default",
                subset.index.difference(passed.index).tolist()[:10],
            )

    for col in sorted(CORE_REQUIRED_FIELDS):
        if col not in df.columns:
            continue
        missing = df[col].isna()
        if not missing.any():
            continue
        subset = df.loc[missing, [col]]
        passed = apply_filters(subset, {col: default_bounds[col]}, default_bounds)
        if len(passed) == 0:
            sec.pass_(f"{col}: missing values hard-excluded ({missing.sum():,} rows)")
        else:
            sec.fail(f"{col}: {len(passed)} rows with missing values incorrectly pass structural filter")

    filtered = simulate_default_pipeline(df)
    sec.info(f"Default filter pass count: {len(filtered):,} / {len(df):,} ({pct(len(filtered), len(df))})")
    if abs(len(filtered) - EXPECTED_DEFAULT_PASS) <= DEFAULT_PASS_TOLERANCE:
        sec.pass_(f"Default pass count near expected ~{EXPECTED_DEFAULT_PASS:,} (actual {len(filtered):,})")
    else:
        sec.warn(
            f"Default pass count {len(filtered):,} differs from expected ~{EXPECTED_DEFAULT_PASS:,} "
            f"(±{DEFAULT_PASS_TOLERANCE})"
        )

    if "pka_predicted" in df.columns and "logD_pH6" in df.columns:
        bounds = default_bounds.copy()
        bounds["pka_predicted"] = (5.0, 8.0)
        bounds["logD_pH6"] = (2.0, 3.5)
        missing_pka = df[df["pka_predicted"].isna()]
        missing_logd = df[df["logD_pH6"].isna()]
        pka_pass = apply_filters(missing_pka, {"pka_predicted": bounds["pka_predicted"]}, default_bounds)
        logd_pass = apply_filters(missing_logd, {"logD_pH6": bounds["logD_pH6"]}, default_bounds)
        if len(pka_pass) == 0 and len(logd_pass) == 0:
            sec.pass_("Narrowed pKa/logD filters exclude missing-data rows as expected")
        else:
            sec.fail(
                "Narrowed pKa/logD filters still pass rows with missing data",
                [f"pKa missing passed: {len(pka_pass)}", f"logD missing passed: {len(logd_pass)}"],
            )

    return sec


def section_6_consistency(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 6 — Cross-field logical consistency")
    df, today = ctx.df, ctx.today

    if {"physical_state", "melting_point_c"}.issubset(df.columns):
        gas_high_mp = df[
            df["physical_state"].astype(str).str.lower().eq("gas")
            & df["melting_point_c"].notna()
            & (df["melting_point_c"] > 25)
        ]
        if len(gas_high_mp):
            sec.warn(
                f"{len(gas_high_mp)} rows: physical_state=gas but melting point > 25°C",
                gas_high_mp[["name", "melting_point_c"]].head(5).astype(str).values.tolist(),
            )
        else:
            sec.pass_("No gas/ high melting point contradictions")

    if {"big_pharma_owned", "original_applicant_normalized"}.issubset(df.columns):
        bad = df[df["big_pharma_owned"].fillna(False).astype(bool) & df["original_applicant_normalized"].isna()]
        if len(bad):
            sec.fail(f"{len(bad)} rows: big_pharma_owned=True but no normalized applicant", bad["name"].head(10).tolist())
        else:
            sec.pass_("big_pharma_owned always has original_applicant_normalized")

    if "cns_target" in df.columns:
        cns_no_target = df[
            df["cns_target"].fillna(False).astype(bool)
            & df["mechanism_of_action"].isna()
            & df.get("target_names", pd.Series(index=df.index)).isna()
            & df.get("primary_target", pd.Series(index=df.index)).isna()
        ]
        if len(cns_no_target):
            sec.warn(
                f"{len(cns_no_target)} CNS-flagged rows without mechanism or target data",
                cns_no_target["name"].head(10).tolist(),
            )
        else:
            sec.pass_("CNS flags backed by mechanism or target data")

    if "earliest_patent_expiry" in df.columns:
        expiry = pd.to_datetime(df["earliest_patent_expiry"], errors="coerce")
        expired = df[expiry.notna() & (expiry.dt.date < today)]
        sec.info(f"Patents with expiry before today ({today}): {len(expired):,} (informational, not necessarily wrong)")
        implausible = df[expiry.notna() & ((expiry.dt.year < 2000) | (expiry.dt.year > 2050))]
        if len(implausible):
            sec.fail(f"{len(implausible)} implausible patent expiry dates", implausible[["name", "earliest_patent_expiry"]].head(10).astype(str).values.tolist())
        else:
            sec.pass_("Patent expiry dates within 2000–2050")

    if {"all_applicants", "applicant_count"}.issubset(df.columns):
        subset = df[df["all_applicants"].notna()].copy()
        subset["_expected_count"] = subset["all_applicants"].map(
            lambda v: len([p for p in str(v).split("|") if p.strip()])
        )
        count_mismatches = subset[subset["_expected_count"] != subset["applicant_count"].fillna(-1)]
        if len(count_mismatches):
            details = [
                f"{r['name']}: listed={int(r['_expected_count'])} count={r['applicant_count']}"
                for _, r in count_mismatches.head(10).iterrows()
            ]
            sec.fail(f"applicant_count mismatches ({len(count_mismatches)})", details)
        else:
            sec.pass_("applicant_count matches pipe-separated all_applicants entries")

    return sec


def section_7_conflicts(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 7 — Conflict tracking audit")
    df = ctx.df

    conflict_cols = [
        "mechanism_of_action_conflict",
        "indication_class_conflict",
        "mw_conflict",
    ]
    for col in conflict_cols:
        if col not in df.columns:
            continue
        nn = int(df[col].fillna(False).astype(bool).sum())
        sec.info(f"{col}: {nn:,}")

    if "mw_conflict" in df.columns:
        mw_conf = df[df["mw_conflict"].fillna(False).astype(bool) & df["molecular_weight"].notna() & df["molecular_weight_db"].notna()]
        salt_like = 0
        samples: list[str] = []
        for _, row in mw_conf.head(50).iterrows():
            diff = abs(float(row["molecular_weight_db"]) - float(row["molecular_weight"]))
            matched = any(abs(diff - mass) < 2.0 for mass in SALT_COUNTERION_MASSES.values())
            if matched:
                salt_like += 1
            if len(samples) < 10:
                samples.append(
                    f"{row['name']}: ChEMBL={row['molecular_weight']:.1f} DB={row['molecular_weight_db']:.1f} Δ={diff:.1f}"
                )
        if len(mw_conf):
            ratio = 100.0 * salt_like / min(len(mw_conf), 50)
            sec.info(f"MW conflict salt-form pattern (sample): {salt_like}/{min(len(mw_conf), 50)} ({ratio:.0f}%)")
            for s in samples:
                sec.info(f"    {s}")
            if ratio >= 50:
                sec.pass_("MW conflicts are mostly benign salt-form differences")
            else:
                sec.warn("MW conflicts may include non-salt discrepancies — review sample above")
        else:
            sec.pass_("No MW conflicts to audit")

    if "field_sources" in df.columns:
        populated = df[df["field_sources"].notna() & (df["field_sources"].astype(str).str.strip() != "")]
        bad_json: list[str] = []

        def _check_json(raw: Any) -> str | None:
            try:
                parsed = json.loads(str(raw))
                if not isinstance(parsed, dict):
                    return "not a dict"
            except json.JSONDecodeError:
                return "invalid JSON"
            return None

        for _, row in populated.iterrows():
            err = _check_json(row["field_sources"])
            if err:
                bad_json.append(f"{row['name']}: {err}")
            if len(bad_json) >= 10:
                break
        total_bad = int(populated["field_sources"].map(lambda v: _check_json(v) is not None).sum())
        if total_bad:
            sec.fail(f"Invalid field_sources JSON on {total_bad} rows", bad_json[:10])
        else:
            sec.pass_("field_sources JSON valid on all populated rows")

    return sec


def section_8_nosa_candidates(ctx: AuditContext) -> SectionResult:
    sec = SectionResult("Section 8 — NOSA candidate spot-check")
    df = ctx.df

    key_fields = [
        "molecular_weight", "logP", "PSA", "H_donors", "pka_predicted", "logD_pH6",
        "max_dose_mg", "dose_feasible_nosa", "original_applicant_normalized",
        "earliest_patent_expiry", "cns_target", "composite_score",
    ]

    ideal, ranges = active_score_config(df)
    scored = df.copy()
    scored["composite_score"] = composite_score(scored, ideal, ranges)

    missing_critical: list[str] = []
    for candidate in sorted(NOSA_CANDIDATES):
        rows = df[df["name"].str.lower().str.strip() == candidate]
        if rows.empty:
            sec.fail(f"Missing NOSA candidate: {candidate}")
            missing_critical.append(candidate)
            continue
        row = rows.iloc[0]
        score_row = scored.loc[row.name]
        sec.info(f"--- {row['name']} ---")
        summary_parts = []
        for field in key_fields:
            val = score_row.get(field, row.get(field))
            summary_parts.append(f"{field}={val}")
        sec.info("  " + " | ".join(summary_parts))
        flags = []
        if pd.isna(row.get("molecular_weight")) or pd.isna(row.get("logP")):
            flags.append("missing core physicochemistry")
        if pd.isna(row.get("max_dose_mg")):
            flags.append("missing dose")
        if pd.isna(row.get("original_applicant_normalized")):
            flags.append("missing patent owner")
        if not row.get("nosa_candidate"):
            flags.append("nosa_candidate flag not set")
        if flags:
            missing_critical.append(f"{candidate}: {', '.join(flags)}")

    if missing_critical:
        sec.warn(f"{len(missing_critical)} candidate issues", missing_critical)
    else:
        sec.pass_("All 10 NOSA candidates present with key properties")

    return sec


def section_9_coverage(ctx: AuditContext, prior_sections: list[SectionResult]) -> SectionResult:
    sec = SectionResult("Section 9 — Coverage summary table and recommendations")
    df, n = ctx.df, ctx.n

    important_cols = [
        "chembl_id", "drugbank_id", "SMILES", "pubchem_cid", "vapor_pressure_mmhg",
        "melting_point_c", "boiling_point_c", "pka_predicted", "logD_pH6", "max_dose_mg",
        "ct_max_dose_mg", "mechanism_of_action", "atc_code", "mesh_heading",
        "target_names", "cns_target", "earliest_patent_expiry", "original_applicant_normalized",
        "big_pharma_owned", "field_sources",
    ]

    sec.info(f"{'Column':<32} {'Count':>8} {'Coverage':>10}  Verdict")
    sec.info("-" * 70)
    for col in important_cols:
        nn, p = coverage(df, col)
        if p >= 80:
            verdict = "good"
        elif p >= 20:
            verdict = "partial"
        elif p > 0:
            verdict = "sparse"
        else:
            verdict = "empty"
        sec.info(f"{col:<32} {nn:>8,} {p:>9.1f}%  {verdict}")

    recs: list[tuple[int, str]] = []
    smiles_no_db = len(df[df["SMILES"].notna() & df["drugbank_id"].isna()])
    if smiles_no_db > 200:
        recs.append((smiles_no_db, f"InChIKey re-match: recover DrugBank for ~{smiles_no_db:,} SMILES rows missing name join"))
    ct_n, _ = coverage(df, "ct_max_dose_mg")
    dose_n, _ = coverage(df, "max_dose_mg")
    if dose_n < n * 0.4:
        recs.append((n - dose_n, f"Dose enrichment: ~{n - dose_n:,} rows still lack unified max_dose_mg"))
    vp_n, vp_pct = coverage(df, "vapor_pressure_mmhg")
    if vp_pct < 10:
        recs.append((n - vp_n, f"PubChem VP enrichment: vapor pressure only {vp_pct:.1f}% — rerun enrich_pubchem.py"))
    ob_n, ob_pct = coverage(df, "original_applicant_normalized")
    if ob_pct < 25:
        recs.append((n - ob_n, f"Orange Book name matching: ~{n - ob_n:,} rows may need better ingredient join"))
    for sec_prior in prior_sections:
        for text in sec_prior.recommendations:
            recs.append((500, text))

    seen_text: set[str] = set()
    unique_recs: list[tuple[int, str]] = []
    for impact, text in sorted(recs, key=lambda x: -x[0]):
        if text in seen_text:
            continue
        seen_text.add(text)
        unique_recs.append((impact, text))
    recs = unique_recs[:5]
    sec.info("")
    sec.info("Prioritized recommendations:")
    for i, (impact, text) in enumerate(recs, 1):
        sec.info(f"  {i}. [{impact:,} rows] {text}")

    return sec


def overall_verdict(sections: list[SectionResult]) -> Status:
    if any(s.status == "FAIL" for s in sections):
        return "CRITICAL ISSUES — see FAIL flags"
    if any(s.status == "WARN" for s in sections):
        return "MINOR ISSUES — see WARN flags"
    return "DATABASE HEALTHY"


def run_audit() -> tuple[str, list[SectionResult]]:
    ctx = load_audit_data()
    sections = [
        section_1_structural(ctx),
        section_2_source_utilization(ctx),
        section_3_join_quality(ctx),
        section_4_calculations(ctx),
        section_5_missing_data(ctx),
        section_6_consistency(ctx),
        section_7_conflicts(ctx),
        section_8_nosa_candidates(ctx),
    ]
    sections.append(section_9_coverage(ctx, sections))

    verdict = overall_verdict(sections)
    buf = io.StringIO()
    print("=" * 80, file=buf)
    print("NOSA-DB ADVANCED SANITY CHECK", file=buf)
    print("=" * 80, file=buf)
    print(f"OVERALL VERDICT: {verdict}", file=buf)
    print("", file=buf)
    for sec in sections:
        print(f"{sec.title} [{sec.status}]", file=buf)
        for line in sec.lines:
            print(line, file=buf)
        print("", file=buf)
    print("=" * 80, file=buf)
    return buf.getvalue(), sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced read-only NOSA database audit.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write full output to sanity_report.txt (does not modify database files).",
    )
    args = parser.parse_args()

    try:
        report, _ = run_audit()
    except Exception as exc:
        print(f"CRITICAL: audit failed to run: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(report)
    if args.report:
        out = ROOT / "sanity_report.txt"
        out.write_text(report, encoding="utf-8")
        print(f"Report written to {out.name}")


if __name__ == "__main__":
    main()
