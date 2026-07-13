#!/usr/bin/env python3
"""Enrich NOSA drug database with maximum clinical doses from ClinicalTrials.gov."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "nosa_drug_database.csv"
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"

from structure_match import recompute_dose_feasible_nosa, sync_unified_dose  # noqa: E402

CT_API = "https://clinicaltrials.gov/api/v2/studies"
CT_FIELDS = "protocolSection.identificationModule,protocolSection.armsInterventionsModule"
PAGE_SIZE = 20

REQUEST_SLEEP_S = 0.3
MAX_RETRIES = 3
SAVE_EVERY = 500
PROGRESS_EVERY = 100
NOSA_DOSE_LIMIT_MG = 100.0
MAX_PLAUSIBLE_DOSE_MG = 2000.0
MIN_REVIEW_DOSE_MG = 0.001

DOSE_COLUMNS = [
    "ct_max_dose_mg",
    "ct_max_dose_raw",
    "ct_dose_unit",
    "ct_n_studies",
    "ct_dose_feasible",
    "max_dose_mg",
    "max_dose_source",
    "dose_feasible_nosa",
]

# mg / g / µg variants; exclude mg/kg, mg/mL concentrations, and weight-normalized doses
DOSE_RE = re.compile(
    r"(?<![/\d])(\d+(?:\.\d+)?)\s*"
    r"(mg|milligram(?:s)?|[μµ]g|ug|mcg|microgram(?:s)?|(?<![a-z])g(?![a-z/])|gram(?:s)?)"
    r"(?!\s*/\s*(?:kg|m[lL]|d|day|daily|dose))",
    re.IGNORECASE,
)

PREFERRED_DOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|milligram(?:s)?|[μµ]g|ug|mcg|microgram(?:s)?|(?<![a-z])g(?![a-z/])|gram(?:s)?)"
    r"\s*(?:once\s+daily|per\s+day|/day|daily|bid|tid|qid|q\d+h|every\s+\d+\s+hours?)?",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich NOSA drug database with ClinicalTrials.gov dose data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: overwrite --input in place)",
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Only enrich NOSA candidates (nosa_candidate=True)",
    )
    return parser.parse_args()


def request_json(url: str, session: requests.Session) -> dict[str, Any] | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=60)
            time.sleep(REQUEST_SLEEP_S)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            if attempt == MAX_RETRIES:
                print(f"  HTTP error after {MAX_RETRIES} tries: {url}", file=sys.stderr)
                return None
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"  Request failed after {MAX_RETRIES} tries: {exc}", file=sys.stderr)
                return None
            time.sleep(REQUEST_SLEEP_S * attempt)
    return None


def search_studies(drug_name: str, query_param: str, session: requests.Session) -> list[dict[str, Any]]:
    encoded = quote(str(drug_name), safe="")
    url = (
        f"{CT_API}?{query_param}={encoded}"
        f"&fields={CT_FIELDS}&pageSize={PAGE_SIZE}&format=json"
    )
    data = request_json(url, session)
    if not data:
        return []
    return data.get("studies") or []


def normalize_drug_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def intervention_matches(drug_name: str, intervention: dict[str, Any]) -> bool:
    drug_norm = normalize_drug_name(drug_name)
    if not drug_norm:
        return False
    candidates = [str(intervention.get("name") or "")]
    candidates.extend(str(n) for n in (intervention.get("otherNames") or []))
    for candidate in candidates:
        cand_norm = normalize_drug_name(candidate)
        if not cand_norm:
            continue
        if cand_norm == drug_norm or drug_norm in cand_norm or cand_norm in drug_norm:
            return True
    return False


def arm_matches_drug(drug_name: str, arm: dict[str, Any]) -> bool:
    drug_lower = str(drug_name).lower()
    label = str(arm.get("label") or "").lower()
    if drug_lower in label or normalize_drug_name(drug_name) in normalize_drug_name(label):
        return True
    for name in arm.get("interventionNames") or []:
        if drug_lower in str(name).lower():
            return True
    return False


def extract_study_texts(study: dict[str, Any], drug_name: str) -> list[str]:
    texts: list[str] = []
    module = study.get("protocolSection", {}).get("armsInterventionsModule") or {}
    matched_arms: set[str] = set()

    for intr in module.get("interventions") or []:
        if not intervention_matches(drug_name, intr):
            continue
        for key in ("name", "description", "dosageForm"):
            value = intr.get(key)
            if value:
                texts.append(str(value))
        for name in intr.get("otherNames") or []:
            texts.append(str(name))
        for label in intr.get("armGroupLabels") or []:
            matched_arms.add(str(label))

    for arm in module.get("armGroups") or []:
        if arm_matches_drug(drug_name, arm) or str(arm.get("label") or "") in matched_arms:
            for key in ("label", "description"):
                value = arm.get(key)
                if value:
                    texts.append(str(value))

    return texts


def study_nct_id(study: dict[str, Any]) -> str:
    return (
        study.get("protocolSection", {})
        .get("identificationModule", {})
        .get("nctId")
        or ""
    )


def normalize_unit(unit: str) -> tuple[str, float]:
    u = unit.lower().strip().replace("μ", "µ")
    if u in {"µg", "ug", "mcg"} or u.startswith("microgram"):
        return "µg", 0.001
    if u in {"g", "gram", "grams"}:
        return "g", 1000.0
    return "mg", 1.0


def cap_dose_mg(dose_mg: float | None) -> tuple[float | None, bool]:
    """Return capped dose and whether it was rejected as implausible."""
    if dose_mg is None:
        return None, False
    if dose_mg > MAX_PLAUSIBLE_DOSE_MG:
        return None, True
    return dose_mg, False


def extract_doses_from_text(text: str, drug_name: str) -> list[tuple[float, str, str]]:
    """Return (dose_mg, original_unit, snippet) for dose mentions tied to this drug."""
    results: list[tuple[float, str, str]] = []
    drug_tokens = {
        t for t in re.split(r"[\s\-/]+", str(drug_name).lower()) if len(t) >= 4
    }
    drug_norm = normalize_drug_name(drug_name)
    lower = text.lower()

    # Skip concentration-only snippets (mg/mL, mg/ml)
    if re.search(r"\d+(?:\.\d+)?\s*mg\s*/\s*ml", lower):
        if not re.search(r"(?:once daily|per day|/day|daily|bid|tid|qid)", lower):
            return results

    patterns = [PREFERRED_DOSE_RE, DOSE_RE]
    seen_spans: set[tuple[int, int]] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            value = float(match.group(1))
            unit_label, factor = normalize_unit(match.group(2))
            dose_mg = value * factor
            dose_mg, rejected = cap_dose_mg(dose_mg)
            if rejected:
                continue
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            snippet = text[start:end].strip()
            snippet_norm = normalize_drug_name(snippet)
            if drug_norm not in snippet_norm and not any(tok in snippet.lower() for tok in drug_tokens):
                continue
            results.append((dose_mg, unit_label, snippet))
    return results


def enrich_drug(drug_name: str, session: requests.Session) -> dict[str, Any]:
    studies_by_nct: dict[str, dict[str, Any]] = {}
    for query_param in ("query.term", "query.intr"):
        for study in search_studies(drug_name, query_param, session):
            nct = study_nct_id(study) or str(id(study))
            studies_by_nct[nct] = study

    matching_studies = 0
    best_mg: float | None = None
    best_unit: str | None = None
    best_raw: str | None = None

    for study in studies_by_nct.values():
        texts = extract_study_texts(study, drug_name)
        if not texts:
            continue
        matching_studies += 1
        for text in texts:
            for dose_mg, unit, snippet in extract_doses_from_text(text, drug_name):
                if best_mg is None or dose_mg > best_mg:
                    best_mg = dose_mg
                    best_unit = unit
                    best_raw = snippet

    ct_feasible: bool | None = None
    if best_mg is not None:
        best_mg, rejected = cap_dose_mg(best_mg)
        if rejected:
            best_mg = None
            best_raw = f"REJECTED>{MAX_PLAUSIBLE_DOSE_MG}mg: {best_raw}"
        elif best_mg is not None and best_mg < MIN_REVIEW_DOSE_MG:
            best_raw = f"REVIEW_UG? {best_raw}"
    if best_mg is not None:
        ct_feasible = best_mg <= NOSA_DOSE_LIMIT_MG

    return {
        "ct_max_dose_mg": best_mg,
        "ct_max_dose_raw": best_raw,
        "ct_dose_unit": best_unit,
        "ct_n_studies": matching_studies,
        "ct_dose_feasible": ct_feasible,
    }


def apply_unified_dose(row: pd.Series) -> pd.Series:
    return sync_unified_dose(row)


def finalize_all_doses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(sync_unified_dose, axis=1)
    return recompute_dose_feasible_nosa(df)


def already_searched(row: pd.Series) -> bool:
    """Resume: skip rows already queried (ct_n_studies set)."""
    if "ct_n_studies" in row.index and pd.notna(row.get("ct_n_studies")):
        return True
    return False


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    string_cols = {"ct_max_dose_raw", "ct_dose_unit", "max_dose_source"}
    bool_cols = {"ct_dose_feasible", "dose_feasible_nosa"}
    for col in DOSE_COLUMNS:
        if col not in df.columns:
            if col in string_cols:
                df[col] = pd.Series([None] * len(df), dtype="object")
            elif col in bool_cols:
                df[col] = pd.NA
            else:
                df[col] = None
    return df


def merge_dose_columns(source_df: pd.DataFrame, target_path: Path) -> None:
    if not target_path.exists():
        return
    target = pd.read_csv(target_path, low_memory=False)
    dose_cols = [c for c in DOSE_COLUMNS if c in source_df.columns]
    target = target.drop(columns=[c for c in dose_cols if c in target.columns], errors="ignore")
    patch = source_df[["name"] + dose_cols]
    merged = target.merge(patch, on="name", how="left")
    merged.to_csv(target_path, index=False)
    print(f"  Also updated dose columns in {target_path.name}")


def print_progress(processed: int, total: int, doses_found: int, studies_total: int) -> None:
    pct = 100.0 * processed / total if total else 0.0
    print(
        f"Progress: {processed}/{total} ({pct:.1f}%) | "
        f"doses found: {doses_found} | studies: {studies_total:,}"
    )


def print_summary(df: pd.DataFrame, searched: int, studies_total: int, output_path: Path) -> None:
    n = len(df)
    extracted = df["ct_max_dose_mg"].notna().sum()
    le100 = (df["ct_max_dose_mg"].notna() & (df["ct_max_dose_mg"] <= NOSA_DOSE_LIMIT_MG)).sum()
    gt100 = (df["ct_max_dose_mg"].notna() & (df["ct_max_dose_mg"] > NOSA_DOSE_LIMIT_MG)).sum()
    unknown = n - extracted
    pct = 100.0 * extracted / searched if searched else 0.0

    unified = df["max_dose_mg"].notna().sum()
    chembl = df["max_dose_source"].eq("chembl").sum() if "max_dose_source" in df.columns else 0
    ct = df["max_dose_source"].eq("clinicaltrials").sum() if "max_dose_source" in df.columns else 0

    print("\n── ClinicalTrials.gov dose enrichment ────────────")
    print(f"  Drugs searched:              {searched:>7,}")
    print(f"  Studies found (any):         {studies_total:>7,}")
    print(f"  Dose data extracted:         {extracted:>7,}  ({pct:.1f}%)")
    print(f"  Dose ≤ 100mg:                {le100:>7,}")
    print(f"  Dose > 100mg:                {gt100:>7,}")
    print(f"  Dose unknown:                {unknown:>7,}")
    print(f"  Unified max_dose_mg:         {unified:>7,}  (ChEMBL: {chembl:,}, CT.gov: {ct:,})")
    print("  ─────────────────────────────────────────────────")
    print(f"  Output → {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = ensure_columns(pd.read_csv(args.input, low_memory=False))

    if args.priority_only:
        if "nosa_candidate" not in df.columns:
            print("No nosa_candidate column — nothing to enrich in priority mode.", file=sys.stderr)
            sys.exit(1)
        work_mask = df["nosa_candidate"].fillna(False).astype(bool)
    else:
        work_mask = pd.Series(True, index=df.index)

    resume_mask = df.apply(already_searched, axis=1)
    to_process = df[work_mask & ~resume_mask].copy()
    skipped = int(work_mask.sum() - len(to_process)) if args.priority_only else int(resume_mask.sum())

    if skipped:
        print(f"Resuming: skipping {skipped:,} rows already searched")

    session = requests.Session()
    session.headers.update({"User-Agent": "NOSA-DB/1.0 (clinical dose enrichment)"})

    processed = 0
    doses_found = int(df.loc[work_mask, "ct_max_dose_mg"].notna().sum())
    studies_total = int(df.loc[work_mask, "ct_n_studies"].fillna(0).sum())

    total_to_run = len(to_process)

    for idx, row in to_process.iterrows():
        name = str(row["name"])
        result = enrich_drug(name, session)

        for col, val in result.items():
            df.at[idx, col] = val

        processed += 1
        studies_total += int(result["ct_n_studies"] or 0)
        if result["ct_max_dose_mg"] is not None:
            doses_found += 1

        if processed % SAVE_EVERY == 0:
            df = finalize_all_doses(df)
            df.to_csv(output_path, index=False)
            print(f"  [checkpoint] saved after {processed} new lookups")

        if processed % PROGRESS_EVERY == 0:
            print_progress(processed, total_to_run, doses_found, studies_total)

    df = finalize_all_doses(df)
    rejected = int(df["ct_max_dose_raw"].astype(str).str.contains("REJECTED>", na=False).sum()) if "ct_max_dose_raw" in df.columns else 0
    review = int(df["ct_max_dose_raw"].astype(str).str.contains("REVIEW_UG", na=False).sum()) if "ct_max_dose_raw" in df.columns else 0
    if rejected:
        print(f"  Implausible CT doses rejected (>{MAX_PLAUSIBLE_DOSE_MG} mg): {rejected:,}")
    if review:
        print(f"  Sub-{MIN_REVIEW_DOSE_MG} mg values flagged for review: {review:,}")
    df.to_csv(output_path, index=False)

    if output_path.resolve() != ENRICHED_CSV.resolve() and ENRICHED_CSV.exists():
        merge_dose_columns(df, ENRICHED_CSV)

    searched = int(work_mask.sum())
    print_summary(df.loc[work_mask] if args.priority_only else df, searched, studies_total, output_path)


if __name__ == "__main__":
    main()
