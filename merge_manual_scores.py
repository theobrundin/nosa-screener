#!/usr/bin/env python3
"""Merge manually curated screening scores into the NOSA drug database.

Re-runnable: overwrites manual_score / manual_score_comment on each run.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# Swap this path when the reviewer sends an updated scores file.
MANUAL_SCORES_CSV = ROOT / "nosa_screening_results_with_atc_and_target.csv"

DATABASE_CSV = ROOT / "nosa_drug_database.csv"
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"

SCORE_COL = "Score"
COMMENT_COL = "Score comment"
MANUAL_SCORE = "manual_score"
MANUAL_COMMENT = "manual_score_comment"
MANUAL_EVAL_SET = "manual_eval_set"

# Carry through only if the target DB is missing the column.
OPTIONAL_CARRY_COLUMNS = [
    "cns_target",
    "primary_target",
    "nasal_cyp_risk",
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
    "indication_class",
]


def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.upper().strip()
    return re.sub(r"\s+", " ", s)


def is_valid_chembl_id(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "—", "-", "–"}:
        return False
    # Reject latin1 mangled punctuation (e.g. \x97 em-dash).
    if any(ord(ch) > 127 for ch in text):
        return False
    return text.upper().startswith("CHEMBL")


def load_manual_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manual scores file not found: {path}")

    scores = pd.read_csv(path, sep=";", encoding="latin1")
    if SCORE_COL not in scores.columns or COMMENT_COL not in scores.columns:
        raise ValueError(
            f"Expected columns '{SCORE_COL}' and '{COMMENT_COL}' in {path.name}; "
            f"got {list(scores.columns)}"
        )
    if "name" not in scores.columns:
        raise ValueError(f"Expected 'name' column in {path.name}")

    scores = scores.copy()
    scores[MANUAL_SCORE] = pd.to_numeric(scores[SCORE_COL], errors="coerce").astype("Int64")
    scores[MANUAL_COMMENT] = scores[COMMENT_COL].where(scores[COMMENT_COL].notna(), pd.NA)
    scores[MANUAL_COMMENT] = scores[MANUAL_COMMENT].astype("string")
    scores["_name_key"] = scores["name"].map(normalize_name)

    if "chembl_id" in scores.columns:
        scores["_chembl_key"] = scores["chembl_id"].map(
            lambda v: str(v).strip().upper() if is_valid_chembl_id(v) else pd.NA
        )
    else:
        scores["_chembl_key"] = pd.Series([pd.NA] * len(scores), dtype="string")

    keep = [MANUAL_SCORE, MANUAL_COMMENT, "_name_key", "_chembl_key", "name"]
    for col in OPTIONAL_CARRY_COLUMNS:
        if col in scores.columns and col not in keep:
            keep.append(col)

    return scores[keep].drop_duplicates(subset=["_name_key"], keep="first")


def merge_scores_into_db(db: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = db.copy()

    # Clean prior merge artifacts so re-runs overwrite cleanly.
    drop_cols = [
        c
        for c in out.columns
        if c in {MANUAL_SCORE, MANUAL_COMMENT, MANUAL_EVAL_SET}
        or c.startswith(f"{MANUAL_SCORE}_")
        or c.startswith(f"{MANUAL_COMMENT}_")
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out["_name_key"] = out["name"].map(normalize_name)
    out["_row_id"] = range(len(out))
    out[MANUAL_EVAL_SET] = False

    matched_ids: set[int] = set()
    unmatched_names: list[str] = []

    # Preferred join: chembl_id
    if "chembl_id" in out.columns:
        out["_chembl_key"] = out["chembl_id"].map(
            lambda v: str(v).strip().upper() if is_valid_chembl_id(v) else pd.NA
        )
        by_chembl = scores.dropna(subset=["_chembl_key"]).drop_duplicates(
            subset=["_chembl_key"], keep="first"
        )
        chembl_map = by_chembl.set_index("_chembl_key")
        for chembl_key, row in chembl_map.iterrows():
            hits = out.index[out["_chembl_key"] == chembl_key]
            for idx in hits:
                matched_ids.add(int(out.at[idx, "_row_id"]))
                out.at[idx, MANUAL_SCORE] = row[MANUAL_SCORE]
                out.at[idx, MANUAL_COMMENT] = row[MANUAL_COMMENT]
                out.at[idx, MANUAL_EVAL_SET] = True
    else:
        out["_chembl_key"] = pd.NA

    # Fallback join: normalized name for remaining score rows
    already_matched_keys = set()
    if matched_ids:
        already_matched_keys = set(
            out.loc[out["_row_id"].isin(matched_ids), "_chembl_key"].dropna().tolist()
        )

    for _, row in scores.iterrows():
        chembl_key = row.get("_chembl_key")
        if pd.notna(chembl_key) and chembl_key in already_matched_keys:
            continue
        name_key = row["_name_key"]
        hits = out.index[out["_name_key"] == name_key]
        if len(hits) == 0:
            unmatched_names.append(str(row["name"]))
            continue
        for idx in hits:
            rid = int(out.at[idx, "_row_id"])
            if rid in matched_ids:
                continue
            matched_ids.add(rid)
            out.at[idx, MANUAL_SCORE] = row[MANUAL_SCORE]
            out.at[idx, MANUAL_COMMENT] = row[MANUAL_COMMENT]
            out.at[idx, MANUAL_EVAL_SET] = True

    # Ensure dtypes after assignment
    out[MANUAL_SCORE] = pd.to_numeric(out.get(MANUAL_SCORE), errors="coerce").astype("Int64")
    if MANUAL_COMMENT not in out.columns:
        out[MANUAL_COMMENT] = pd.Series([pd.NA] * len(out), dtype="string")
    else:
        out[MANUAL_COMMENT] = out[MANUAL_COMMENT].astype("string")
    out[MANUAL_EVAL_SET] = out[MANUAL_EVAL_SET].fillna(False).astype(bool)

    # Fill missing DB columns from the screening file (only where absent).
    carried: list[str] = []
    score_by_name = scores.drop_duplicates(subset=["_name_key"], keep="first").set_index("_name_key")
    for col in OPTIONAL_CARRY_COLUMNS:
        if col not in scores.columns:
            continue
        if col in out.columns:
            continue
        out[col] = out["_name_key"].map(score_by_name[col])
        carried.append(col)

    report = {
        "score_rows": len(scores),
        "matched": len(matched_ids),
        "unmatched": len(unmatched_names),
        "unmatched_names": unmatched_names,
        "carried_columns": carried,
        "evaluated": int(out[MANUAL_SCORE].notna().sum()),
        "pass_count": int((out[MANUAL_SCORE] == 1).sum()),
        "fail_count": int((out[MANUAL_SCORE] == 0).sum()),
        "pending": int(((out[MANUAL_EVAL_SET]) & out[MANUAL_SCORE].isna()).sum()),
    }

    out = out.drop(columns=["_name_key", "_chembl_key", "_row_id"], errors="ignore")
    return out, report


def print_report(target: Path, report: dict) -> None:
    print(f"\n=== Manual score merge → {target.name} ===")
    print(f"  Score file rows:     {report['score_rows']}")
    print(f"  Rows matched:        {report['matched']}")
    print(f"  Rows unmatched:      {report['unmatched']}")
    if report["unmatched_names"]:
        print("  Unmatched molecules:")
        for name in report["unmatched_names"]:
            print(f"    · {name}")
    if report["carried_columns"]:
        print(f"  New columns carried: {', '.join(report['carried_columns'])}")
    else:
        print("  New columns carried: (none — DB already had optional fields)")
    print(
        f"  In DB after merge:   evaluated={report['evaluated']} "
        f"pass={report['pass_count']} fail={report['fail_count']} "
        f"pending={report['pending']}"
    )


def merge_into_file(db_path: Path, scores: pd.DataFrame) -> dict:
    db = pd.read_csv(db_path, low_memory=False)
    merged, report = merge_scores_into_db(db, scores)
    merged.to_csv(db_path, index=False)
    print_report(db_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=MANUAL_SCORES_CSV,
        help="Path to semicolon/latin1 screening scores CSV",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_CSV,
        help="Master database CSV to update",
    )
    parser.add_argument(
        "--skip-enriched",
        action="store_true",
        help="Do not also update nosa_drug_database_enriched.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_manual_scores(args.scores)
    print(f"Loaded {len(scores)} manual score rows from {args.scores.name}")

    if not args.database.exists():
        raise FileNotFoundError(f"Database not found: {args.database}")
    merge_into_file(args.database, scores)

    if not args.skip_enriched and ENRICHED_CSV.exists() and ENRICHED_CSV.resolve() != args.database.resolve():
        merge_into_file(ENRICHED_CSV, scores)


if __name__ == "__main__":
    main()
