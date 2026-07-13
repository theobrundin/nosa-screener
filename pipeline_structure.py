#!/usr/bin/env python3
"""Structure-first pipeline merges for build_database.py."""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from structure_match import (
    StructureIndex,
    add_structure_columns,
    compute_structure_keys,
    load_inchikey_cache,
    load_pubchem_name_cache,
    resolve_pubchem_structure,
    save_inchikey_cache,
    save_pubchem_name_cache,
)


def enrich_drugbank_structures(
    drugbank: pd.DataFrame,
    drugcentral: pd.DataFrame,
    pubchem_cache: dict[str, dict[str, Any]],
    inchi_cache: dict[str, tuple[str | None, str | None]],
) -> pd.DataFrame:
    """Attach InChIKeys to DrugBank rows via DrugCentral then PubChem name cache."""
    if drugbank.empty:
        return drugbank

    dc = drugcentral.drop_duplicates("name_key").set_index("name_key")
    rows: list[dict[str, Any]] = []
    for _, db_row in drugbank.iterrows():
        rec = db_row.to_dict()
        nk = rec.get("name_key")
        ik = sk = None
        if nk and nk in dc.index:
            ik = dc.loc[nk, "inchikey"] if "inchikey" in dc.columns else None
            sk = dc.loc[nk, "inchikey_skeleton"] if "inchikey_skeleton" in dc.columns else None
        if not ik:
            name = str(rec.get("name", "")).strip().lower()
            pc = pubchem_cache.get(name) or pubchem_cache.get(str(nk))
            if pc:
                ik = pc.get("inchikey")
                sk = pc.get("inchikey_skeleton")
        if not ik and rec.get("smiles"):
            ik, sk = compute_structure_keys(smiles=rec.get("smiles"), cache=inchi_cache)
        rec["inchikey"] = ik
        rec["inchikey_skeleton"] = sk
        rows.append(rec)
    return pd.DataFrame(rows)


def build_drugbank_index(
    drugbank: pd.DataFrame,
    drugcentral: pd.DataFrame,
    pubchem_cache: dict[str, dict[str, Any]],
    inchi_cache: dict[str, tuple[str | None, str | None]],
) -> tuple[StructureIndex, pd.DataFrame]:
    enriched = enrich_drugbank_structures(drugbank, drugcentral, pubchem_cache, inchi_cache)
    index = StructureIndex.from_dataframe(enriched)
    return index, enriched


def lookup_drugbank_payload(index: StructureIndex, row: pd.Series) -> tuple[dict[str, Any] | None, str | None]:
    payload, method = index.lookup(row)
    if payload is None:
        return None, None
    if isinstance(payload, pd.Series):
        return payload.to_dict(), method
    return dict(payload), method


def deduplicate_with_structure(
    df: pd.DataFrame,
    normalize_name_fn,
) -> tuple[pd.DataFrame, int, list[str], list[str]]:
    """Deduplicate by name_key, merge exact InChIKey duplicates, link skeleton-related forms."""
    before = len(df)
    work = df.copy()
    work["_priority"] = work["source"].map({"chembl": 0, "manual": 1}).fillna(2)
    work["_phase_sort"] = pd.to_numeric(work["chembl_max_phase"], errors="coerce").fillna(-1)
    work = work.sort_values(["_priority", "_phase_sort"], ascending=[True, False])

    merges: list[str] = []
    if "inchikey" in work.columns:
        for ik, group in work[work["inchikey"].notna()].groupby("inchikey"):
            if len(group) <= 1:
                continue
            keep_name = group.iloc[0]["name"]
            drop_names = group.iloc[1:]["name"].tolist()
            merges.append(f"{keep_name} ← merged {', '.join(drop_names)} (same InChIKey)")
            work = work.drop(index=group.index[1:])

    work["related_forms"] = None
    relations: list[str] = []
    if "inchikey_skeleton" in work.columns:
        by_sk: dict[str, list[int]] = {}
        for idx, row in work.iterrows():
            sk = row.get("inchikey_skeleton")
            if sk and str(sk).strip():
                by_sk.setdefault(str(sk), []).append(idx)
        for sk, indices in by_sk.items():
            if len(indices) <= 1:
                continue
            names = [str(work.at[i, "name"]) for i in indices]
            full_keys = {str(work.at[i, "inchikey"]) for i in indices if pd.notna(work.at[i, "inchikey"])}
            if len(full_keys) <= 1:
                continue
            for i in indices:
                others = [n for j, n in zip(indices, names) if j != i]
                if others:
                    work.at[i, "related_forms"] = "|".join(sorted(set(others)))
            relations.append(f"{names[0]} ↔ {names[1]} (+{len(names)-2}) skeleton {sk}")

    work = work.drop_duplicates(subset="name_key", keep="first")
    work = work.drop(columns=["_priority", "_phase_sort"], errors="ignore")
    return work, before - len(work), merges[:30], relations[:30]


def fill_smiles_from_drugcentral_tiered(
    df: pd.DataFrame,
    drugcentral: pd.DataFrame,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Fill SMILES using InChIKey → skeleton → name tier matching."""
    df = df.copy()
    if "drugcentral_match_method" not in df.columns:
        df["drugcentral_match_method"] = None

    dc = drugcentral.drop_duplicates("name_key").copy()
    dc_index = StructureIndex.from_dataframe(dc)
    filled = 0
    tier_counts = {"inchikey": 0, "skeleton": 0, "name": 0}

    for idx, row in df[df["SMILES"].isna()].iterrows():
        payload, method = dc_index.lookup(row)
        if payload is None:
            continue
        if isinstance(payload, pd.Series):
            smiles = payload.get("smiles")
        else:
            smiles = payload.get("smiles")
        if not smiles or not str(smiles).strip():
            continue
        df.at[idx, "SMILES"] = smiles
        df.at[idx, "drugcentral_match_method"] = method
        df.at[idx, "smiles_source"] = "drugcentral"
        df.at[idx, "sources"] = df.at[idx, "sources"] if pd.notna(df.at[idx, "sources"]) else "drugcentral"
        if pd.notna(df.at[idx, "sources"]) and "drugcentral" not in str(df.at[idx, "sources"]):
            df.at[idx, "sources"] = f"{df.at[idx, 'sources']}|drugcentral"
        if method:
            tier_counts[method] = tier_counts.get(method, 0) + 1
        filled += 1

    return df, filled, tier_counts


def merge_orange_book_extended(
    df: pd.DataFrame,
    orange_book: pd.DataFrame,
    pubchem_cache: dict[str, dict[str, Any]],
    inchi_cache: dict[str, tuple[str, str | None] | tuple[str | None, str | None]],
    *,
    allow_pubchem_api: bool = False,
    session=None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Name join first, then skeleton match via PubChem-resolved Orange Book ingredient names."""
    stats = {"name": 0, "skeleton": 0, "pubchem_resolved": 0}
    ob_cols = [
        c
        for c in orange_book.columns
        if c
        not in {"name_key", "ingredient"}
    ]

    merged = df.merge(
        orange_book,
        on="name_key",
        how="left",
        suffixes=("", "_ob_dup"),
    )
    if "orange_book_match_method" not in merged.columns:
        merged["orange_book_match_method"] = None

    name_matched = merged["original_applicant"].notna() if "original_applicant" in merged.columns else pd.Series(False, index=merged.index)
    merged.loc[name_matched, "orange_book_match_method"] = "name"
    stats["name"] = int(name_matched.sum())

    master_index = StructureIndex.from_dataframe(
        merged[merged["inchikey_skeleton"].notna()].copy()
        if "inchikey_skeleton" in merged.columns
        else merged.iloc[0:0]
    )

    unmatched_ob = orange_book[~orange_book["name_key"].isin(set(df["name_key"]))].copy()
    for _, ob_row in unmatched_ob.iterrows():
        ingredient = ob_row.get("ingredient") or ob_row.get("name_key")
        if not ingredient:
            continue
        name_key = str(ob_row.get("name_key", ""))
        pc = pubchem_cache.get(str(ingredient).strip().lower()) or pubchem_cache.get(name_key)
        if not pc and allow_pubchem_api:
            pc = resolve_pubchem_structure(str(ingredient), session=session, cache=pubchem_cache)
            if pc.get("inchikey"):
                stats["pubchem_resolved"] += 1
        sk = pc.get("inchikey_skeleton") if pc else None
        if not sk:
            continue
        # Find master rows with this skeleton lacking OB data
        mask = (
            merged["inchikey_skeleton"].astype(str).eq(str(sk))
            & ~merged["orange_book_match_method"].astype(str).eq("name")
        )
        if "original_applicant" in merged.columns:
            mask &= merged["original_applicant"].isna()
        if not mask.any():
            continue
        for col in ob_cols:
            if col in ob_row.index and col in merged.columns:
                merged.loc[mask, col] = ob_row[col]
        merged.loc[mask, "orange_book_match_method"] = "skeleton"
        stats["skeleton"] += int(mask.sum())

    return merged, stats


def print_match_gain_table(
    label: str,
    before_pct: float,
    after_count: int,
    total: int,
    tier_counts: dict[str, int] | None = None,
) -> None:
    after_pct = 100.0 * after_count / total if total else 0.0
    gain = after_count - int(before_pct / 100.0 * total)
    print(f"  {label:<14} {before_pct:>6.1f}%   {after_pct:>6.1f}%   +{max(gain, 0):,} rows")
    if tier_counts:
        parts = ", ".join(f"{k}={v:,}" for k, v in sorted(tier_counts.items()) if v)
        if parts:
            print(f"    tiers: {parts}")


def prepare_structure_caches() -> tuple[dict, dict]:
    inchi_cache = load_inchikey_cache()
    pubchem_cache = load_pubchem_name_cache()
    return inchi_cache, pubchem_cache


def finalize_structure_caches(
    inchi_cache: dict[str, tuple[str | None, str | None]],
    pubchem_cache: dict[str, dict[str, Any]],
) -> None:
    save_inchikey_cache(inchi_cache)
    save_pubchem_name_cache(pubchem_cache)


def seed_pubchem_cache_from_dataframe(df: pd.DataFrame, pubchem_cache: dict[str, dict[str, Any]]) -> None:
    """Seed name→structure cache from dataframe names, synonyms, and keys."""
    for _, row in df.iterrows():
        ik = row.get("inchikey")
        sk = row.get("inchikey_skeleton")
        smi = row.get("SMILES")
        if not ik:
            continue
        payload = {
            "cid": row.get("pubchem_cid"),
            "smiles": smi,
            "inchikey": ik,
            "inchikey_skeleton": sk,
        }
        for label in [row.get("name"), row.get("name_key"), row.get("inn")]:
            if label and str(label).strip():
                pubchem_cache.setdefault(str(label).strip().lower(), payload)
        syns = row.get("synonyms")
        if pd.notna(syns):
            for part in str(syns).split("|"):
                part = part.strip()
                if part:
                    pubchem_cache.setdefault(part.lower(), payload)


def seed_pubchem_cache_from_drugcentral(drugcentral: pd.DataFrame, pubchem_cache: dict[str, dict[str, Any]]) -> None:
    for _, row in drugcentral.iterrows():
        inn = str(row.get("inn", "")).strip().lower()
        if not inn:
            continue
        pubchem_cache.setdefault(
            inn,
            {
                "cid": None,
                "smiles": row.get("smiles"),
                "inchikey": row.get("inchikey"),
                "inchikey_skeleton": row.get("inchikey_skeleton"),
            },
        )


def count_smiles_no_drugbank(df: pd.DataFrame) -> int:
    if "drugbank_id" not in df.columns or "SMILES" not in df.columns:
        return 0
    return int(df["SMILES"].notna().sum() - df.loc[df["SMILES"].notna(), "drugbank_id"].notna().sum())
