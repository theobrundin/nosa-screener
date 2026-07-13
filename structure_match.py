#!/usr/bin/env python3
"""InChIKey-based structure matching helpers for the NOSA pipeline."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
INCHIKEY_CACHE = ROOT / "inchikey_cache.pkl"
PUBCHEM_NAME_CACHE = ROOT / "pubchem_name_cache.json"
PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

MatchMethod = str  # inchikey | skeleton | name


def inchikey_skeleton(inchikey: str | None) -> str | None:
    if not inchikey or not isinstance(inchikey, str):
        return None
    text = inchikey.strip().upper()
    if not text:
        return None
    return text.split("-", 1)[0]


def _cache_get(cache: dict[str, tuple[str | None, str | None]], key: str) -> tuple[str | None, str | None]:
    return cache.get(key, (None, None))


def compute_structure_keys(
    smiles: str | None = None,
    inchi: str | None = None,
    inchikey: str | None = None,
    cache: dict[str, tuple[str | None, str | None]] | None = None,
) -> tuple[str | None, str | None]:
    """Return (full InChIKey, skeleton) from SMILES, InChI, or precomputed InChIKey."""
    if inchikey and str(inchikey).strip():
        full = str(inchikey).strip().upper()
        return full, inchikey_skeleton(full)

    cache_key = None
    if smiles and str(smiles).strip():
        cache_key = f"SMILES:{str(smiles).strip()}"
    elif inchi and str(inchi).strip():
        cache_key = f"InChI:{str(inchi).strip()}"

    if cache is not None and cache_key:
        cached = _cache_get(cache, cache_key)
        if cached[0]:
            return cached

    mol = None
    if smiles and str(smiles).strip():
        mol = Chem.MolFromSmiles(str(smiles).strip())
    elif inchi and str(inchi).strip():
        mol = Chem.MolFromInchi(str(inchi).strip())

    if mol is None:
        result = (None, None)
    else:
        try:
            full = Chem.MolToInchiKey(mol)
            result = (full, inchikey_skeleton(full))
        except Exception:
            result = (None, None)

    if cache is not None and cache_key:
        cache[cache_key] = result
    return result


def load_inchikey_cache() -> dict[str, tuple[str | None, str | None]]:
    if INCHIKEY_CACHE.exists():
        with INCHIKEY_CACHE.open("rb") as handle:
            data = pickle.load(handle)
        if isinstance(data, dict):
            return data
    return {}


def save_inchikey_cache(cache: dict[str, tuple[str | None, str | None]]) -> None:
    with INCHIKEY_CACHE.open("wb") as handle:
        pickle.dump(cache, handle)


def load_pubchem_name_cache() -> dict[str, dict[str, Any]]:
    if PUBCHEM_NAME_CACHE.exists():
        with PUBCHEM_NAME_CACHE.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    return {}


def save_pubchem_name_cache(cache: dict[str, dict[str, Any]]) -> None:
    with PUBCHEM_NAME_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=0)


def add_structure_columns(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    inchi_col: str | None = None,
    inchikey_col: str | None = None,
    cache: dict[str, tuple[str | None, str | None]] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    keys: list[tuple[str | None, str | None]] = []
    for _, row in df.iterrows():
        keys.append(
            compute_structure_keys(
                smiles=row.get(smiles_col) if smiles_col in df.columns else None,
                inchi=row.get(inchi_col) if inchi_col and inchi_col in df.columns else None,
                inchikey=row.get(inchikey_col) if inchikey_col and inchikey_col in df.columns else None,
                cache=cache,
            )
        )
    df["inchikey"] = [k[0] for k in keys]
    df["inchikey_skeleton"] = [k[1] for k in keys]
    return df


class StructureIndex:
    """Three-tier lookup index: exact InChIKey → skeleton → name_key."""

    def __init__(self) -> None:
        self.by_inchikey: dict[str, Any] = {}
        self.by_skeleton: dict[str, Any] = {}
        self.by_name_key: dict[str, Any] = {}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        payload_col: str | None = None,
        name_key_col: str = "name_key",
    ) -> StructureIndex:
        index = cls()
        for _, row in df.iterrows():
            payload = row.to_dict() if payload_col is None else row[payload_col]
            name_key = row.get(name_key_col)
            if name_key:
                index.by_name_key[str(name_key)] = payload
            ik = row.get("inchikey")
            sk = row.get("inchikey_skeleton")
            if ik and str(ik).strip():
                index.by_inchikey[str(ik).strip().upper()] = payload
            if sk and str(sk).strip():
                index.by_skeleton[str(sk).strip().upper()] = payload
        return index

    def lookup(self, row: pd.Series, name_key_col: str = "name_key") -> tuple[Any | None, MatchMethod | None]:
        ik = row.get("inchikey")
        sk = row.get("inchikey_skeleton")
        nk = row.get(name_key_col)
        if ik and str(ik).strip():
            hit = self.by_inchikey.get(str(ik).strip().upper())
            if hit is not None:
                return hit, "inchikey"
        if sk and str(sk).strip():
            hit = self.by_skeleton.get(str(sk).strip().upper())
            if hit is not None:
                return hit, "skeleton"
        if nk and str(nk).strip():
            hit = self.by_name_key.get(str(nk).strip())
            if hit is not None:
                return hit, "name"
        return None, None


def resolve_pubchem_structure(
    name: str,
    session: requests.Session | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    sleep_s: float = 0.22,
) -> dict[str, Any]:
    """Resolve name → CID, SMILES, InChIKey via PubChem (cached)."""
    key = str(name).strip().lower()
    if cache is not None and key in cache:
        return cache[key]

    result: dict[str, Any] = {"cid": None, "smiles": None, "inchikey": None, "inchikey_skeleton": None}
    if not key:
        if cache is not None:
            cache[key] = result
        return result

    own_session = session is None
    if own_session:
        session = requests.Session()
        session.headers.update({"User-Agent": "NOSA-DB/1.0 (structure matching)"})

    encoded = quote(str(name), safe="")
    props = "ConnectivitySMILES,InChI,InChIKey"
    url = f"{PUG_BASE}/compound/name/{encoded}/property/{props}/JSON"
    try:
        response = session.get(url, timeout=60)
        if sleep_s:
            import time

            time.sleep(sleep_s)
        if response.status_code == 404:
            if cache is not None:
                cache[key] = result
            return result
        response.raise_for_status()
        data = response.json()
        properties = (data.get("PropertyTable") or {}).get("Properties") or []
        if properties:
            row = properties[0]
            result["cid"] = row.get("CID")
            result["smiles"] = row.get("ConnectivitySMILES")
            ik = row.get("InChIKey")
            if ik:
                result["inchikey"] = str(ik).strip().upper()
                result["inchikey_skeleton"] = inchikey_skeleton(result["inchikey"])
    except Exception:
        pass

    if cache is not None:
        cache[key] = result
    return result


def recompute_dose_feasible_nosa(df: pd.DataFrame) -> pd.DataFrame:
    """Unified dose feasibility: ≤100 True, >100 False, null dose → null."""
    df = df.copy()
    if "max_dose_mg" not in df.columns:
        return df

    def _feasible(dose: Any) -> bool | None:
        if dose is None or (isinstance(dose, float) and pd.isna(dose)):
            return None
        return float(dose) <= 100.0

    df["dose_feasible_nosa"] = df["max_dose_mg"].map(_feasible)
    return df


def sync_unified_dose(row: pd.Series) -> pd.Series:
    chembl = row.get("max_clinical_dose_mg")
    ct = row.get("ct_max_dose_mg")
    if pd.notna(chembl):
        row["max_dose_mg"] = float(chembl)
        row["max_dose_source"] = "chembl"
    elif pd.notna(ct):
        row["max_dose_mg"] = float(ct)
        row["max_dose_source"] = "clinicaltrials"
    elif "max_dose_mg" not in row.index or pd.isna(row.get("max_dose_mg")):
        row["max_dose_mg"] = None
        row["max_dose_source"] = None

    dose = row.get("max_dose_mg")
    row["dose_feasible_nosa"] = None if pd.isna(dose) else float(dose) <= 100.0
    return row


MAX_PLAUSIBLE_DOSE_MG = 2000.0


def sanitize_dose_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Null out implausible unified/CT doses left by bad extraction."""
    df = df.copy()
    for col in ("max_dose_mg", "ct_max_dose_mg", "max_clinical_dose_mg"):
        if col in df.columns:
            bad = df[col].notna() & (df[col] > MAX_PLAUSIBLE_DOSE_MG)
            df.loc[bad, col] = None
    return df


def count_non_null_cells(df: pd.DataFrame) -> int:
    return int(df.notna().sum().sum())
