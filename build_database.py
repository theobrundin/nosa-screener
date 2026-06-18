#!/usr/bin/env python3
"""Build NOSA drug screening database from ChEMBL, Orange Book, and local sources."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

from enrich_pubchem import PUG_BASE, REQUEST_SLEEP_S, request_json

ROOT = Path(__file__).resolve().parent
ORANGE_BOOK_DIR = ROOT / "EOBZIP_2026_04"
REGULATORY_CSV = ROOT / "FDA+EMA+PMDA_Approved.csv"
STRUCTURES_TSV = ROOT / "structures.smiles.tsv"
OUTPUT_CSV = ROOT / "nosa_drug_database.csv"
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"
CHEMBL_CACHE = ROOT / "chembl_raw_cache.pkl"
MANUAL_CACHE = ROOT / "manual_compounds_cache.json"

ENRICHMENT_COLUMNS = [
    "pubchem_cid",
    "vapor_pressure_mmhg",
    "vapor_pressure_raw",
    "melting_point_c",
    "melting_point_raw",
    "boiling_point_c",
    "boiling_point_raw",
]

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

MANUAL_COMPOUNDS = list(
    dict.fromkeys(
        [
            # Terpenes
            "menthol",
            "eucalyptol",
            "linalool",
            "borneol",
            "camphor",
            "thymol",
            "carvacrol",
            "limonene",
            "alpha-pinene",
            "geraniol",
            # Other volatiles
            "capsaicin",
            "piperine",
            "vanillin",
            "eugenol",
            "methyl salicylate",
            "cinnamaldehyde",
            "benzaldehyde",
            # CNS/nasal relevant naturals
            "caffeine",
            "scopolamine",
            "nicotine",
            "melatonin",
            "menthol",
            # Additional
            "piperidine",
            "pyridine",
            "indole",
        ]
    )
)

SALT_SUFFIXES = (
    " hydrochloride",
    " hcl",
    " dihydrochloride",
    " sulfate",
    " bisulfate",
    " sodium",
    " potassium",
    " calcium",
    " mesylate",
    " maleate",
    " tartrate",
    " acetate",
    " phosphate",
    " dihydrate",
    " monohydrate",
    " hemihydrate",
    " succinate",
    " besylate",
    " tosylate",
    " citrate",
    " lactate",
    " bromide",
    " chloride",
    " nitrate",
    " mesilate",
    " hydrobromide",
    " dihydrobromide",
    " xinafoate",
    " propionate",
    " butyrate",
    " valerate",
    " dipropionate",
    " mononitrate",
    " dinitrate",
    " hemisulfate",
    " hemifumarate",
    " hydroxybenzoate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NOSA drug database.")
    parser.add_argument(
        "--skip-chembl",
        action="store_true",
        help="Skip ChEMBL API fetch; load chembl_raw_cache.pkl instead.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    """Normalize drug names for joining across sources."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    changed = True
    while changed:
        changed = False
        for suffix in SALT_SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                changed = True
    return s


def add_source(existing: str | None, new_source: str) -> str:
    parts: list[str] = []
    if existing and str(existing).strip():
        parts = [p for p in str(existing).split("|") if p]
    if new_source not in parts:
        parts.append(new_source)
    return "|".join(parts)


def parse_orange_book_date(value: str) -> pd.Timestamp | pd.NaT:
    if not isinstance(value, str) or not value.strip():
        return pd.NaT
    try:
        return pd.to_datetime(value.strip(), format="%b %d, %Y")
    except (ValueError, TypeError):
        return pd.NaT


def fetch_chembl_molecules() -> pd.DataFrame:
    """Pull small molecules with max_phase >= 2 from ChEMBL."""
    molecule = new_client.molecule
    drug = new_client.drug

    records = molecule.filter(
        molecule_type="Small molecule",
        max_phase__gte=2,
    ).only(
        [
            "molecule_chembl_id",
            "pref_name",
            "max_phase",
            "molecule_properties",
            "molecule_structures",
        ]
    )

    rows: list[dict] = []
    for rec in tqdm(records, desc="Fetching ChEMBL molecules"):
        props = rec.get("molecule_properties") or {}
        structs = rec.get("molecule_structures") or {}
        rows.append(
            {
                "name": rec.get("pref_name"),
                "chembl_id": rec.get("molecule_chembl_id"),
                "chembl_max_phase": rec.get("max_phase"),
                "molecular_weight": props.get("full_mwt"),
                "logP": props.get("alogp"),
                "PSA": props.get("psa"),
                "H_donors": props.get("hbd"),
                "H_acceptors": props.get("hba"),
                "rotatable_bonds": props.get("rtb"),
                "SMILES": structs.get("canonical_smiles"),
            }
        )

    df = pd.DataFrame(rows)
    df["name_key"] = df["name"].map(normalize_name)
    df["source"] = "chembl"
    df["sources"] = "chembl"

    indication_by_molecule: dict[str, str] = {}
    chembl_ids = df["chembl_id"].dropna().unique().tolist()
    batch_size = 50
    for i in range(0, len(chembl_ids), batch_size):
        batch = chembl_ids[i : i + batch_size]
        for d in drug.filter(molecule_chembl_id__in=batch):
            indication = d.get("indication_class")
            if not indication:
                continue
            mol_ids = d.get("molecule_chembl_id") or []
            if isinstance(mol_ids, str):
                mol_ids = [mol_ids]
            for mol_id in mol_ids:
                if mol_id and mol_id not in indication_by_molecule:
                    indication_by_molecule[mol_id] = indication

    df["indication_class"] = df["chembl_id"].map(indication_by_molecule)
    return df


def load_orange_book() -> pd.DataFrame:
    """Merge Orange Book products, patents, and exclusivity; aggregate per ingredient."""
    products = pd.read_csv(ORANGE_BOOK_DIR / "products.txt", sep="~", dtype=str)
    patents = pd.read_csv(ORANGE_BOOK_DIR / "patent.txt", sep="~", dtype=str)
    exclusivity = pd.read_csv(ORANGE_BOOK_DIR / "exclusivity.txt", sep="~", dtype=str)

    key_cols = ["Appl_Type", "Appl_No", "Product_No"]

    patents = patents[key_cols + ["Patent_Expire_Date_Text"]].copy()
    patents["patent_expiry"] = patents["Patent_Expire_Date_Text"].map(parse_orange_book_date)
    patents = patents.groupby(key_cols, as_index=False)["patent_expiry"].min()

    exclusivity = exclusivity[key_cols + ["Exclusivity_Date"]].copy()
    exclusivity["exclusivity_date"] = exclusivity["Exclusivity_Date"].map(
        parse_orange_book_date
    )
    exclusivity = exclusivity.groupby(key_cols, as_index=False)["exclusivity_date"].min()

    merged = products.merge(patents, on=key_cols, how="left").merge(
        exclusivity, on=key_cols, how="left"
    )

    ingredient_rows: list[dict] = []
    for _, row in merged.iterrows():
        ingredients = [
            part.strip() for part in str(row["Ingredient"]).split(";") if part.strip()
        ]
        for ingredient in ingredients:
            ingredient_rows.append(
                {
                    "ingredient": ingredient,
                    "name_key": normalize_name(ingredient),
                    "patent_expiry": row["patent_expiry"],
                    "exclusivity_date": row["exclusivity_date"],
                }
            )

    return (
        pd.DataFrame(ingredient_rows)
        .groupby("name_key", as_index=False)
        .agg(
            ingredient=("ingredient", "first"),
            earliest_patent_expiry=("patent_expiry", "min"),
            earliest_exclusivity_date=("exclusivity_date", "min"),
        )
    )


def load_regulatory_annotations() -> tuple[set[str], pd.DataFrame]:
    """Load FDA+EMA+PMDA file and return normalized INN keys for annotation."""
    raw = pd.read_csv(REGULATORY_CSV, header=None)
    print("FDA+EMA+PMDA_Approved.csv columns (inferred):", raw.columns.tolist())
    print("FDA+EMA+PMDA_Approved.csv sample:")
    print(raw.head(5).to_string(index=False))

    if raw.shape[1] >= 2:
        reg = raw.iloc[:, :2].copy()
        reg.columns = ["drugcentral_id", "inn"]
    else:
        reg = raw.copy()
        reg.columns = ["inn"][: raw.shape[1]]

    reg["name_key"] = reg["inn"].map(normalize_name)
    approved_keys = set(reg["name_key"].dropna()) - {""}
    return approved_keys, reg


def annotate_regulatory_flags(df: pd.DataFrame, approved_keys: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["ema_approved"] = df["name_key"].isin(approved_keys)
    df["pmda_approved"] = df["name_key"].isin(approved_keys)
    return df


def load_drugcentral_structures() -> pd.DataFrame:
    structures = pd.read_csv(STRUCTURES_TSV, sep="\t")
    print("structures.smiles.tsv columns:", structures.columns.tolist())
    structures = structures.rename(columns={"INN": "inn", "SMILES": "smiles"})
    structures["name_key"] = structures["inn"].map(normalize_name)
    return structures[["name_key", "smiles"]].drop_duplicates(subset="name_key", keep="first")


def fill_smiles_from_drugcentral(df: pd.DataFrame, structures: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    smiles_lookup = structures.set_index("name_key")["smiles"]
    missing = df["SMILES"].isna()
    fill_mask = missing & df["name_key"].isin(smiles_lookup.index)
    filled = int(fill_mask.sum())

    df.loc[fill_mask, "SMILES"] = df.loc[fill_mask, "name_key"].map(smiles_lookup)
    chembl_smiles = df["SMILES"].notna() & (df["source"] == "chembl")
    df.loc[chembl_smiles, "smiles_source"] = "chembl"
    df.loc[fill_mask, "smiles_source"] = "drugcentral"
    df.loc[fill_mask, "sources"] = df.loc[fill_mask, "sources"].map(
        lambda s: add_source(s, "drugcentral")
    )
    return df, filled


def fetch_pubchem_properties(name: str, session: requests.Session) -> dict[str, Any]:
    encoded = quote(str(name), safe="")
    props = (
        "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,"
        "RotatableBondCount,ConnectivitySMILES"
    )
    url = f"{PUG_BASE}/compound/name/{encoded}/property/{props}/JSON"
    data = request_json(url, session)
    if not data:
        return {}
    properties = (data.get("PropertyTable") or {}).get("Properties") or []
    if not properties:
        return {}
    row = properties[0]
    return {
        "pubchem_cid": row.get("CID"),
        "molecular_weight": _to_float(row.get("MolecularWeight")),
        "logP": _to_float(row.get("XLogP")),
        "PSA": _to_float(row.get("TPSA")),
        "H_donors": _to_float(row.get("HBondDonorCount")),
        "H_acceptors": _to_float(row.get("HBondAcceptorCount")),
        "rotatable_bonds": _to_float(row.get("RotatableBondCount")),
        "SMILES": row.get("ConnectivitySMILES") or row.get("IsomericSMILES"),
    }


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_manual_cache() -> dict[str, dict[str, Any]]:
    if not MANUAL_CACHE.exists():
        return {}
    with MANUAL_CACHE.open() as handle:
        return json.load(handle)


def save_manual_cache(cache: dict[str, dict[str, Any]]) -> None:
    with MANUAL_CACHE.open("w") as handle:
        json.dump(cache, handle, indent=2)


def build_manual_compounds(chembl_keys: set[str]) -> tuple[pd.DataFrame, list[str]]:
    cache = load_manual_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": "NOSA-DB/1.0 (drug screening research)"})

    rows: list[dict] = []
    found_in_chembl: list[str] = []

    for compound in tqdm(MANUAL_COMPOUNDS, desc="Fetching manual compounds"):
        key = normalize_name(compound)
        if key in chembl_keys:
            found_in_chembl.append(compound)
            continue

        if key not in cache or not cache.get(key):
            cache[key] = fetch_pubchem_properties(compound, session)

        props = cache[key]
        rows.append(
            {
                "name": compound,
                "name_key": key,
                "chembl_id": None,
                "chembl_max_phase": None,
                "indication_class": "Natural product / volatile",
                "molecular_weight": props.get("molecular_weight"),
                "logP": props.get("logP"),
                "PSA": props.get("PSA"),
                "H_donors": props.get("H_donors"),
                "H_acceptors": props.get("H_acceptors"),
                "rotatable_bonds": props.get("rotatable_bonds"),
                "SMILES": props.get("SMILES"),
                "pubchem_cid": props.get("pubchem_cid"),
                "source": "manual",
                "sources": "manual",
            }
        )

    save_manual_cache(cache)
    return pd.DataFrame(rows), found_in_chembl


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.copy()
    df["_priority"] = df["source"].map({"chembl": 0, "manual": 1}).fillna(2)
    df["_phase_sort"] = pd.to_numeric(df["chembl_max_phase"], errors="coerce").fillna(-1)
    df = df.sort_values(["_priority", "_phase_sort"], ascending=[True, False])
    df = df.drop_duplicates(subset="name_key", keep="first")
    df = df.drop(columns=["_priority", "_phase_sort"])
    return df, before - len(df)


def carry_forward_enrichment(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    enrich_source = ENRICHED_CSV
    if not enrich_source.exists():
        backup = ENRICHED_CSV.with_name(f"{ENRICHED_CSV.stem}.csv.bak")
        if backup.exists():
            enrich_source = backup
        else:
            for col in ENRICHMENT_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df, 0

    enriched = pd.read_csv(enrich_source)
    keep_cols = ["name"] + [c for c in ENRICHMENT_COLUMNS if c in enriched.columns]
    enriched = enriched[keep_cols].drop_duplicates(subset="name", keep="first")

    for col in ENRICHMENT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    merged = df.merge(enriched, on="name", how="left")
    carried = int(merged[ENRICHMENT_COLUMNS[0]].notna().sum()) if ENRICHMENT_COLUMNS else 0
    return merged, carried


def flag_nosa_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def is_candidate(row: pd.Series) -> bool:
        raw = str(row.get("name", "")).lower().strip()
        return raw in NOSA_CANDIDATES or row.get("name_key") in NOSA_CANDIDATES

    df["nosa_candidate"] = df.apply(is_candidate, axis=1)
    return df


def print_build_summary(stats: dict[str, int]) -> None:
    new_rows = stats["total_rows"] - stats["enriched_carried"]
    print("\n── Build summary ──────────────────────────────────")
    print(f"  ChEMBL phase 4 (approved):     {stats['phase_4']:>7,}")
    print(f"  ChEMBL phase 3:                {stats['phase_3']:>7,}")
    print(f"  ChEMBL phase 2:                {stats['phase_2']:>7,}")
    print(f"  Manual compounds added:        {stats['manual_added']:>7,}")
    print(f"  Duplicates removed:            {stats['duplicates_removed']:>7,}")
    print(f"  Enriched rows carried forward: {stats['enriched_carried']:>7,}")
    print(f"  SMILES filled from DrugCentral: {stats['smiles_filled']:>6,}")
    print(f"  EMA approved flags set:        {stats['ema_flags']:>7,}")
    print(f"  PMDA approved flags set:       {stats['pmda_flags']:>7,}")
    print("  ────────────────────────────────────────────────")
    print(f"  TOTAL ROWS:                    {stats['total_rows']:>7,}")
    print(f"  Output → {OUTPUT_CSV.name}")
    print("  Next step: python enrich_pubchem.py to fill")
    print(f"             VP/MP for ~{new_rows:,} new rows")


def build_database(skip_chembl: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {
        "phase_4": 0,
        "phase_3": 0,
        "phase_2": 0,
        "manual_added": 0,
        "duplicates_removed": 0,
        "enriched_carried": 0,
        "smiles_filled": 0,
        "ema_flags": 0,
        "pmda_flags": 0,
        "total_rows": 0,
    }

    if skip_chembl:
        if not CHEMBL_CACHE.exists():
            raise FileNotFoundError(
                f"--skip-chembl requires cache file: {CHEMBL_CACHE.name}"
            )
        print(f"Loading ChEMBL data from cache ({CHEMBL_CACHE.name})...")
        with CHEMBL_CACHE.open("rb") as handle:
            chembl = pickle.load(handle)
    else:
        print("Fetching small molecules from ChEMBL (max_phase >= 2)...")
        chembl = fetch_chembl_molecules()
        with CHEMBL_CACHE.open("wb") as handle:
            pickle.dump(chembl, handle)
        print(f"  Cached ChEMBL raw data → {CHEMBL_CACHE.name}")

    print(f"  ChEMBL records (pre-dedup): {len(chembl):,}")

    chembl_keys = set(chembl["name_key"].dropna()) - {""}
    manual_df, manual_in_chembl = build_manual_compounds(chembl_keys)
    stats["manual_added"] = len(manual_df)
    if manual_in_chembl:
        print("  Manual compounds already in ChEMBL (skipped):")
        for name in manual_in_chembl:
            print(f"    - {name}")

    combined = pd.concat([chembl, manual_df], ignore_index=True, sort=False)
    combined, stats["duplicates_removed"] = deduplicate(combined)

    phase_numeric = pd.to_numeric(combined["chembl_max_phase"], errors="coerce")
    stats["phase_4"] = int((phase_numeric == 4).sum())
    stats["phase_3"] = int((phase_numeric == 3).sum())
    stats["phase_2"] = int((phase_numeric == 2).sum())

    print("Loading and merging Orange Book files...")
    orange_book = load_orange_book()
    merged = combined.merge(orange_book, on="name_key", how="left")

    approved_keys, _ = load_regulatory_annotations()
    merged = annotate_regulatory_flags(merged, approved_keys)
    stats["ema_flags"] = int(merged["ema_approved"].sum())
    stats["pmda_flags"] = int(merged["pmda_approved"].sum())
    print(f"  EMA approved flags set: {stats['ema_flags']:,}")
    print(f"  PMDA approved flags set: {stats['pmda_flags']:,}")

    structures = load_drugcentral_structures()
    merged["smiles_source"] = None
    merged.loc[merged["SMILES"].notna(), "smiles_source"] = "chembl"
    merged, stats["smiles_filled"] = fill_smiles_from_drugcentral(merged, structures)
    print(f"  SMILES filled from DrugCentral: {stats['smiles_filled']:,}")

    merged = flag_nosa_candidates(merged)
    merged, stats["enriched_carried"] = carry_forward_enrichment(merged)
    print(f"  Enriched rows carried forward: {stats['enriched_carried']:,}")

    column_order = [
        "name",
        "chembl_id",
        "chembl_max_phase",
        "indication_class",
        "molecular_weight",
        "logP",
        "PSA",
        "H_donors",
        "H_acceptors",
        "rotatable_bonds",
        "SMILES",
        "smiles_source",
        "source",
        "sources",
        "ingredient",
        "earliest_patent_expiry",
        "earliest_exclusivity_date",
        "ema_approved",
        "pmda_approved",
        "nosa_candidate",
        *ENRICHMENT_COLUMNS,
    ]
    for col in column_order:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[column_order]
    stats["total_rows"] = len(merged)
    return merged, stats


def main() -> None:
    args = parse_args()
    df, stats = build_database(skip_chembl=args.skip_chembl)
    df.to_csv(OUTPUT_CSV, index=False)
    print_build_summary(stats)


if __name__ == "__main__":
    main()
