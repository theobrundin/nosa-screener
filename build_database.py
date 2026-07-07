#!/usr/bin/env python3
"""Build NOSA drug screening database from ChEMBL, Orange Book, and local sources."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from tqdm import tqdm

from enrich_pubchem import PUG_BASE, REQUEST_SLEEP_S, request_json

ROOT = Path(__file__).resolve().parent
ORANGE_BOOK_DIR = ROOT / "EOBZIP_2026_04"
REGULATORY_CSV = ROOT / "FDA+EMA+PMDA_Approved.csv"
STRUCTURES_TSV = ROOT / "structures.smiles.tsv"
DRUGBANK_CSV = ROOT / "kaggle-drugbank" / "drugbank_clean.csv"
OUTPUT_CSV = ROOT / "nosa_drug_database.csv"
ENRICHED_CSV = ROOT / "nosa_drug_database_enriched.csv"
CHEMBL_CACHE = ROOT / "chembl_raw_cache.pkl"
MANUAL_CACHE = ROOT / "manual_compounds_cache.json"
BATCH_SIZE = 50

METADATA_COLUMNS = [
    "mesh_heading",
    "atc_code",
    "atc_level1",
    "mechanism_of_action",
    "mechanism_of_action_all",
    "target_name",
    "target_chembl_id",
    "action_type",
    "synonyms",
    "max_synonym_count",
    "route_of_administration",
    "dosage_form",
    "max_clinical_dose_mg",
    "max_clinical_dose_phase",
    "dose_feasible_nosa",
    "pka_predicted",
    "logD_pH6",
    "logD_pH74",
    "ionization_class",
]

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


def iter_batches(items: list[str], size: int = BATCH_SIZE) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def pipe_join_unique(values: list[str | None]) -> str | None:
    seen: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return "|".join(seen) if seen else None


def _mol_ids_from_record(record: dict[str, Any], key: str = "molecule_chembl_id") -> list[str]:
    raw = record.get(key)
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw if item]


def parse_dose_mg(value: Any, units: str | None) -> float | None:
    if value is None or units is None:
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    unit = str(units).lower().strip()
    if unit in {"mg", "mg/day", "mg/d", "mg per day"}:
        return amount
    if unit in {"g/day", "g/d", "g per day"}:
        return amount * 1000.0
    if unit in {"mg kg-1 day-1", "mg/kg/day", "mg kg-1"}:
        return amount * 70.0
    return None


# RDKit SMARTS-based pKa estimation (dimorphite-dl requires Python 3.10+).
PKA_PATTERNS: list[tuple[str, float, str]] = [
    ("[CX3](=O)[OX2H1]", 4.2, "acid"),
    ("[PX4](=O)([OX2H])[OX2H]", 2.0, "acid"),
    ("[SX4](=O)(=O)[OX2H]", -1.0, "acid"),
    ("[#6][OX2H]", 10.0, "acid"),
    ("[NX4+]", 10.5, "base"),
    ("[NX3;H2,H1;!$(NC=O);!$(N=O)]", 9.5, "base"),
    ("[nX3;+]", 5.0, "base"),
    ("[nX2]", 4.5, "base"),
    ("[NX3](=O)[OX2H]", 9.0, "acid"),
]


def predict_pka_from_smiles(smiles: str) -> tuple[float | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    acidic: list[float] = []
    basic: list[float] = []
    for smarts, pka, site_type in PKA_PATTERNS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            if site_type == "acid":
                acidic.append(pka)
            else:
                basic.append(pka)

    if not acidic and not basic:
        return None, "neutral"

    if acidic and basic:
        ion_class = "zwitterion"
        pka = basic[0] if basic else acidic[0]
    elif basic:
        ion_class = "base"
        pka = max(basic)
    else:
        ion_class = "acid"
        pka = min(acidic)

    return pka, ion_class


def calculate_logd(logp: float | None, pka: float | None, ion_class: str | None, ph: float) -> float | None:
    if logp is None or pka is None or ion_class is None:
        return None
    if ion_class == "neutral":
        return logp
    if ion_class in {"base", "zwitterion"}:
        return logp - math.log10(1.0 + 10.0 ** (pka - ph))
    if ion_class == "acid":
        return logp - math.log10(1.0 + 10.0 ** (ph - pka))
    return None


def add_pka_logd_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pka_vals: list[float | None] = []
    ion_classes: list[str | None] = []
    logd6: list[float | None] = []
    logd74: list[float | None] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting pKa/logD"):
        smiles = row.get("SMILES")
        logp = _to_float(row.get("logP"))
        if not isinstance(smiles, str) or not smiles.strip():
            pka_vals.append(None)
            ion_classes.append(None)
            logd6.append(None)
            logd74.append(None)
            continue
        pka, ion_class = predict_pka_from_smiles(smiles.strip())
        pka_vals.append(pka)
        ion_classes.append(ion_class)
        logd6.append(calculate_logd(logp, pka, ion_class, 6.0))
        logd74.append(calculate_logd(logp, pka, ion_class, 7.4))

    df["pka_predicted"] = pka_vals
    df["ionization_class"] = ion_classes
    df["logD_pH6"] = logd6
    df["logD_pH74"] = logd74
    return df


def enrich_chembl_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Batch-fetch ChEMBL drug metadata and annotate the molecule dataframe."""
    chembl_ids = df["chembl_id"].dropna().astype(str).unique().tolist()
    if not chembl_ids:
        return df

    mesh_by_mol: dict[str, list[str]] = {}
    atc_by_mol: dict[str, list[str]] = {}
    route_by_mol: dict[str, list[str]] = {}
    dosage_by_mol: dict[str, list[str]] = {}
    mech_by_mol: dict[str, list[dict[str, Any]]] = {}
    dose_by_mol: dict[str, list[float]] = {}
    synonym_by_mol: dict[str, list[str]] = {}

    drug_indication = new_client.drug_indication
    drug_resource = new_client.drug
    mechanism_resource = new_client.mechanism
    activity_resource = new_client.activity
    target_resource = new_client.target

    for batch in tqdm(iter_batches(chembl_ids), desc="ChEMBL drug indications (MeSH)"):
        for rec in drug_indication.filter(molecule_chembl_id__in=batch):
            mol_id = rec.get("molecule_chembl_id")
            heading = rec.get("mesh_heading")
            if mol_id and heading:
                mesh_by_mol.setdefault(mol_id, []).append(str(heading))

    for batch in tqdm(iter_batches(chembl_ids), desc="ChEMBL drug resource (ATC/routes)"):
        for rec in drug_resource.filter(molecule_chembl_id__in=batch):
            mol_id = rec.get("molecule_chembl_id")
            if not mol_id:
                continue
            atc_entries = rec.get("atc_classification") or []
            for entry in atc_entries:
                code = entry.get("code") if isinstance(entry, dict) else None
                if code:
                    atc_by_mol.setdefault(mol_id, []).append(str(code))
            forms: list[str] = []
            if rec.get("oral"):
                forms.append("oral")
            if rec.get("parenteral"):
                forms.append("parenteral")
            if rec.get("topical"):
                forms.append("topical")
            if forms:
                dosage_by_mol.setdefault(mol_id, []).extend(forms)
                route_by_mol.setdefault(mol_id, []).append(forms[0])

    for batch in tqdm(iter_batches(chembl_ids), desc="ChEMBL mechanisms"):
        for rec in mechanism_resource.filter(molecule_chembl_id__in=batch):
            mol_id = rec.get("molecule_chembl_id")
            if not mol_id:
                continue
            mech_by_mol.setdefault(mol_id, []).append(rec)

    target_ids = {
        str(rec.get("target_chembl_id"))
        for recs in mech_by_mol.values()
        for rec in recs
        if rec.get("target_chembl_id")
    }
    target_names: dict[str, str] = {}
    target_id_list = sorted(target_ids)
    for batch in tqdm(iter_batches(target_id_list), desc="ChEMBL targets"):
        for rec in target_resource.filter(target_chembl_id__in=batch):
            tid = rec.get("target_chembl_id")
            if tid:
                target_names[tid] = rec.get("pref_name") or rec.get("target_type")

    for batch in tqdm(iter_batches(chembl_ids), desc="ChEMBL dose activities"):
        for rec in activity_resource.filter(
            molecule_chembl_id__in=batch,
            standard_type="Dose",
        ):
            mol_id = rec.get("molecule_chembl_id")
            dose_mg = parse_dose_mg(rec.get("standard_value"), rec.get("standard_units"))
            if mol_id and dose_mg is not None:
                dose_by_mol.setdefault(mol_id, []).append(dose_mg)

    for _, row in df.iterrows():
        mol_id = row.get("chembl_id")
        if not mol_id or pd.isna(mol_id):
            continue
        mol_id = str(mol_id)
        syns = row.get("molecule_synonyms") or []
        names: list[str] = []
        if isinstance(syns, list):
            for entry in syns:
                if isinstance(entry, dict):
                    for key in ("molecule_synonym", "synonyms"):
                        val = entry.get(key)
                        if val:
                            names.append(str(val))
        if names:
            synonym_by_mol[mol_id] = names

    mesh_final: dict[str, str | None] = {}
    for mol_id, headings in mesh_by_mol.items():
        counts = Counter(headings)
        ordered = [h for h, _ in counts.most_common()]
        mesh_final[mol_id] = pipe_join_unique(ordered)

    atc_final: dict[str, str | None] = {}
    atc_l1_final: dict[str, str | None] = {}
    for mol_id, codes in atc_by_mol.items():
        atc_final[mol_id] = pipe_join_unique(codes)
        levels = sorted({code[0] for code in codes if code})
        atc_l1_final[mol_id] = pipe_join_unique(levels)

    mech_primary: dict[str, str | None] = {}
    mech_all: dict[str, str | None] = {}
    target_name_map: dict[str, str | None] = {}
    target_id_map: dict[str, str | None] = {}
    action_map: dict[str, str | None] = {}
    for mol_id, records in mech_by_mol.items():
        actions = [str(r.get("mechanism_of_action")) for r in records if r.get("mechanism_of_action")]
        mech_primary[mol_id] = actions[0] if actions else None
        mech_all[mol_id] = pipe_join_unique(actions)
        primary = records[0]
        tid = primary.get("target_chembl_id")
        target_id_map[mol_id] = tid
        target_name_map[mol_id] = target_names.get(tid) if tid else None
        action_map[mol_id] = primary.get("action_type")

    dose_mg_map: dict[str, float | None] = {}
    for mol_id, doses in dose_by_mol.items():
        dose_mg_map[mol_id] = max(doses) if doses else None

    df = df.copy()
    df["mesh_heading"] = df["chembl_id"].map(mesh_final)
    df["atc_code"] = df["chembl_id"].map(atc_final)
    df["atc_level1"] = df["chembl_id"].map(atc_l1_final)
    df["mechanism_of_action"] = df["chembl_id"].map(mech_primary)
    df["mechanism_of_action_all"] = df["chembl_id"].map(mech_all)
    df["target_name"] = df["chembl_id"].map(target_name_map)
    df["target_chembl_id"] = df["chembl_id"].map(target_id_map)
    df["action_type"] = df["chembl_id"].map(action_map)
    df["synonyms"] = df["chembl_id"].map(
        lambda cid: pipe_join_unique(synonym_by_mol.get(str(cid), [])) if pd.notna(cid) else None
    )
    df["max_synonym_count"] = df["chembl_id"].map(
        lambda cid: len(synonym_by_mol.get(str(cid), [])) if pd.notna(cid) else None
    )
    df["route_of_administration"] = df["chembl_id"].map(
        lambda cid: pipe_join_unique(route_by_mol.get(str(cid), [])) if pd.notna(cid) else None
    )
    df["dosage_form"] = df["chembl_id"].map(
        lambda cid: pipe_join_unique(dosage_by_mol.get(str(cid), [])) if pd.notna(cid) else None
    )
    df["max_clinical_dose_mg"] = df["chembl_id"].map(dose_mg_map)
    df["max_clinical_dose_phase"] = df.apply(
        lambda row: row["chembl_max_phase"]
        if pd.notna(row.get("max_clinical_dose_mg")) and pd.notna(row.get("chembl_max_phase"))
        else None,
        axis=1,
    )
    df["dose_feasible_nosa"] = df["max_clinical_dose_mg"].map(
        lambda dose: True if pd.notna(dose) and dose <= 100 else False if pd.notna(dose) else None
    )

    print(f"  Molecules with mechanism data: {len(mech_by_mol):,}")
    return df


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
            "molecule_synonyms",
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
                "molecule_synonyms": rec.get("molecule_synonyms"),
            }
        )

    df = pd.DataFrame(rows)
    df["name_key"] = df["name"].map(normalize_name)
    df["source"] = "chembl"
    df["sources"] = "chembl"

    indication_by_molecule: dict[str, str] = {}
    chembl_ids = df["chembl_id"].dropna().unique().tolist()
    for batch in iter_batches(chembl_ids):
        for d in drug.filter(molecule_chembl_id__in=batch):
            indication = d.get("indication_class")
            if not indication:
                continue
            for mol_id in _mol_ids_from_record(d):
                if mol_id not in indication_by_molecule:
                    indication_by_molecule[mol_id] = indication

    df["indication_class"] = df["chembl_id"].map(indication_by_molecule)
    df = enrich_chembl_metadata(df)
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


DRUGBANK_PK_MAP = {
    "absorption": "absorption",
    "half-life": "half_life",
    "protein-binding": "protein_binding",
    "metabolism": "metabolism",
    "route-of-elimination": "route_of_elimination",
    "volume-of-distribution": "volume_of_distribution",
    "clearance": "clearance",
    "toxicity": "toxicity",
}

CHEMBL_PRIMARY_FIELDS = {
    "molecular_weight",
    "logP",
    "PSA",
    "H_donors",
    "H_acceptors",
    "rotatable_bonds",
    "SMILES",
    "max_clinical_dose_mg",
    "dosage_form",
    "chembl_max_phase",
}

DRUGBANK_PROTEINS_TSV = ROOT / "kaggle-drugbank" / "proteins.tsv"
UNIPROT_LOOKUP_CSV = ROOT / "kaggle-drugbank" / "uniprot_lookup.csv"

ANNOTATION_COLUMNS = [
    "target_names",
    "target_genes",
    "target_count",
    "primary_target",
    "cns_target",
    "metabolizing_enzymes",
    "cyp_enzymes",
    "enzyme_count",
    "nasal_cyp_risk",
    "cyp3a4_substrate",
    "transporters",
    "transporter_count",
    "pgp_substrate",
]

BE_TOKEN_RE = re.compile(r"^BE\d{4,7}$", re.IGNORECASE)
UNIPROT_TOKEN_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{4,8}$")

CYP_TEXT_RE = re.compile(
    r"Cytochrome\s+P450\s+(\d+[A-Z]?\d*)|\b(CYP\s*\d+[A-Z0-9]*)\b",
    re.IGNORECASE,
)

NASAL_CYP_PATTERNS = (
    r"CYP1A1",
    r"CYP1A2",
    r"CYP2A6",
    r"CYP3A4",
    r"CYP3A5",
    r"CYP2C\d*",
    r"CYP2D6",
    r"Cytochrome\s+P450\s+1A1",
    r"Cytochrome\s+P450\s+1A2",
    r"Cytochrome\s+P450\s+2A6",
    r"Cytochrome\s+P450\s+3A4",
    r"Cytochrome\s+P450\s+3A5",
    r"Cytochrome\s+P450\s+2C",
    r"Cytochrome\s+P450\s+2D6",
)
NASAL_CYP_RE = re.compile("|".join(NASAL_CYP_PATTERNS), re.IGNORECASE)
CYP3A4_RE = re.compile(r"CYP\s*3A4|Cytochrome\s+P450\s+3A4", re.IGNORECASE)
PGP_RE = re.compile(
    r"P-glycoprotein|P-gp|ABCB1|MDR1|Multidrug resistance protein 1",
    re.IGNORECASE,
)

CNS_NAME_PATTERNS = (
    r"\bnmda\b",
    r"\bgaba\b",
    r"gamma-aminobutyric",
    r"dopamine",
    r"serotonin",
    r"adrenergic",
    r"opioid",
    r"cannabinoid",
    r"acetylcholine",
    r"muscarinic",
    r"nicotinic",
    r"glutamate",
    r"histamine\s*h\s*[13]",
    r"\b5-ht\b",
    r"beta\s*adrenoceptor",
    r"alpha\s*adrenoceptor",
)
CNS_GENE_PATTERNS = (
    r"^GRIN\d+",
    r"^DRD\d+",
    r"^HTR\d+",
    r"^GABR[A-Z]\d+",
    r"^CHRM\d+",
    r"^CHRN[A-Z]\d+",
    r"^OPRM?\d+",
    r"^OPRK\d+",
    r"^OPRD\d+",
    r"^CNR\d+",
    r"^HRH[13]$",
    r"^ADRA\d+",
    r"^ADRB\d+",
    r"^SLC6A\d+",
)
CNS_NAME_RE = re.compile("|".join(CNS_NAME_PATTERNS), re.IGNORECASE)
CNS_GENE_RE = re.compile("|".join(CNS_GENE_PATTERNS), re.IGNORECASE)


def parse_db_tokens(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [tok for tok in str(value).split() if tok.strip()]


def print_drugbank_field_samples(drugbank: pd.DataFrame) -> None:
    print("\n── DrugBank field format samples ─────────────────")
    for col in ("targets", "enzymes", "transporters"):
        if col not in drugbank.columns:
            print(f"  {col}: column not found")
            continue
        samples = drugbank[col].dropna().astype(str)
        samples = samples[samples.str.strip() != ""]
        print(f"  {col} ({len(samples):,} non-null) — space-delimited BE / UniProt IDs:")
        for idx, sample in enumerate(samples.head(3)):
            label = drugbank.loc[samples.index[idx], "name"] if "name" in drugbank.columns else f"row {idx}"
            text = sample if len(sample) <= 120 else sample[:117] + "..."
            print(f"    [{label}] {text}")


def load_uniprot_lookup() -> dict[str, dict[str, str]]:
    if not UNIPROT_LOOKUP_CSV.exists():
        print(f"  UniProt lookup not found: {UNIPROT_LOOKUP_CSV}")
        return {}
    lookup_df = pd.read_csv(UNIPROT_LOOKUP_CSV)
    lookup: dict[str, dict[str, str]] = {}
    for _, row in lookup_df.iterrows():
        uid = str(row["uniprot_id"]).strip()
        lookup[uid] = {
            "protein_name": str(row.get("protein_name") or "").strip(),
            "gene_symbol": str(row.get("gene_symbol") or "").strip(),
        }
    print(f"  UniProt lookup entries: {len(lookup):,}")
    return lookup


def load_drugbank_proteins() -> pd.DataFrame:
    if not DRUGBANK_PROTEINS_TSV.exists():
        print(f"  DrugBank proteins.tsv not found: {DRUGBANK_PROTEINS_TSV}")
        return pd.DataFrame(columns=["drugbank_id", "category", "uniprot_id", "entrez_gene_id"])
    proteins = pd.read_csv(DRUGBANK_PROTEINS_TSV, sep="\t")
    proteins["drugbank_id"] = proteins["drugbank_id"].astype(str)
    print(f"  DrugBank protein bindings: {len(proteins):,}")
    return proteins


def resolve_uniprot_name(uniprot_id: str, lookup: dict[str, dict[str, str]]) -> str:
    info = lookup.get(uniprot_id, {})
    return info.get("protein_name") or uniprot_id


def resolve_uniprot_gene(uniprot_id: str, lookup: dict[str, dict[str, str]]) -> str:
    info = lookup.get(uniprot_id, {})
    return info.get("gene_symbol") or ""


def proteins_for_drug(
    drugbank_id: str,
    category: str,
    proteins: pd.DataFrame,
    lookup: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    if proteins.empty or not drugbank_id:
        return [], []
    rows = proteins[(proteins["drugbank_id"] == drugbank_id) & (proteins["category"] == category)]
    names: list[str] = []
    genes: list[str] = []
    for _, row in rows.iterrows():
        uid = str(row.get("uniprot_id") or "").strip()
        if not uid:
            continue
        names.append(resolve_uniprot_name(uid, lookup))
        gene = resolve_uniprot_gene(uid, lookup)
        if gene:
            genes.append(gene)
    return names, genes


def is_valid_cyp_label(label: str) -> bool:
    return bool(re.match(r"^CYP\d+[A-Z]", label.upper()))


def extract_cyp_mentions(*texts: str | None) -> list[str]:
    found: list[str] = []
    for text in texts:
        if not text or (isinstance(text, float) and pd.isna(text)):
            continue
        for match in CYP_TEXT_RE.finditer(str(text)):
            label = (match.group(1) or match.group(2) or "").strip()
            if not label or label == "450":
                continue
            label = re.sub(r"\s+", "", label.upper())
            if not label.startswith("CYP"):
                label = f"CYP{label}"
            if is_valid_cyp_label(label):
                found.append(label)
    return found


def cyp_labels_from_names(names: list[str]) -> list[str]:
    labels: list[str] = []
    for name in names:
        match = re.search(r"Cytochrome\s+P450\s+(\d+[A-Z]\d*)", name, re.IGNORECASE)
        if match:
            candidate = f"CYP{match.group(1).upper()}"
            if is_valid_cyp_label(candidate):
                labels.append(candidate)
            continue
        match = re.search(r"\b(CYP\d+[A-Z0-9]*)\b", name, re.IGNORECASE)
        if match:
            candidate = re.sub(r"\s+", "", match.group(1).upper())
            if is_valid_cyp_label(candidate):
                labels.append(candidate)
    return labels


def is_cns_associated(*text_blobs: str | None) -> bool:
    for blob in text_blobs:
        if not blob or (isinstance(blob, float) and pd.isna(blob)):
            continue
        text = str(blob)
        if CNS_NAME_RE.search(text):
            return True
        for gene in re.split(r"[|,\s]+", text):
            gene = gene.strip()
            if gene and CNS_GENE_RE.search(gene):
                return True
    return False


def has_nasal_cyp_risk(*text_blobs: str | None) -> bool:
    for blob in text_blobs:
        if not blob or (isinstance(blob, float) and pd.isna(blob)):
            continue
        if NASAL_CYP_RE.search(str(blob)):
            return True
    return False


def is_cyp3a4_substrate(*text_blobs: str | None) -> bool:
    for blob in text_blobs:
        if not blob or (isinstance(blob, float) and pd.isna(blob)):
            continue
        if CYP3A4_RE.search(str(blob)):
            return True
    return False


def is_pgp_substrate(*text_blobs: str | None) -> bool:
    for blob in text_blobs:
        if not blob or (isinstance(blob, float) and pd.isna(blob)):
            continue
        if PGP_RE.search(str(blob)):
            return True
    return False


def extract_drugbank_annotations(
    db_row: pd.Series,
    proteins: pd.DataFrame,
    lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    drugbank_id = _clean_db_value(db_row.get("drugbank-id"))
    result: dict[str, Any] = {col: None for col in ANNOTATION_COLUMNS}
    for flag_col in ("cns_target", "nasal_cyp_risk", "cyp3a4_substrate", "pgp_substrate"):
        result[flag_col] = False

    target_tokens = parse_db_tokens(db_row.get("targets"))
    enzyme_tokens = parse_db_tokens(db_row.get("enzymes"))
    transporter_tokens = parse_db_tokens(db_row.get("transporters"))

    if target_tokens:
        result["target_count"] = len(target_tokens)
        names, genes = proteins_for_drug(str(drugbank_id), "target", proteins, lookup)
        if not names:
            names = [tok for tok in target_tokens if not BE_TOKEN_RE.match(tok)]
        result["target_names"] = pipe_join_unique(names)
        result["target_genes"] = pipe_join_unique(genes)
        result["primary_target"] = names[0] if names else None

    enzyme_names: list[str] = []
    if enzyme_tokens:
        result["enzyme_count"] = len(enzyme_tokens)
        prot_names, _ = proteins_for_drug(str(drugbank_id), "enzyme", proteins, lookup)
        enzyme_names.extend(prot_names)
        for tok in enzyme_tokens:
            if UNIPROT_TOKEN_RE.match(tok):
                enzyme_names.append(resolve_uniprot_name(tok, lookup))

    metabolism = _clean_db_value(db_row.get("metabolism"))
    cyp_from_text = extract_cyp_mentions(metabolism, db_row.get("mechanism-of-action"))
    enzyme_names.extend(cyp_from_text)
    if enzyme_names or enzyme_tokens or metabolism:
        result["metabolizing_enzymes"] = pipe_join_unique(enzyme_names)
        cyp_labels = cyp_labels_from_names(enzyme_names)
        cyp_labels.extend(extract_cyp_mentions(metabolism, db_row.get("mechanism-of-action")))
        result["cyp_enzymes"] = pipe_join_unique(cyp_labels)
        if not result["enzyme_count"] and (enzyme_names or cyp_from_text):
            result["enzyme_count"] = len(enzyme_tokens) or len(enzyme_names)

    if transporter_tokens:
        result["transporter_count"] = len(transporter_tokens)
        t_names, _ = proteins_for_drug(str(drugbank_id), "transporter", proteins, lookup)
        if not t_names:
            t_names = [tok for tok in transporter_tokens if not BE_TOKEN_RE.match(tok)]
        result["transporters"] = pipe_join_unique(t_names)

    cns_text = [
        result.get("target_names"),
        result.get("target_genes"),
        result.get("primary_target"),
        db_row.get("mechanism-of-action"),
        db_row.get("pharmacodynamics"),
    ]
    result["cns_target"] = is_cns_associated(*cns_text)

    cyp_text = [result.get("metabolizing_enzymes"), result.get("cyp_enzymes"), metabolism]
    result["nasal_cyp_risk"] = has_nasal_cyp_risk(*cyp_text)
    result["cyp3a4_substrate"] = is_cyp3a4_substrate(*cyp_text)

    trans_text = [result.get("transporters")]
    result["pgp_substrate"] = is_pgp_substrate(*trans_text)

    return result


def merge_drugbank_annotations(
    df: pd.DataFrame,
    drugbank: pd.DataFrame,
    proteins: pd.DataFrame,
    lookup: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "target_data": 0,
        "cns_active": 0,
        "enzyme_data": 0,
        "nasal_cyp_risk": 0,
        "transporter_data": 0,
        "pgp_substrate": 0,
    }
    df = df.copy()
    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = None

    if drugbank.empty:
        return df, stats

    db_by_key = drugbank.set_index("name_key", drop=False)

    for idx, row in df.iterrows():
        if row["name_key"] not in db_by_key.index:
            continue
        db_row = db_by_key.loc[row["name_key"]]
        if isinstance(db_row, pd.DataFrame):
            db_row = db_row.iloc[0]
        ann = extract_drugbank_annotations(db_row, proteins, lookup)
        for col, val in ann.items():
            df.at[idx, col] = val
        if ann.get("target_count"):
            stats["target_data"] += 1
        if ann.get("cns_target"):
            stats["cns_active"] += 1
        if ann.get("metabolizing_enzymes") or ann.get("cyp_enzymes"):
            stats["enzyme_data"] += 1
        if ann.get("nasal_cyp_risk"):
            stats["nasal_cyp_risk"] += 1
        if ann.get("transporters"):
            stats["transporter_data"] += 1
        if ann.get("pgp_substrate"):
            stats["pgp_substrate"] += 1

    return df, stats


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).lower().strip())


def values_meaningfully_differ(left: Any, right: Any) -> bool:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return False
    return a != b


def _clean_db_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def load_drugbank() -> pd.DataFrame:
    """Load Kaggle DrugBank export indexed by normalized drug name."""
    if not DRUGBANK_CSV.exists():
        print(f"  DrugBank file not found: {DRUGBANK_CSV}")
        return pd.DataFrame()

    db = pd.read_csv(DRUGBANK_CSV, low_memory=False)
    db["name_key"] = db["name"].map(normalize_name)
    db = db[db["name_key"] != ""].copy()
    db["_richness"] = db.notna().sum(axis=1)
    db = db.sort_values("_richness", ascending=False)
    db = db.drop_duplicates(subset="name_key", keep="first")
    db = db.drop(columns=["_richness"])
    print(f"  DrugBank records indexed: {len(db):,}")
    print_drugbank_field_samples(db)
    return db


def merge_drugbank_priorities(df: pd.DataFrame, drugbank: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply field-level DrugBank vs ChEMBL source priority with conflict tracking."""
    stats = {
        "mechanism_drugbank_primary": 0,
        "mechanism_chembl_fallback": 0,
        "mechanism_conflicts": 0,
        "indication_conflicts": 0,
        "mw_conflicts": 0,
        "physical_state_populated": 0,
        "pk_populated": 0,
        "drugbank_matches": 0,
    }

    df = df.copy()
    for col in [
        "physical_state",
        *DRUGBANK_PK_MAP.values(),
        "mechanism_of_action_chembl",
        "mechanism_of_action_conflict",
        "indication_class_chembl",
        "indication_class_conflict",
        "molecular_weight_db",
        "mw_conflict",
        "field_sources",
        "drugbank_id",
    ]:
        if col not in df.columns:
            df[col] = None

    if drugbank.empty:
        df["field_sources"] = df.apply(lambda _: json.dumps({}), axis=1)
        return df, stats

    db_by_key = drugbank.set_index("name_key", drop=False)

    def merge_row(row: pd.Series) -> pd.Series:
        sources: dict[str, str] = {}
        chembl_mechanism = _clean_db_value(row.get("mechanism_of_action"))
        chembl_indication = _clean_db_value(row.get("indication_class"))
        chembl_mw = _to_float(row.get("molecular_weight"))

        for field in CHEMBL_PRIMARY_FIELDS:
            if field == "chembl_max_phase":
                if pd.notna(row.get("chembl_max_phase")):
                    sources["max_phase"] = "chembl"
            elif pd.notna(row.get(field)) and str(row.get(field)).strip() != "":
                sources[field] = "chembl"

        if row["name_key"] not in db_by_key.index:
            row["field_sources"] = json.dumps(sources, sort_keys=True)
            return row

        stats["drugbank_matches"] += 1
        db_row = db_by_key.loc[row["name_key"]]
        if isinstance(db_row, pd.DataFrame):
            db_row = db_row.iloc[0]

        row["drugbank_id"] = _clean_db_value(db_row.get("drugbank-id"))
        row["sources"] = add_source(row.get("sources"), "drugbank")

        db_mechanism = _clean_db_value(db_row.get("mechanism-of-action"))
        db_indication = _clean_db_value(db_row.get("indication"))
        db_state = _clean_db_value(db_row.get("state"))
        db_mass = _to_float(db_row.get("average-mass"))

        if db_mechanism:
            row["mechanism_of_action"] = db_mechanism
            sources["mechanism_of_action"] = "drugbank"
            stats["mechanism_drugbank_primary"] += 1
            if chembl_mechanism and values_meaningfully_differ(db_mechanism, chembl_mechanism):
                row["mechanism_of_action_chembl"] = chembl_mechanism
                row["mechanism_of_action_conflict"] = True
                stats["mechanism_conflicts"] += 1
        elif chembl_mechanism:
            row["mechanism_of_action"] = chembl_mechanism
            sources["mechanism_of_action"] = "chembl"
            stats["mechanism_chembl_fallback"] += 1

        if db_indication:
            row["indication_class"] = db_indication
            sources["indication_class"] = "drugbank"
            if chembl_indication and values_meaningfully_differ(db_indication, chembl_indication):
                row["indication_class_chembl"] = chembl_indication
                row["indication_class_conflict"] = True
                stats["indication_conflicts"] += 1
        elif chembl_indication:
            row["indication_class"] = chembl_indication
            sources["indication_class"] = "chembl"

        if db_state:
            row["physical_state"] = db_state
            sources["physical_state"] = "drugbank"
            stats["physical_state_populated"] += 1

        if db_mass is not None and chembl_mw is not None and abs(db_mass - chembl_mw) > 1.0:
            row["molecular_weight_db"] = db_mass
            row["mw_conflict"] = True
            stats["mw_conflicts"] += 1
        elif db_mass is not None:
            row["molecular_weight_db"] = db_mass

        pk_any = False
        for db_col, out_col in DRUGBANK_PK_MAP.items():
            value = _clean_db_value(db_row.get(db_col))
            if value is not None:
                row[out_col] = value
                sources[out_col] = "drugbank"
                pk_any = True
        if pk_any:
            stats["pk_populated"] += 1

        row["field_sources"] = json.dumps(sources, sort_keys=True)
        return row

    # Use explicit loop so stats mutate correctly (apply with closure is awkward for counters)
    rows_out: list[pd.Series] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging DrugBank priorities"):
        rows_out.append(merge_row(row.copy()))
    return pd.DataFrame(rows_out), stats


def print_drugbank_summary(stats: dict[str, int]) -> None:
    print("\n── Source priority & conflicts ───────────────────")
    print(f"  DrugBank name matches:           {stats.get('drugbank_matches', 0):>7,}")
    print(f"  Mechanism: DrugBank primary     {stats.get('mechanism_drugbank_primary', 0):>7,}")
    print(f"  Mechanism: ChEMBL fallback used  {stats.get('mechanism_chembl_fallback', 0):>7,}")
    print(f"  Mechanism conflicts flagged      {stats.get('mechanism_conflicts', 0):>7,}")
    print(f"  Indication conflicts flagged     {stats.get('indication_conflicts', 0):>7,}")
    print(f"  MW conflicts (>1 Da difference)  {stats.get('mw_conflicts', 0):>7,}")
    print(f"  Physical state (DrugBank only)   {stats.get('physical_state_populated', 0):>7,}")
    print(f"  PK fields populated (DrugBank)   {stats.get('pk_populated', 0):>7,}")
    print("\n── DrugBank targets/enzymes/transporters ─────────")
    print(f"  Target data extracted:         {stats.get('target_data', 0):>7,}")
    print(f"  CNS-active drugs flagged:      {stats.get('cns_active', 0):>7,}")
    print(f"  Enzyme/CYP data:               {stats.get('enzyme_data', 0):>7,}")
    print(f"  Nasal CYP risk flagged:        {stats.get('nasal_cyp_risk', 0):>7,}")
    print(f"  Transporter data:              {stats.get('transporter_data', 0):>7,}")
    print(f"  P-gp substrates flagged:       {stats.get('pgp_substrate', 0):>7,}")


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
    print(f"  Mechanism of action data:      {stats['mechanism_rows']:>7,}")
    print(f"  ATC codes:                     {stats['atc_rows']:>7,}")
    print(f"  MeSH headings:                 {stats['mesh_rows']:>7,}")
    print(f"  pKa predicted:                 {stats['pka_rows']:>7,}")
    print(f"  logD calculated:               {stats['logd_rows']:>7,}")
    print(f"  Dose data available:           {stats['dose_rows']:>7,}")
    print(f"  Dose ≤ 100mg (NOSA feasible):  {stats['dose_feasible']:>7,}")
    print(f"  Dose > 100mg (excluded):       {stats['dose_excluded']:>7,}")
    print(f"  Dose unknown:                  {stats['dose_unknown']:>7,}")
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
        "mechanism_rows": 0,
        "atc_rows": 0,
        "mesh_rows": 0,
        "pka_rows": 0,
        "logd_rows": 0,
        "dose_rows": 0,
        "dose_feasible": 0,
        "dose_excluded": 0,
        "dose_unknown": 0,
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

    drugbank = load_drugbank()
    proteins = load_drugbank_proteins()
    uniprot_lookup = load_uniprot_lookup()
    merged, drugbank_stats = merge_drugbank_priorities(merged, drugbank)
    merged, annotation_stats = merge_drugbank_annotations(merged, drugbank, proteins, uniprot_lookup)
    stats.update(drugbank_stats)
    stats.update(annotation_stats)

    print("Calculating predicted pKa and logD from SMILES...")
    merged = add_pka_logd_predictions(merged)

    merged = flag_nosa_candidates(merged)
    merged, stats["enriched_carried"] = carry_forward_enrichment(merged)
    print(f"  Enriched rows carried forward: {stats['enriched_carried']:,}")

    column_order = [
        "name",
        "chembl_id",
        "chembl_max_phase",
        "indication_class",
        "mesh_heading",
        "indication_class_chembl",
        "indication_class_conflict",
        "atc_code",
        "atc_level1",
        "mechanism_of_action",
        "mechanism_of_action_all",
        "mechanism_of_action_chembl",
        "mechanism_of_action_conflict",
        "target_name",
        "target_chembl_id",
        "action_type",
        "target_names",
        "target_genes",
        "target_count",
        "primary_target",
        "cns_target",
        "metabolizing_enzymes",
        "cyp_enzymes",
        "enzyme_count",
        "nasal_cyp_risk",
        "cyp3a4_substrate",
        "transporters",
        "transporter_count",
        "pgp_substrate",
        "synonyms",
        "max_synonym_count",
        "molecular_weight",
        "molecular_weight_db",
        "mw_conflict",
        "logP",
        "PSA",
        "H_donors",
        "H_acceptors",
        "rotatable_bonds",
        "pka_predicted",
        "logD_pH6",
        "logD_pH74",
        "ionization_class",
        "physical_state",
        "absorption",
        "half_life",
        "protein_binding",
        "metabolism",
        "route_of_elimination",
        "volume_of_distribution",
        "clearance",
        "toxicity",
        "SMILES",
        "smiles_source",
        "source",
        "sources",
        "ingredient",
        "earliest_patent_expiry",
        "earliest_exclusivity_date",
        "ema_approved",
        "pmda_approved",
        "route_of_administration",
        "dosage_form",
        "max_clinical_dose_mg",
        "max_clinical_dose_phase",
        "dose_feasible_nosa",
        "drugbank_id",
        "field_sources",
        "nosa_candidate",
        *ENRICHMENT_COLUMNS,
    ]
    for col in column_order:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[column_order]

    stats["mechanism_rows"] = int(merged["mechanism_of_action"].notna().sum())
    stats["atc_rows"] = int(merged["atc_code"].notna().sum())
    stats["mesh_rows"] = int(merged["mesh_heading"].notna().sum())
    stats["pka_rows"] = int(merged["pka_predicted"].notna().sum())
    stats["logd_rows"] = int(merged["logD_pH6"].notna().sum())
    stats["dose_rows"] = int(merged["max_clinical_dose_mg"].notna().sum())
    stats["dose_feasible"] = int((merged["dose_feasible_nosa"] == True).sum())
    stats["dose_excluded"] = int((merged["dose_feasible_nosa"] == False).sum())
    stats["dose_unknown"] = int(merged["dose_feasible_nosa"].isna().sum())
    stats["total_rows"] = len(merged)
    return merged, stats


def main() -> None:
    args = parse_args()
    df, stats = build_database(skip_chembl=args.skip_chembl)
    df.to_csv(OUTPUT_CSV, index=False)
    print_build_summary(stats)
    print_drugbank_summary(stats)


if __name__ == "__main__":
    main()
