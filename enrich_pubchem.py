#!/usr/bin/env python3
"""Enrich NOSA drug database with PubChem physical properties via PUG-REST."""

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
DEFAULT_OUTPUT = ROOT / "nosa_drug_database_enriched.csv"

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_VIEW_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"

REQUEST_SLEEP_S = 0.22
MAX_RETRIES = 3
SAVE_EVERY = 500
PROGRESS_EVERY = 100

ENRICHMENT_COLUMNS = [
    "pubchem_cid",
    "vapor_pressure_mmhg",
    "vapor_pressure_raw",
    "melting_point_c",
    "melting_point_raw",
    "boiling_point_c",
    "boiling_point_raw",
]

PROPERTY_HEADINGS = {
    "Vapor Pressure": ("vapor_pressure_mmhg", "vapor_pressure_raw"),
    "Melting Point": ("melting_point_c", "melting_point_raw"),
    "Boiling Point": ("boiling_point_c", "boiling_point_raw"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich NOSA drug database with PubChem physical properties."
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
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT.name})",
    )
    return parser.parse_args()


def request_json(url: str, session: requests.Session) -> dict[str, Any] | None:
    """GET JSON with retries; return None on 404 or persistent failure."""
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


def resolve_cid(name: str, session: requests.Session) -> int | None:
    encoded = quote(str(name), safe="")
    url = f"{PUG_BASE}/compound/name/{encoded}/JSON"
    data = request_json(url, session)
    if not data:
        return None
    try:
        id_block = data["PC_Compounds"][0]["id"]
        if "cid" in id_block:
            return int(id_block["cid"])
        nested = id_block.get("id", {})
        if "cid" in nested:
            return int(nested["cid"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def extract_strings_from_information(info: dict[str, Any]) -> list[str]:
    strings: list[str] = []
    for item in info.get("Value", {}).get("StringWithMarkup", []) or []:
        if isinstance(item, dict) and item.get("String"):
            strings.append(str(item["String"]).strip())
    if not strings and info.get("Name"):
        strings.append(str(info["Name"]).strip())
    return strings


def walk_sections(
    sections: list[dict[str, Any]] | None,
    found: dict[str, list[str]],
) -> None:
    if not sections:
        return
    for section in sections:
        heading = section.get("TOCHeading", "")
        if heading in PROPERTY_HEADINGS and heading not in found:
            collected: list[str] = []
            for info in section.get("Information", []) or []:
                collected.extend(extract_strings_from_information(info))
            if collected:
                found[heading] = collected
        walk_sections(section.get("Section"), found)


def fetch_property_strings(cid: int, session: requests.Session) -> dict[str, list[str]]:
    url = f"{PUG_VIEW_BASE}/{cid}/JSON"
    data = request_json(url, session)
    if not data:
        return {}
    record = data.get("Record", {})
    found: dict[str, list[str]] = {}
    walk_sections(record.get("Section"), found)
    return found


def normalize_scientific(text: str) -> str:
    """Convert PubChem-style scientific notation (e.g. 4.0X10-2) to standard form."""
    return re.sub(
        r"(\d+(?:\.\d+)?)\s*[xX×]\s*10\s*([+-]?\d+)",
        lambda m: f"{float(m.group(1))}e{m.group(2)}",
        text,
    )


def first_number(text: str) -> float | None:
    cleaned = normalize_scientific(text.replace(",", ""))
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def parse_vapor_pressure_mmhg(raw: str) -> float | None:
    value = first_number(raw)
    if value is None:
        return None
    lower = raw.lower()
    if re.search(r"\b(kpa|kilopascal)\b", lower):
        return value * 7.50062
    if re.search(r"\b(pa|pascal)\b", lower) and "kpa" not in lower:
        return value / 133.322
    if re.search(r"\b(atm|atmosphere)\b", lower):
        return value * 760.0
    if re.search(r"\b(bar)\b", lower) and "mbar" not in lower:
        return value * 750.062
    if re.search(r"\b(mbar|millibar|hpa|hectopascal)\b", lower):
        return value * 0.750062
    # torr, mmHg, mm Hg, or no unit — assume mmHg
    return value


def parse_temperature_c(raw: str) -> float | None:
    value = first_number(raw)
    if value is None:
        return None
    lower = raw.lower()
    if re.search(r"\b(f|°f|deg\s*f|fahrenheit)\b", lower):
        return (value - 32.0) * 5.0 / 9.0
    if re.search(r"\b(k|°k|kelvin)\b", lower):
        return value - 273.15
    # °C, deg C, Celsius, or no unit — assume °C
    return value


def parse_property_value(heading: str, raw_strings: list[str]) -> tuple[float | None, str | None]:
    if not raw_strings:
        return None, None
    for raw in raw_strings:
        if heading == "Vapor Pressure":
            value = parse_vapor_pressure_mmhg(raw)
        else:
            value = parse_temperature_c(raw)
        if value is not None:
            return value, raw
    return None, raw_strings[0]


def empty_enrichment() -> dict[str, Any]:
    return {col: None for col in ENRICHMENT_COLUMNS}


def enrich_row(name: str, session: requests.Session) -> dict[str, Any]:
    result = empty_enrichment()
    cid = resolve_cid(name, session)
    if cid is None:
        return result
    result["pubchem_cid"] = cid

    property_strings = fetch_property_strings(cid, session)
    for heading, (value_col, raw_col) in PROPERTY_HEADINGS.items():
        strings = property_strings.get(heading, [])
        value, raw = parse_property_value(heading, strings)
        result[value_col] = value
        result[raw_col] = raw
    return result


def load_existing_enrichment(
    input_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load enrichment already present in output file or carried in input columns."""
    enriched_rows: dict[str, dict[str, Any]] = {}

    if output_path.exists():
        existing = pd.read_csv(output_path)
        for _, row in existing.iterrows():
            enriched_rows[str(row["name"])] = {
                col: row.get(col) for col in ENRICHMENT_COLUMNS if col in existing.columns
            }

    for _, row in input_df.iterrows():
        name = str(row["name"])
        if name in enriched_rows and pd.notna(enriched_rows[name].get("pubchem_cid")):
            continue
        if pd.notna(row.get("pubchem_cid")):
            enriched_rows[name] = {
                col: row.get(col) if col in input_df.columns else None
                for col in ENRICHMENT_COLUMNS
            }

    return enriched_rows


def merge_and_save(
    input_df: pd.DataFrame,
    enriched_rows: dict[str, dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    df = input_df.copy()
    for col in ENRICHMENT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    for name, values in enriched_rows.items():
        mask = df["name"] == name
        for col, val in values.items():
            df.loc[mask, col] = val

    df.to_csv(output_path, index=False)
    return df


def print_progress(
    processed: int,
    total: int,
    vp_found: int,
    mp_found: int,
) -> None:
    pct = 100.0 * processed / total if total else 0.0
    print(
        f"Progress: {processed}/{total} ({pct:.1f}%) | "
        f"vapor pressure: {vp_found} | melting point: {mp_found}"
    )


def print_summary(df: pd.DataFrame, total_input: int, output_path: Path) -> None:
    n = len(df)
    cid_resolved = df["pubchem_cid"].notna().sum()
    vp = df["vapor_pressure_mmhg"].notna().sum()
    mp = df["melting_point_c"].notna().sum()
    bp = df["boiling_point_c"].notna().sum()

    def pct(count: int) -> str:
        return f"{count}/{n} ({100.0 * count / n:.1f}%)" if n else "0/0"

    print("\n=== Enrichment summary ===")
    print(f"Total rows: {n} (input had {total_input})")
    print(f"CID resolved: {pct(cid_resolved)}")
    print(f"Vapor pressure: {pct(vp)}")
    print(f"Melting point: {pct(mp)}")
    print(f"Boiling point: {pct(bp)}")
    print(f"Saved to: {output_path}")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    input_df = pd.read_csv(args.input)
    total = len(input_df)
    enriched_rows = load_existing_enrichment(input_df, args.output)

    to_process = input_df[~input_df["name"].isin(enriched_rows)]
    skipped = total - len(to_process)
    if skipped:
        print(f"Resuming: skipping {skipped} rows with existing PubChem data")

    session = requests.Session()
    session.headers.update({"User-Agent": "NOSA-DB/1.0 (drug screening research)"})

    processed_new = 0
    vp_found = int(
        sum(1 for v in enriched_rows.values() if pd.notna(v.get("vapor_pressure_mmhg")))
    )
    mp_found = int(
        sum(1 for v in enriched_rows.values() if pd.notna(v.get("melting_point_c")))
    )

    for _, row in to_process.iterrows():
        name = str(row["name"])
        enriched_rows[name] = enrich_row(name, session)
        processed_new += 1

        if enriched_rows[name].get("vapor_pressure_mmhg") is not None:
            vp_found += 1
        if enriched_rows[name].get("melting_point_c") is not None:
            mp_found += 1

        if processed_new % SAVE_EVERY == 0:
            merge_and_save(input_df, enriched_rows, args.output)
            print(f"  [checkpoint] saved {len(enriched_rows)} rows to {args.output.name}")

        if processed_new % PROGRESS_EVERY == 0:
            print_progress(len(enriched_rows), total, vp_found, mp_found)

    final_df = merge_and_save(input_df, enriched_rows, args.output)
    print_summary(final_df, total, args.output)


if __name__ == "__main__":
    main()
