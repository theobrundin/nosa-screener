# NOSA Drug Database — Presentation Summary
*Prepared June 17, 2026*

## What This Is

A screening platform for **nasal vapor drug delivery** (NOSA's noseplug device). It ranks ~10,500 FDA/EMA-approved small molecules by physicochemical similarity to **memantine**, the lead benchmark compound, across 7 properties relevant to polymer embedding and vapor release.

---

## Tonight's Demo

```bash
cd /Users/theobrundin/Desktop/NOSA-DB
source .venv/bin/activate
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### App walkthrough (5 min)

1. **Sidebar filters** — Each property has **Min / Max number fields** (type exact values) plus a slider. Default filters show ~9,200 drugs.
2. **Ranked table** — Sorted by composite score (0–100). NOSA candidates highlighted in cyan.
3. **Scatter plot** — logP vs molecular weight; gold diamond = memantine benchmark.
4. **Download** — Export filtered results as CSV.

### Key talking points

| Topic | Detail |
|---|---|
| Database size | **10,545** small molecules (ChEMBL phase ≥ 2 + 11 manual volatiles) |
| Approved drugs | **2,617** at ChEMBL phase 4 (FDA-approved) |
| Composite score | 7 equal-weight criteria vs memantine benchmark |
| NOSA candidates | 10 flagged compounds (memantine, nicotine, melatonin, etc.) |
| PubChem enrichment | **98.5%** CID resolution; VP/MP sparse in PubChem itself (~4%/16%) |
| Data sources | ChEMBL, FDA Orange Book, DrugCentral, FDA+EMA+PMDA, PubChem |

### Memantine score note

Memantine scores ~80/100 (not 100) because the **melting point ideal is 130°C** (platform thermal target for injection molding), while memantine's measured MP is ~290°C. The 5 structural properties match perfectly; MP and VP pull the score down by design.

### Nicotine note

Nicotine has a melting point of **−79°C** (liquid at room temperature — correct PubChem data). The MP filter now spans −100 to 400°C so nicotine appears in the default view.

---

## Changes Made Today

### 1. Database pipeline (`build_database.py`)
- Expanded from ~3,300 to **10,545 rows** via ChEMBL (max phase ≥ 2)
- Added Orange Book patent/exclusivity dates
- Added EMA/PMDA regulatory flags (89 compounds)
- DrugCentral SMILES backfill for missing structures
- **11 manual volatile compounds** (linalool, limonene, vanillin, etc.)
- Deduplication with ChEMBL priority (1,230 duplicates removed)
- Fixed PubChem property fetch (`ConnectivitySMILES` instead of deprecated `CanonicalSMILES`)
- Enrichment carry-forward from previous enriched file

### 2. PubChem enrichment (`enrich_pubchem.py`)
- Full run completed on all 10,545 rows (~3.8 hours)
- **98.5%** PubChem CID resolution
- **382** vapor pressure values (3.6%) — limited by PubChem data availability
- **1,707** melting point values (16.2%)
- **473** boiling point values (4.5%)
- Resumable checkpoints every 500 rows

### 3. Streamlit app (`app.py`)
- NOSA branding (teal `#1E6464`, accent `#00BFBF`)
- 7-criteria composite scoring (MW, logP, PSA, H-donors, rotatable bonds, VP, MP)
- PubChem VP/MP filter sliders with platform feasibility notes
- NOSA candidate row highlighting
- logP vs MW scatter with memantine benchmark marker
- CSV export of filtered results
- Auto-loads enriched CSV when available
- **Keyboard-editable Min/Max inputs** on all filters (today's final update)
- MP filter extended to −100°C for compounds like nicotine

### 4. Validation (`sanity_check.py`)
- Automated checks for schema, duplicates, coverage, NOSA flags, scoring
- All checks passing

---

## Current Database State

| Field | Coverage |
|---|---|
| Total rows | 10,545 |
| ChEMBL phase 4 | 2,617 |
| ChEMBL phase 3 | 1,062 |
| ChEMBL phase 2 | 6,855 |
| Manual volatiles | 11 |
| SMILES | 97.2% |
| Molecular weight | 98.8% |
| PubChem CID | 98.5% |
| Vapor pressure | 3.6% |
| Melting point | 16.2% |
| Orange Book ingredient | 17.8% |
| Patent expiry dates | 7.9% |
| EMA/PMDA approved flags | 89 |
| Duplicate names | 0 |

### Files

| File | Purpose |
|---|---|
| `nosa_drug_database.csv` | Master database (built by pipeline) |
| `nosa_drug_database_enriched.csv` | Master + PubChem VP/MP/BP (app loads this) |
| `build_database.py` | Rebuild pipeline |
| `enrich_pubchem.py` | PubChem enrichment (resumable) |
| `app.py` | Streamlit screener |
| `sanity_check.py` | Validation script |

### Rebuild commands

```bash
python build_database.py              # Full rebuild (~17 min first time)
python build_database.py --skip-chembl  # Fast rebuild from cache
python enrich_pubchem.py                # Fill VP/MP (resumable)
python sanity_check.py                  # Validate everything
```

---

## Known Limitations (for Q&A)

1. **Low VP coverage (3.6%)** — PubChem simply doesn't report vapor pressure for most drugs. Missing VP values pass filters but score 0 on that criterion.
2. **EMA/PMDA same flag** — Source file has no separate EMA vs PMDA columns; both flags set together.
3. **Indication class sparse (0.1%)** — Only manual volatiles tagged; ChEMBL indication data not yet integrated.
4. **Composite score weighting** — All 7 criteria weighted equally; could be tuned per platform priorities.

---

## NOSA Candidates (all 10 flagged)

| Drug | Default view rank | Score |
|---|---|---|
| Dimethyl fumarate | ~#42 | ~89 |
| Memantine | ~#126 | ~80 |
| Melatonin | ~#238 | ~77 |
| Propranolol | ~#284 | ~76 |
| Morphine | ~#397 | ~73 |
| Nicotine | visible (MP filter fixed) | — |
| Valproic acid | ~#1,297 | ~66 |
| Vortioxetine | ~#1,836 | ~64 |
| Brivaracetam | ~#2,705 | ~62 |
| Zolmitriptan | ~#3,351 | ~61 |

*Ranks approximate; depend on current filter settings.*
