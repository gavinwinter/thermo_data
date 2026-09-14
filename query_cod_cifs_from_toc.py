"""
Second-pass CIF retrieval for Barin toc entries that Materials Project didn't
have (query_mp_cifs_from_toc.py's cif_unmatched_mp.csv), using the free,
open Crystallography Open Database (COD) REST search
(https://www.crystallography.net/cod/) -- unlike ICSD, COD is CC0 and its
search/download endpoints are meant for exactly this kind of automated use.

No API key needed for this stage. Run after query_mp_cifs_from_toc.py, with
a Python environment that has pymatgen + pandas installed (see README):

    python query_cod_cifs_from_toc.py

Nothing else needs to change: paths are resolved relative to this script's
own location, so the repo can live anywhere.

For each row in cif_unmatched_mp.csv:
  0. Set aside rows with no parseable composition (unparseable_formula /
     suspected_ocr_corruption) and rows that are room-temperature organic
     liquids (no realistic crystal structure applies) -- neither is worth
     sending to COD. Both go straight into the final outputs.
  1. Query COD by element set (exact element count match, any stoichiometry).
  2. Parse each candidate's reported formula and compare its element ratios
     to our target composition (tight tolerance -- this is a stoichiometry
     check, not a fuzzy search, since a wrong match here would be worse than
     no match).
  3. On a hit, download the actual CIF from crystallography.net/cod/<id>.cif
     into cifs_barin/, alongside the Materials Project results.
  4. Write cif_matches_cod.csv, cif_no_structure_expected.csv (organic
     liquids, informational only), and cif_unmatched_final.csv -- the
     genuine remaining candidates for manual ICSD lookup.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from pymatgen.core.composition import Composition
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CIF_DIR = ROOT / "cifs_barin"
UNMATCHED_IN = ROOT / "cif_unmatched_mp.csv"
COD_SUMMARY_CSV = ROOT / "cif_matches_cod.csv"
NO_STRUCTURE_CSV = ROOT / "cif_no_structure_expected.csv"
STILL_UNMATCHED_CSV = ROOT / "cif_unmatched_final.csv"

CIF_DIR.mkdir(exist_ok=True)

RATIO_TOL = 0.02  # max per-element absolute deviation in fractional composition to accept a match


def cod_search(elements):
    params = {"format": "json", "strictmin": str(len(elements)), "strictmax": str(len(elements))}
    for i, el in enumerate(elements, start=1):
        params[f"el{i}"] = el
    url = "https://www.crystallography.net/cod/result.php?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  COD query error: {e}")
        return []


def parse_cod_formula(formula_str):
    """COD 'formula' field looks like '- Al2 O5 Si -'. Parse into a Composition."""
    tokens = formula_str.replace("-", " ").split()
    pieces = []
    i = 0
    while i < len(tokens):
        el = tokens[i]
        amt = "1"
        if i + 1 < len(tokens) and re.match(r"^[\d.]+$", tokens[i + 1]):
            amt = tokens[i + 1]
            i += 2
        else:
            i += 1
        pieces.append(f"{el}{amt}")
    try:
        return Composition("".join(pieces))
    except Exception:
        return None


def ratio_error(target: Composition, candidate: Composition):
    """Max per-element absolute deviation in fractional composition, or None
    if the candidate doesn't even have the same element set."""
    t = target.fractional_composition.get_el_amt_dict()
    c = candidate.fractional_composition.get_el_amt_dict()
    if set(t.keys()) != set(c.keys()):
        return None
    return max(abs(t[el] - c[el]) for el in t)


def best_candidate(target_comp, docs, tol=RATIO_TOL):
    """Pick the candidate with the smallest stoichiometric deviation from the
    target (not just any passing candidate), and only accept it if that
    deviation is within tol -- this favors e.g. the closest wuestite
    composition over a same-element-set but chemically distinct phase
    (troilite FeS vs. pyrrhotite Fe0.877S) that happens to also pass a loose
    threshold.
    """
    scored = []
    for d in docs:
        formula_str = d.get("formula") or d.get("calcformula")
        if not formula_str:
            continue
        cand_comp = parse_cod_formula(formula_str)
        if cand_comp is None:
            continue
        err = ratio_error(target_comp, cand_comp)
        if err is None or err > tol:
            continue
        has_sg = 1 if d.get("sg") else 0
        scored.append((err, -has_sg, int(d["file"]), d))
    if not scored:
        return None
    scored.sort()
    return scored[0][3]


def sanitize_filename(formula: str) -> str:
    # '(' ')' are kept as-is (they're valid on every filesystem this repo
    # runs on, just needing quotes in a shell command) so hydrate/complex
    # formulas like 'Al2(SO4)3' read as themselves in the filename instead
    # of becoming 'Al2_SO4_3'. '*' (Barin's hydrate separator, e.g.
    # 'ZnSO4*7H2O') is kept as-is too, matching how read_all_thermodata_pdf.py
    # names the corresponding .json -- so a hydrate's .cif and .json share
    # the same base name instead of one saying '_hyd_' and the other '*'.
    return re.sub(r"[^A-Za-z0-9_.()\[\]*-]", "_", formula)


# Elements that appear in this dataset's actual organic entries (hydrocarbons,
# alcohols, acids: C/H, plus the O/S/N/halogens in their functional groups). A
# formula needs C and H *and* nothing outside this set to count as organic --
# otherwise something like NaHCO3 (sodium bicarbonate, a genuine mineral with
# Na present) would get misclassified as organic just because it contains
# both C and H.
_ORGANIC_ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def is_organic_liquid(bare_formula) -> bool:
    """Contains both carbon and hydrogen, with no elements outside the
    CHNOPS+halogens set -- these are the room-temperature organic
    liquids/gases in Barin's table, none of which have a meaningful
    solid-state crystal structure to look up (and, per query_mp_cifs_from_toc's
    organic_ambiguous_isomers routing, can't be safely guessed by stoichiometry
    anyway since isomers share a formula). Composition-based rather than a
    hardcoded name-keyword list so it generalizes to every such entry, not
    just the ones someone thought to list.
    """
    try:
        comp = Composition(str(bare_formula))
    except Exception:
        return False
    elements = {str(e) for e in comp.elements}
    return "C" in elements and "H" in elements and elements <= _ORGANIC_ALLOWED_ELEMENTS


def main():
    df = pd.read_csv(UNMATCHED_IN)

    is_organic = df["bare_formula"].apply(is_organic_liquid)
    no_structure = df[is_organic]
    no_structure.to_csv(NO_STRUCTURE_CSV, index=False)

    searchable = df[df["reduced_formula"].notna() & ~is_organic].copy()
    unparseable = df[df["reduced_formula"].isna() & ~is_organic].copy()

    print(f"Querying COD for {len(searchable)} formulas Materials Project didn't have...")
    results = []
    for _, row in tqdm(searchable.iterrows(), total=len(searchable), desc="Querying COD"):
        target_comp = Composition(row["reduced_formula"])
        elements = sorted(str(e) for e in target_comp.elements)
        tqdm.write(f"{row['Formula']} ({row['Name']}) elements={elements}")
        docs = cod_search(elements)
        cand = best_candidate(target_comp, docs)
        time.sleep(0.3)
        if cand is None:
            results.append({
                **row.to_dict(), "cod_id": None, "cod_sg": None, "phase_name_verified": None, "cif_file": None
            })
            continue
        cod_id = cand["file"]
        fname = f"{sanitize_filename(row['Formula'])}.cif"
        # Unlike the MP pipeline's spacegroup-hint/remarks cross-checks, COD
        # matching here is element-set + stoichiometry-ratio only -- there's
        # no mechanism that confirms this candidate is the *specific* named
        # polymorph/phase Barin tabulates, so this is always False for a
        # match (never None, since a match with a CIF was in fact made).
        try:
            with urllib.request.urlopen(f"https://www.crystallography.net/cod/{cod_id}.cif", timeout=30) as resp:
                cif_text = resp.read().decode("utf-8", errors="replace")
            cif_text = f"# Source: Crystallography Open Database COD-{cod_id} (https://www.crystallography.net/cod/{cod_id}.html)\n" + cif_text
            (CIF_DIR / fname).write_text(cif_text)
            tqdm.write(f"  -> matched COD {cod_id} ({cand.get('sg')}), wrote {fname}")
            results.append({
                **row.to_dict(), "cod_id": cod_id, "cod_sg": cand.get("sg"),
                "phase_name_verified": False, "cif_file": fname,
            })
        except Exception as e:
            tqdm.write(f"  CIF download failed for {cod_id}: {e}")
            results.append({
                **row.to_dict(), "cod_id": cod_id, "cod_sg": cand.get("sg"),
                "phase_name_verified": False, "cif_file": None,
            })
        time.sleep(0.3)

    res_df = pd.DataFrame(results)
    res_df.to_csv(COD_SUMMARY_CSV, index=False)
    matched = res_df[res_df["cif_file"].notna()] if len(res_df) else res_df
    cod_unmatched = res_df[res_df["cif_file"].isna()] if len(res_df) else res_df

    still_unmatched = pd.concat([cod_unmatched, unparseable], ignore_index=True)
    still_unmatched.to_csv(STILL_UNMATCHED_CSV, index=False)

    print(f"\nDone. COD matched {len(matched)} / {len(searchable)}")
    print(f"  COD summary: {COD_SUMMARY_CSV}")
    print(f"  No structure expected (organic liquids, informational): {NO_STRUCTURE_CSV}")
    print(f"  Still unmatched (genuine ICSD candidates): {STILL_UNMATCHED_CSV}")


if __name__ == "__main__":
    main()
