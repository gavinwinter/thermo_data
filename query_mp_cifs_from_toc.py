"""
First-pass retrieval of CIF crystal structures for phases listed in
toc_barin.csv (produced by read_all_thermodata_pdf.py), using the Materials
Project API (mp-api).

Setup: this is the one thing you need to provide -- a free Materials
Project API key (https://next.materialsproject.org/api), exported as an
environment variable:

    export MP_API_KEY=your_key_here

Run with a Python environment that has mp-api + pandas installed (see
README):

    python query_mp_cifs_from_toc.py

Nothing else needs to change: paths are resolved relative to this script's
own location, so the repo can live anywhere.

Pipeline:
  0. Correct systematic OCR errors in the Formula column. Letter<->letter
     confusions (lowercase 'l' misread as capital 'I', 'Al' misread as
     'AI', 'Cl' misread as 'CI') are always safe to fix outright -- see
     fix_ocr_letters(). Letter<->digit confusions (typically an element
     letter like 'O' or 'S' misread as a digit, silently dropping that
     element) are repaired by cross-checking against the Name column,
     which almost always spells out the compound's elements -- see
     fix_missing_element_from_name() and repair_single_element_collapse().
     A short OCR_OVERRIDES table covers the residual one-off artifacts
     that don't fit any repeatable pattern.
  1. Parse toc_barin.csv -> drop gas-phase [g] entries, strip polymorph tags
     (e.g. Al2O3[C]) and expand hydrate notation (e.g. AlCl3*6H2O) into a
     pymatgen Composition.
  2. Batch-query Materials Project for every unique reduced formula.
  3. For each toc row, select a candidate structure:
       - single candidate -> use it
       - polymorph tag matches a known spacegroup hint -> use that match
       - otherwise -> lowest energy-above-hull (most stable) candidate,
         flagged as a "best-guess" pick for manual review
     Rows whose parsed composition collapses to a single element despite a
     long/complex formula string (a strong signature of leftover OCR
     corruption) are pulled out and sent to the unmatched list instead of
     being auto-matched.
  4. Fetch structures for the selected material_ids and write one CIF per
     toc row into cifs_barin/, plus a match summary CSV and an unmatched
     CSV (picked up next by query_cod_cifs_from_toc.py).
"""
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from pymatgen.core.composition import Composition
from mp_api.client import MPRester
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TOC_CSV = ROOT / "toc_barin.csv"
CIF_DIR = ROOT / "cifs_barin"
SUMMARY_CSV = ROOT / "cif_matches_mp.csv"
UNMATCHED_CSV = ROOT / "cif_unmatched_mp.csv"

CIF_DIR.mkdir(exist_ok=True)

# Known amorphous / non-crystalline entries: no CIF applies, don't bother querying.
AMORPHOUS_TAGS = {"GL"}  # glass

# Barin's toc was produced via OCR and has systematic character-confusion
# errors. Most of them are caught and repaired automatically below by
# cross-checking against the Name column, which almost always spells out
# the compound's elements (directly, via a compound-class suffix like
# "...OXIDE"/"...SULFATE", or an explicit count like "4-CALCIUM
# 3-TITANIUM 10-OXIDE"):
#   - a digit run hiding a dropped element letter (e.g. '209' -> '2O9')
#     -> fix_missing_element_from_name()
#   - a formula that fails to parse at all because a leading digit is a
#     misread element-starting capital letter -> fix_unparseable_leading_element()
#   - a real-but-wrong element where Name wants a different, same-length
#     one (e.g. 'Pr' misread for 'Pt') -> fix_wrong_element_swap()
#   - two adjacent single-letter elements that should merge into one
#     2-letter element (e.g. 'P'+'I' misread for 'Pt') -> fix_adjacent_orphan_merge()
#   - a single-letter element missing its second letter (e.g. 'S' for
#     'Sc') -> fix_orphan_element_extension()
#   - a formula collapsing to one element despite a long string, fixed by
#     trying common trailing-character confusions -> repair_single_element_collapse()
#
# What's left here is the residual handful of genuinely irreducible cases:
# a letter dropped from the *middle* of a digit run too long to safely
# guess without exact stoichiometry (Cr2306), a count given only as a
# Greek-numeral prefix rather than a named element (LiSAIF6's "TRI-"), a
# missing element masked because it's already present via another part of
# a multi-part formula (4PbO*PbS04), and formulas whose Name is a mineral
# or organic name that doesn't spell out its own chemistry at all
# (LiAISi206, C8Hi6, Ca5Si6O17*10.5w). Each was individually verified
# against the Name column. Keyed by the exact raw Formula string as it
# appears in toc_barin.csv.
OCR_OVERRIDES = {
    "Cr2306": "Cr23C6",              # 23-CHROMIUM 6-CARBIDE (mid-string 'C' dropped as '0')
    "LiAISi206": "LiAlSi2O6",        # ALPHA-SPODUMENE (mineral name doesn't spell out O)
    "4PbO*PbS04": "4PbO*PbSO4",      # PENTALEAD TETRAOXIDE SULFATE (O already present via the
                                      # '4PbO' part masks the missing-element check on 'PbS04')
    "Ca5Si6O17*10.5w": "Ca5Si6O17*10.5H2O",  # best-guess: truncated hydrate suffix
    "C8Hi6": "C8H16",                # ETHYLCYCLOHEXANE ('1' misread as 'i')
    "LiSAIF6": "Li3AlF6",            # TRILITHIUM HEXAFLUOROALUMINATE ('3' misread as 'S',
                                      # a Greek-numeral-prefix count Name doesn't literally spell
                                      # out as an element -- see module docstring)
}

# Full element symbol table, used to read the element(s) a Name column cell
# actually claims to contain -- e.g. "COBALT SELENITE" implies {Co, Se}.
ELEMENT_NAMES = {
    "HYDROGEN": "H", "HELIUM": "He", "LITHIUM": "Li", "BERYLLIUM": "Be", "BORON": "B",
    "CARBON": "C", "NITROGEN": "N", "OXYGEN": "O", "FLUORINE": "F", "NEON": "Ne",
    "SODIUM": "Na", "MAGNESIUM": "Mg", "ALUMINIUM": "Al", "ALUMINUM": "Al", "SILICON": "Si",
    "PHOSPHORUS": "P", "SULFUR": "S", "SULPHUR": "S", "CHLORINE": "Cl", "ARGON": "Ar",
    "POTASSIUM": "K", "CALCIUM": "Ca", "SCANDIUM": "Sc", "TITANIUM": "Ti", "VANADIUM": "V",
    "CHROMIUM": "Cr", "MANGANESE": "Mn", "IRON": "Fe", "COBALT": "Co", "NICKEL": "Ni",
    "COPPER": "Cu", "ZINC": "Zn", "GALLIUM": "Ga", "GERMANIUM": "Ge", "ARSENIC": "As",
    "SELENIUM": "Se", "BROMINE": "Br", "KRYPTON": "Kr", "RUBIDIUM": "Rb", "STRONTIUM": "Sr",
    "YTTRIUM": "Y", "ZIRCONIUM": "Zr", "NIOBIUM": "Nb", "MOLYBDENUM": "Mo", "TECHNETIUM": "Tc",
    "RUTHENIUM": "Ru", "RHODIUM": "Rh", "PALLADIUM": "Pd", "SILVER": "Ag", "CADMIUM": "Cd",
    "INDIUM": "In", "TIN": "Sn", "ANTIMONY": "Sb", "TELLURIUM": "Te", "IODINE": "I",
    "XENON": "Xe", "CESIUM": "Cs", "CAESIUM": "Cs", "BARIUM": "Ba", "LANTHANUM": "La",
    "CERIUM": "Ce", "PRASEODYMIUM": "Pr", "NEODYMIUM": "Nd", "SAMARIUM": "Sm",
    "EUROPIUM": "Eu", "GADOLINIUM": "Gd", "TERBIUM": "Tb", "DYSPROSIUM": "Dy",
    "HOLMIUM": "Ho", "ERBIUM": "Er", "THULIUM": "Tm", "YTTERBIUM": "Yb", "LUTETIUM": "Lu",
    "HAFNIUM": "Hf", "TANTALUM": "Ta", "TUNGSTEN": "W", "RHENIUM": "Re", "OSMIUM": "Os",
    "IRIDIUM": "Ir", "PLATINUM": "Pt", "GOLD": "Au", "MERCURY": "Hg", "THALLIUM": "Tl",
    "LEAD": "Pb", "BISMUTH": "Bi", "POLONIUM": "Po", "RADON": "Rn", "FRANCIUM": "Fr",
    "RADIUM": "Ra", "ACTINIUM": "Ac", "THORIUM": "Th", "PROTACTINIUM": "Pa", "URANIUM": "U",
    "NEPTUNIUM": "Np", "PLUTONIUM": "Pu", "AMERICIUM": "Am", "CURIUM": "Cm",
}
# Longest names first so e.g. "CAESIUM" isn't shadowed by a shorter false match.
_ELEMENT_NAMES_BY_LENGTH = sorted(ELEMENT_NAMES, key=len, reverse=True)

# Compound-class suffixes that imply a specific non-metal element is present,
# independent of the metal named elsewhere in the Name -- e.g. "...SULFATE"
# implies {S, O} regardless of what cation it's attached to. Suffixes ending
# in "ATE"/"ITE" normally imply O too, EXCEPT halide complexes like
# "HEXAFLUOROALUMINATE" (Al + F, no O at all) -- detected by the halide
# prefix immediately preceding the suffix, see implied_elements_from_name().
_ANION_SUFFIXES = {
    "OXIDE": {"O"}, "HYDROXIDE": {"O", "H"}, "PEROXIDE": {"O"},
    "CARBONATE": {"C", "O"}, "CARBIDE": {"C"}, "CYANIDE": {"C", "N"},
    "SULFATE": {"S", "O"}, "SULFITE": {"S", "O"}, "SULFIDE": {"S"},
    "NITRATE": {"N", "O"}, "NITRITE": {"N", "O"}, "NITRIDE": {"N"},
    "PHOSPHATE": {"P", "O"}, "PHOSPHITE": {"P", "O"}, "PHOSPHIDE": {"P"},
    "CHLORATE": {"Cl", "O"}, "PERCHLORATE": {"Cl", "O"}, "CHLORIDE": {"Cl"}, "CHLORITE": {"Cl", "O"},
    "BROMATE": {"Br", "O"}, "BROMIDE": {"Br"},
    "IODATE": {"I", "O"}, "IODIDE": {"I"},
    "FLUORIDE": {"F"},
    "SELENATE": {"Se", "O"}, "SELENITE": {"Se", "O"}, "SELENIDE": {"Se"},
    "TELLURATE": {"Te", "O"}, "TELLURIDE": {"Te"},
    "ARSENATE": {"As", "O"}, "ARSENIDE": {"As"},
    "ANTIMONATE": {"Sb", "O"}, "ANTIMONIDE": {"Sb"},
    "BORATE": {"B", "O"}, "BORIDE": {"B"},
    "SILICATE": {"Si", "O"}, "SILICIDE": {"Si"},
    "CHROMATE": {"Cr", "O"}, "VANADATE": {"V", "O"}, "MOLYBDATE": {"Mo", "O"},
    "TUNGSTATE": {"W", "O"}, "TITANATE": {"Ti", "O"}, "ALUMINATE": {"Al", "O"},
    "STANNATE": {"Sn", "O"}, "PLUMBATE": {"Pb", "O"}, "FERRATE": {"Fe", "O"},
    "MANGANATE": {"Mn", "O"}, "ZIRCONATE": {"Zr", "O"}, "NIOBATE": {"Nb", "O"},
    "URANATE": {"U", "O"},
}
# Halide-complex prefixes right before an "-ATE" suffix mean it's a
# fluoro-/chloro-/bromo-/iodo-metalate complex (e.g. cryolite's
# "HEXAFLUOROALUMINATE" = Al + F, no oxygen at all) -- suppress the
# suffix's usual oxygen implication in that case.
_HALIDE_ATE_PREFIXES = ("FLUORO", "CHLORO", "BROMO", "IODO")


def implied_elements_from_name(name: str) -> set:
    """Best-effort set of element symbols a Name cell claims the compound
    contains, read straight off the chemical nomenclature -- e.g. "COBALT
    SELENITE" -> {Co, Se, O}, "TRILITHIUM HEXAFLUOROALUMINATE" -> {Li, Al, F}
    (no O, because it's a fluoro-complex, not an oxyanion)."""
    upper = str(name).upper()
    words = re.findall(r"[A-Z]+", upper)
    implied = set()

    for ename in _ELEMENT_NAMES_BY_LENGTH:
        if re.search(r"\b" + ename, upper):
            implied.add(ELEMENT_NAMES[ename])

    for word in words:
        for suffix, elements in _ANION_SUFFIXES.items():
            if word.endswith(suffix):
                elements = set(elements)
                if "O" in elements and suffix.endswith(("ATE", "ITE")):
                    prefix = word[: -len(suffix)]
                    if prefix.endswith(_HALIDE_ATE_PREFIXES):
                        elements.discard("O")
                implied |= elements
                break
    return implied


def _digit_run_fix_positions(formula: str):
    """Positions where a dropped element letter could plausibly be hiding
    inside a run of consecutive digits -- e.g. Barin's 'Al4B209' merges the
    missing 'O' between boron's count '2' and oxygen's count '9' into a
    single run '209'. Only two run lengths are structurally unambiguous
    enough to guess blindly:
      - a 2-digit run: the letter must be the *first* digit (the only split
        that leaves both the preceding and the new element with a
        non-empty count) -- 'B04' -> 'B' + 'O' + '4'.
      - a 3-digit run: the letter must be the *middle* digit, preserving a
        count on both sides -- '209' -> '2' + 'O' + '9'.
    A lone digit is left alone (no room to split without leaving some
    element with an empty count, which is too ambiguous to guess), and so
    is a run of 4+ digits (no longer a single unambiguous split point).
    """
    positions = []
    i = 0
    while i < len(formula):
        if formula[i].isdigit():
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            run_len = j - i
            if run_len == 2:
                positions.append(i)
            elif run_len == 3:
                positions.append(i + 1)
            i = j
        else:
            i += 1
    return positions


def fix_missing_element_from_name(formula: str, name: str):
    """If the Name column implies an element that isn't in the parsed
    composition, and a digit run in the formula structurally matches where a
    dropped element letter would hide (see _digit_run_fix_positions), try
    substituting it in (most commonly '0' misread from 'O') and accept the
    first substitution that resolves *every* Name-implied element, not just
    the one being targeted -- requiring the whole set to check out is what
    keeps this from accepting a structurally-valid but chemically-wrong
    split. This generalizes the common Barin OCR error of an element letter
    being read as a digit, instead of hand-listing every affected formula.
    """
    comp = formula_to_composition(formula)
    if comp is None:
        return formula
    have = {str(e) for e in comp.elements}
    implied = implied_elements_from_name(name)
    missing = implied - have
    if not missing:
        return formula

    for el in sorted(missing):
        for i in _digit_run_fix_positions(formula):
            candidate = formula[:i] + el + formula[i + 1 :]
            cand_comp = formula_to_composition(candidate)
            if cand_comp is None:
                continue
            cand_have = {str(e) for e in cand_comp.elements}
            if implied <= cand_have:
                return candidate
    return formula


def _others_unchanged(orig_comp, cand_comp, excluded_elements) -> bool:
    """True if every element other than the ones this specific edit targets
    kept an identical count -- the safeguard that lets the rules below edit
    outside of a digit run without risking a structurally-valid but
    chemically-wrong result (e.g. accidentally changing an unrelated
    element's count)."""
    orig = orig_comp.get_el_amt_dict()
    cand = cand_comp.get_el_amt_dict()
    others = (set(orig) | set(cand)) - excluded_elements
    return all(abs(orig.get(el, 0) - cand.get(el, 0)) < 1e-6 for el in others)


def fix_unparseable_leading_element(formula: str, name: str):
    """A formula that fails to parse *at all* and starts with a digit
    immediately followed by a lowercase letter (e.g. '2rCl2') is invalid in
    a very specific way: a leading stoichiometric multiplier is only ever
    followed by an uppercase element-starting letter, never lowercase -- so
    that leading digit must itself be a misread capital letter, the first
    half of a two-letter element whose second letter is what follows it.
    Try every implied element matching that second letter."""
    if formula_to_composition(formula) is not None:
        return formula
    m = re.match(r"^(\d)([a-z])", formula)
    if not m:
        return formula
    second_letter = m.group(2)
    for el in sorted(implied_elements_from_name(name)):
        if len(el) == 2 and el[1] == second_letter:
            candidate = el + formula[2:]
            if formula_to_composition(candidate) is not None:
                return candidate
    return formula


def fix_wrong_element_swap(formula: str, name: str):
    """The formula parses fine, but one of its elements is a real element
    Name doesn't call for, and Name calls for a same-length element that's
    missing -- almost always a single misread letter inside a 2-letter
    symbol (e.g. platinum's 'Pt' read as the equally-valid element 'Pr').
    Swap the text and require every other element's count to be untouched.
    """
    comp = formula_to_composition(formula)
    if comp is None:
        return formula
    have = {str(e) for e in comp.elements}
    implied = implied_elements_from_name(name)
    missing = implied - have
    wrong = have - implied
    for right_el in sorted(missing):
        for wrong_el in sorted(wrong):
            if len(wrong_el) != len(right_el) or wrong_el not in formula:
                continue
            candidate = formula.replace(wrong_el, right_el, 1)
            cand_comp = formula_to_composition(candidate)
            if cand_comp is None:
                continue
            cand_have = {str(e) for e in cand_comp.elements}
            if implied <= cand_have and _others_unchanged(comp, cand_comp, {wrong_el, right_el}):
                return candidate
    return formula


def fix_orphan_element_extension(formula: str, name: str):
    """The formula parses fine and contains a single-letter element Name
    doesn't call for, which is also the first letter of a 2-letter element
    Name does call for and that's missing -- almost always a dropped second
    letter (e.g. scandium 'Sc' read as just 'S'). Insert the missing letter
    right after the orphan and require every other element's count to be
    untouched."""
    comp = formula_to_composition(formula)
    if comp is None:
        return formula
    have = {str(e) for e in comp.elements}
    implied = implied_elements_from_name(name)
    missing = implied - have
    wrong = have - implied
    for right_el in sorted(missing):
        if len(right_el) != 2:
            continue
        for orphan in sorted(wrong):
            if len(orphan) != 1 or not right_el.startswith(orphan) or orphan not in formula:
                continue
            i = formula.index(orphan)
            candidate = formula[: i + 1] + right_el[1] + formula[i + 1 :]
            cand_comp = formula_to_composition(candidate)
            if cand_comp is None:
                continue
            cand_have = {str(e) for e in cand_comp.elements}
            if implied <= cand_have and _others_unchanged(comp, cand_comp, {orphan, right_el}):
                return candidate
    return formula


def fix_adjacent_orphan_merge(formula: str, name: str):
    """The formula parses fine as two adjacent single-letter elements that
    Name doesn't call for, which together spell out (as plain text) a
    2-letter element Name does call for and that's missing -- a letter
    inside a 2-letter symbol misread as a *different* valid single-letter
    element, splitting one element into two (e.g. platinum 'Pt' read as
    phosphorus 'P' + iodine 'I', because the 't' was misread as a capital
    'I'). Merge the two-character substring into the missing element and
    require every other element's count to be untouched."""
    comp = formula_to_composition(formula)
    if comp is None:
        return formula
    have = {str(e) for e in comp.elements}
    implied = implied_elements_from_name(name)
    missing = implied - have
    wrong = have - implied
    single_wrong = sorted(el for el in wrong if len(el) == 1)
    for right_el in sorted(missing):
        if len(right_el) != 2:
            continue
        for i in range(len(formula) - 1):
            pair = formula[i : i + 2]
            if pair[0] not in single_wrong or pair[1] not in single_wrong:
                continue
            candidate = formula[:i] + right_el + formula[i + 2 :]
            cand_comp = formula_to_composition(candidate)
            if cand_comp is None:
                continue
            cand_have = {str(e) for e in cand_comp.elements}
            if implied <= cand_have and _others_unchanged(comp, cand_comp, set(pair) | {right_el}):
                return candidate
    return formula


# When a formula collapses to a single element despite a long/complex
# string (see load_toc()'s suspicious-corruption check), the usual cause is
# the *last* character being a misread element letter rather than part of
# the stoichiometric count -- e.g. wuestite "Fe0.9470" (trailing '0' should
# be 'O') or pyrrhotite "Fe0.8778" (trailing '8' should be 'S', since '8'
# and 'S' are an easy OCR confusion). Try the common confusions in order and
# keep the first one that resolves the collapse into a real 2+ element
# composition.
_TRAILING_CHAR_CONFUSIONS = {"0": "O", "8": "S", "1": "I", "5": "S"}


def repair_single_element_collapse(formula: str):
    last = formula[-1]
    swap = _TRAILING_CHAR_CONFUSIONS.get(last)
    if not swap:
        return formula
    candidate = formula[:-1] + swap
    comp = formula_to_composition(candidate)
    if comp is not None and len(comp.elements) > 1:
        return candidate
    return formula

# A handful of rows have a Formula that was copy-duplicated from a nearby
# row (an OCR/transcription artifact), independently identifiable by page
# number and cross-checked against the Name column.
# Keyed by (page, original Name) since a page can hold multiple rows and a
# page-only key would clobber an unrelated row that happens to share a page.
PAGE_OVERRIDES = {
    (498, "Cdl2 CADMIUM IODIDE"): ("CdI2", "CADMIUM IODIDE"),  # leaked "Cdl2 " prefix
    (512, "DICERIUM TRICARBIDE"): ("Ce2C3", None),  # Formula duplicated from a nearby Ce2O3 row
}


def fix_ocr_letters(formula: str) -> str:
    """Apply the systematic, low-risk OCR letter-confusion fixes:
    lowercase 'l' misread as capital 'I' (only ambiguous right after the
    real digraphs Al/Cl/Tl, which are protected). 'AI' and 'CI' as literal
    substrings are always safe to fix within this (already gas-filtered)
    non-gas formula set: no real element is a bare 'A', and no non-gas Barin
    entry pairs carbon directly with iodine (that only occurs in the
    halomethane gas-phase entries, which are excluded before this runs).
    """
    formula = re.sub(r"(?<![ACT])l", "I", formula)
    formula = formula.replace("AI", "Al")
    formula = formula.replace("CI", "Cl")
    return formula

# Best-effort spacegroup hints for named polymorphs in the toc (from general
# mineralogical/crystallographic knowledge). Symbols use pymatgen's
# short Hermann-Mauguin form as returned by SymmetryData.symbol.
# Left deliberately partial -- anything not listed here falls back to the
# "most stable" pick and is flagged for manual review.
POLYMORPH_SPACEGROUP_HINTS = {
    ("Al2O3", "C"): "Fd-3m",       # gamma-Al2O3 (spinel-type)
    ("As2O3", "A"): "Fd-3m",       # arsenolite
    ("C", "D"): "Fd-3m",           # diamond
    ("CaCO3", "A"): "Pnma",        # aragonite
    ("SiO2", "CR"): "P4_12_12",    # low cristobalite
    ("TiO2", "A"): "I4_1/amd",     # anatase
    ("ZnS", "S"): "F-43m",         # sphalerite
    ("PbO", "R"): "P4/nmm",        # litharge (red PbO)
}

def strip_phase_tag(formula: str):
    """Return (bare_formula, tag) splitting off a trailing [TAG] annotation."""
    m = re.search(r"\[([^\]]*)\]$", formula)
    if not m:
        return formula, None
    return formula[: m.start()], m.group(1)


def parse_leading_coefficient(part: str) -> str:
    """pymatgen won't parse a bare leading multiplier like '6H2O' or '2PbO'
    (whole-formula multiplication needs parens), and not at all for a
    fractional one like '7/6H2O'. Rewrite '<coeff><rest>' as '(<rest>)<coeff>'
    so pymatgen applies the multiplier to the whole group.
    """
    m = re.match(r"^(\d+/\d+|\d+\.?\d*)([A-Za-z(].*)$", part)
    if not m:
        return part
    coeff, rest = m.groups()
    if "/" in coeff:
        num, den = coeff.split("/")
        coeff = str(float(num) / float(den))
    return f"({rest}){coeff}"


def fix_hydrate_water(formula: str) -> str:
    """Every '*'-joined adduct part in this dataset that consists of just H
    and O is water -- so a hydrate segment OCR'd as '...H20' (zero) instead
    of '...H2O' is unambiguous and safe to fix generally, unlike a bare '0'
    elsewhere in a formula (which could legitimately be part of a multi-digit
    count like '10')."""
    parts = formula.split("*")
    fixed = [parts[0]]
    for part in parts[1:]:
        fixed.append(re.sub(r"H20$", "H2O", part))
    return "*".join(fixed)


def formula_to_composition(formula: str):
    """Best-effort parse of a (bracket-stripped) Barin formula into a pymatgen
    Composition, handling '*' hydrate/adduct notation and leading multipliers
    on either side (e.g. '2PbO*PbSO4', 'AlCl3*6H2O'). Returns None on failure.
    """
    parts = formula.split("*")
    try:
        comp = Composition(parse_leading_coefficient(parts[0]))
        for part in parts[1:]:
            comp += Composition(parse_leading_coefficient(part))
        return comp
    except Exception:
        return None


def load_toc():
    toc = pd.read_csv(TOC_CSV)
    toc.columns = [c.strip() for c in toc.columns]
    rows = []
    for _, r in toc.iterrows():
        formula_raw = str(r["Formula"]).strip()
        name_raw = str(r["Name"]).strip()
        page = int(r["Page number"])
        bare, tag = strip_phase_tag(formula_raw)
        # OCR sometimes mangles the '[g]' gas marker itself (e.g. '(g]',
        # '{g]', '/g]'), so also fall back to the Name column, which
        # reliably says "(GAS)" for every gas-phase entry.
        if tag == "g" or "(GAS)" in name_raw.upper():
            continue  # gas phase, no crystal structure
        if (page, name_raw) in PAGE_OVERRIDES:
            # Formula field was copy-duplicated from a neighboring row.
            formula_raw, fixed_name = PAGE_OVERRIDES[(page, name_raw)]
            if fixed_name:
                name_raw = fixed_name
        corrected = OCR_OVERRIDES.get(formula_raw, formula_raw)
        corrected = fix_ocr_letters(corrected)
        corrected = fix_hydrate_water(corrected)
        bare, tag = strip_phase_tag(corrected)  # tag may itself have been letter-fixed
        bare = fix_unparseable_leading_element(bare, name_raw)
        bare = fix_missing_element_from_name(bare, name_raw)
        bare = fix_wrong_element_swap(bare, name_raw)
        # Merge before extension: a dropped letter that landed on an
        # *adjacent* already-valid element (merge's pattern) must be tried
        # before assuming a single orphan just needs a letter appended
        # (extension's pattern), since both can superficially match the
        # same "target starts with this orphan" condition.
        bare = fix_adjacent_orphan_merge(bare, name_raw)
        bare = fix_orphan_element_extension(bare, name_raw)
        comp = formula_to_composition(bare)
        n_elements = len(comp.elements) if comp is not None else 0
        suspicious = comp is not None and n_elements <= 1 and len(bare) > 4
        if suspicious:
            bare = repair_single_element_collapse(bare)
            comp = formula_to_composition(bare)
            n_elements = len(comp.elements) if comp is not None else 0
            suspicious = comp is not None and n_elements <= 1 and len(bare) > 4
        reduced = comp.reduced_formula if comp is not None else None
        rows.append(
            {
                "Formula": formula_raw,
                "Name": name_raw,
                "Page number": r["Page number"],
                "bare_formula": bare,
                "phase_tag": tag,
                "reduced_formula": reduced,
                "parse_ok": comp is not None and not suspicious,
                "suspected_ocr_corruption": suspicious,
            }
        )
    return pd.DataFrame(rows)


def get_api_key():
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit(
            "MP_API_KEY is not set. Get a free key from "
            "https://next.materialsproject.org/api and run:\n"
            "    export MP_API_KEY=your_key_here"
        )
    return key


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def query_candidates(mpr, reduced_formulas):
    """Batch-query MP summary docs for a list of reduced formulas.
    Returns a DataFrame of candidates (no structure yet)."""
    fields = ["material_id", "formula_pretty", "symmetry", "energy_above_hull", "is_stable", "nsites"]
    all_docs = []
    formulas = sorted(set(reduced_formulas))
    batches = list(chunked(formulas, 100))
    for batch in tqdm(batches, desc="Querying MP by formula batch"):
        for attempt in range(3):
            try:
                docs = mpr.materials.summary.search(formula=batch, fields=fields)
                break
            except Exception as e:
                tqdm.write(f"  query error ({e}), retrying...")
                time.sleep(5)
        else:
            tqdm.write(f"  FAILED batch starting with {batch[0]}, skipping")
            continue
        for d in docs:
            all_docs.append(
                {
                    "reduced_formula": d.formula_pretty,
                    "material_id": str(d.material_id),
                    "spacegroup": d.symmetry.symbol if d.symmetry else None,
                    "energy_above_hull": d.energy_above_hull,
                    "is_stable": d.is_stable,
                    "nsites": d.nsites,
                }
            )
        tqdm.write(f"  queried {len(batch)} formulas, total candidates so far: {len(all_docs)}")
    return pd.DataFrame(all_docs)


def select_candidate(row, candidates):
    """Pick a material_id for one toc row from its candidate pool."""
    cands = candidates[candidates["reduced_formula"] == row["reduced_formula"]]
    if len(cands) == 0:
        return None, "no_mp_match", None
    if len(cands) == 1:
        c = cands.iloc[0]
        return c["material_id"], "single_candidate", c["spacegroup"]

    if row["phase_tag"]:
        hint_sg = POLYMORPH_SPACEGROUP_HINTS.get((row["bare_formula"], row["phase_tag"]))
        if hint_sg:
            match = cands[cands["spacegroup"] == hint_sg]
            if len(match) >= 1:
                c = match.sort_values("energy_above_hull").iloc[0]
                return c["material_id"], "spacegroup_hint", c["spacegroup"]

    c = cands.sort_values("energy_above_hull").iloc[0]
    method = "most_stable_guess" if not row["phase_tag"] else "most_stable_guess_polymorph_unresolved"
    return c["material_id"], method, c["spacegroup"]


def sanitize_filename(formula: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\[\]-]", "_", formula.replace("*", "_hyd_"))


def main():
    toc = load_toc()
    print(f"Non-gas toc rows: {len(toc)}")
    print(f"Rows with amorphous tag (skipped, no CIF possible): "
          f"{(toc['phase_tag'].isin(AMORPHOUS_TAGS)).sum()}")

    parseable = toc[toc["parse_ok"] & ~toc["phase_tag"].isin(AMORPHOUS_TAGS)].copy()
    unparseable = toc[~toc["parse_ok"]].copy()
    print(f"Parseable formulas: {len(parseable)}, unparseable: {len(unparseable)}")

    api_key = get_api_key()
    with MPRester(api_key) as mpr:
        print("Querying Materials Project for candidates...")
        candidates = query_candidates(mpr, parseable["reduced_formula"].dropna().tolist())
        candidates.to_csv(ROOT / "mp_candidates_raw.csv", index=False)

        results = []
        for _, row in parseable.iterrows():
            mid, method, sg = select_candidate(row, candidates)
            results.append({**row.to_dict(), "material_id": mid, "match_method": method, "spacegroup": sg})
        results_df = pd.DataFrame(results)

        matched = results_df[results_df["material_id"].notna()].copy()
        print(f"Matched {len(matched)} / {len(parseable)} parseable rows to an MP material_id")

        print(f"Fetching structures for {len(matched['material_id'].unique())} selected material_ids...")
        mid_list = sorted(matched["material_id"].unique())
        structures = {}
        mid_batches = list(chunked(mid_list, 50))
        for batch in tqdm(mid_batches, desc="Fetching structures"):
            for attempt in range(3):
                try:
                    docs = mpr.materials.summary.search(
                        material_ids=batch, fields=["material_id", "structure"]
                    )
                    break
                except Exception as e:
                    tqdm.write(f"  structure fetch error ({e}), retrying...")
                    time.sleep(5)
            else:
                tqdm.write(f"  FAILED structure batch starting with {batch[0]}, skipping")
                continue
            for d in docs:
                structures[str(d.material_id)] = d.structure

        cif_paths = []
        for _, row in tqdm(matched.iterrows(), total=len(matched), desc="Writing CIF files"):
            mid = row["material_id"]
            struct = structures.get(mid)
            if struct is None:
                cif_paths.append(None)
                continue
            fname = f"{sanitize_filename(row['Formula'])}.cif"
            struct.to(filename=str(CIF_DIR / fname), fmt="cif")
            cif_paths.append(fname)
        matched["cif_file"] = cif_paths

    # Assemble final outputs
    def unparseable_label(row):
        return "suspected_ocr_corruption" if row["suspected_ocr_corruption"] else "unparseable_formula"

    unparseable = unparseable.assign(
        material_id=None,
        match_method=unparseable.apply(unparseable_label, axis=1),
        spacegroup=None,
    )
    unmatched = pd.concat(
        [results_df[results_df["material_id"].isna()], unparseable],
        ignore_index=True,
    )

    cols = ["Formula", "Name", "Page number", "bare_formula", "phase_tag",
            "reduced_formula", "material_id", "match_method", "spacegroup", "cif_file"]
    matched.reindex(columns=cols).to_csv(SUMMARY_CSV, index=False)
    unmatched.reindex(columns=[c for c in cols if c != "cif_file"]).to_csv(UNMATCHED_CSV, index=False)

    print(f"\nDone.")
    print(f"  CIFs written: {matched['cif_file'].notna().sum()} -> {CIF_DIR}")
    print(f"  Match summary: {SUMMARY_CSV}")
    print(f"  Unmatched (need ICSD / manual lookup): {len(unmatched)} -> {UNMATCHED_CSV}")
    print(matched["match_method"].value_counts())


if __name__ == "__main__":
    main()
