"""
First-pass retrieval of CIF crystal structures for phases listed in
toc_barin.csv (produced by read_all_thermodata_pdf.py), using the
Materials Project API (mp-api).

Usage: export MP_API_KEY=your_key_here (https://next.materialsproject.org/api),
then `python query_mp_cifs_from_toc.py`. Paths resolve relative to this
script's own location, so the repo can live anywhere.

Pipeline:
  0. Correct systematic OCR errors in the Formula column -- safe
     letter<->letter fixes (fix_ocr_letters), letter<->digit fixes
     cross-checked against the Name column (fix_missing_element_from_name,
     repair_single_element_collapse), and a residual OCR_OVERRIDES table.
  1. Parse toc_barin.csv -> drop [g] entries, strip polymorph tags, expand
     hydrate notation into a pymatgen Composition.
  2. Batch-query Materials Project for every unique reduced formula.
  3. Select a candidate per row: single candidate, or a polymorph-tag
     spacegroup match, or lowest energy-above-hull as a best-guess pick.
     Rows whose composition collapses to a single element despite a
     complex formula (a sign of leftover OCR corruption) go to unmatched.
  4. Fetch structures and write one CIF per row into cifs_barin/, plus a
     match summary CSV and an unmatched CSV for query_cod_cifs_from_toc.py.
"""
import os
import re
import sys
import time
from fractions import Fraction
from functools import reduce
from math import gcd
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

# toc_barin.csv has already been cleaned of systematic OCR
# character-confusion errors at the source; what follows guards against
# whatever a *different* OCR pass might produce, by cross-checking each
# formula against its Name column (which almost always spells out the
# compound's elements, directly or via a suffix like "...SULFATE"):
#   - digit run hiding a dropped letter ('209'->'2O9') -> fix_missing_element_from_name()
#   - unparseable leading digit (misread element letter) -> fix_unparseable_leading_element()
#   - wrong same-length element ('Pr' for 'Pt') -> fix_wrong_element_swap()
#   - two letters that should merge into one ('P'+'I' for 'Pt') -> fix_adjacent_orphan_merge()
#   - single letter missing its second letter ('S' for 'Sc') -> fix_orphan_element_extension()
#   - collapses to one element despite a long string -> repair_single_element_collapse()

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
    """Positions where a dropped element letter could plausibly hide inside
    a run of consecutive digits (e.g. Barin's 'Al4B209' merges the missing
    'O' between '2' and '9' into '209'). Only two run lengths are
    unambiguous enough to guess: a 2-digit run splits at the *first* digit
    ('B04' -> 'B'+'O'+'4'), a 3-digit run at the *middle* digit ('209' ->
    '2'+'O'+'9'). A lone digit or a 4+ digit run is left alone -- too
    ambiguous to split.
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
    """If the Name column implies an element missing from the parsed
    composition, and a digit run structurally matches where a dropped
    element letter would hide (see _digit_run_fix_positions), substitute
    it in (usually '0' misread from 'O') and accept the first fix that
    resolves *every* Name-implied element, not just the one targeted --
    this generalizes Barin's letter-read-as-digit OCR error instead of
    hand-listing every affected formula.
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
    """The formula parses as two adjacent single-letter elements Name
    doesn't call for, which together spell a missing 2-letter element Name
    does call for -- a letter inside a 2-letter symbol misread as a
    different valid single-letter element (e.g. 'Pt' read as 'P'+'I', the
    't' misread as capital 'I'). Merges the pair into the missing element,
    requiring every other element's count to stay untouched."""
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
# string (see load_toc()'s check), the usual cause is the *last* character
# being a misread element letter, not a stoichiometric count (e.g.
# "Fe0.9470" -> trailing '0' should be 'O'; "Fe0.8778" -> trailing '8'
# should be 'S'). Try common confusions in order, keep the first that
# resolves the collapse into a real 2+ element composition.
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
    ("Al2O3", "C"): "Fd-3m",       # gamma-Al2O3 (spinel-type) -- no current MP
                                    # candidate actually has this symmetry, so
                                    # this hint is presently a no-op; left in
                                    # case MP adds one later
    ("As2O3", "A"): "Fd-3m",       # arsenolite
    ("C", "D"): "Fd-3m",           # diamond
    ("CaCO3", "A"): "Pnma",        # aragonite
    ("SiO2", "CR"): "P4_12_12",    # low cristobalite
    ("TiO2", "A"): "I4_1/amd",     # anatase
    ("ZnS", "S"): "F-43m",         # sphalerite
    ("PbO", "R"): "P4/nmm",        # litharge (red PbO)
    ("Al2SiO5", "A"): "Pnnm",      # andalusite
    ("Al2SiO5", "S"): "Pnma",      # sillimanite (Pbnm in some settings)
    ("Ca2SiO4", "B"): "P2_1/c",    # larnite (beta-Ca2SiO4)
    ("Sb2O3", "O"): "Pccn",        # valentinite (orthorhombic Sb2O3)
    ("Eu2O3", "M"): "C2/m",        # B-type monoclinic Eu2O3
    ("Gd2O3", "M"): "C2/m",        # B-type monoclinic Gd2O3
    ("Sm2O3", "M"): "C2/m",        # B-type monoclinic Sm2O3
    ("Al2O3*H2O", "B"): "Pmn2_1",  # boehmite -- the only other AlOOH
                                    # candidate besides diaspore (Pnma, the
                                    # untagged row's pick)
}

# Some rows carry their polymorph identity only as a Name parenthetical
# (e.g. '(WHITE)', '(CUBIC)') rather than a [tag] -- meaning "most stable
# by DFT energy" isn't actually what that row represents. Verified
# against MP: for each, the lowest-energy_above_hull candidate is a
# different, well-known named polymorph than the one Name specifies.
UNTAGGED_NAME_SPACEGROUP_HINTS = {
    "Sn": "I4_1/amd",       # white/beta-Sn (body-centered tetragonal) -- the
                             # lowest-energy candidate is gray/alpha-Sn (Fd-3m,
                             # diamond-cubic), a different, colder-stable form
    "Si3N4": "P31c",        # alpha-Si3N4 -- lowest-energy candidate is the
                             # higher-symmetry beta form (P6_3/m)
    "Sb2O3": "Fd-3m",       # senarmontite (cubic) -- lowest-energy candidate
                             # is valentinite (Pccn, orthorhombic), which is
                             # the *other* named Sb2O3 polymorph in this toc
                             # (already correctly hinted via its own [O] tag)
    "Pu2O3": "P-3m1",       # alpha-Pu2O3 (A-type hexagonal, by analogy with
                             # the Ln2O3 A/B/C sesquioxide pattern) -- lowest-
                             # energy candidate is the C-type cubic form (Ia-3)
    "SiC": "F-43m",         # 3C-SiC (cubic zinc-blende) -- SiC has dozens of
                             # near-degenerate hexagonal/rhombohedral polytype
                             # candidates within noise of the DFT ground state,
                             # so "lowest energy" is close to an arbitrary pick
                             # among them; this pins it to the specifically
                             # named cubic polytype instead
    "TiO2": "P4_2/mnm",     # rutile (icsd_n=131, overwhelmingly the common
                             # form) -- lowest-energy candidate is anatase
                             # (I4_1/amd), the *other* named TiO2 polymorph in
                             # this toc (already correctly hinted via [A])
    "PbO": "Pbcm",          # massicot/yellow PbO -- lowest-energy candidate
                             # is litharge/red PbO (P4/nmm), the *other* named
                             # PbO polymorph in this toc (already correctly
                             # hinted via [R])
    "LiAlSi2O6": "C2/c",    # alpha-spodumene (pyroxene structure, icsd_n=17)
                             # -- lowest-energy candidate is a weakly-supported
                             # P1 structure (icsd_n=1), not a named polymorph
    "Fe0.778S": "C2/c", 
}

# Elements in this dataset's actual organic entries (hydrocarbons,
# alcohols, acids): C/H plus the O/S/N/halogens in their functional
# groups. A formula needs C and H *and* nothing outside this set to count
# as organic -- otherwise NaHCO3 (a genuine mineral with Na) would get
# misclassified just for containing both C and H.
_ORGANIC_ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def is_organic_formula(bare_formula: str) -> bool:
    """Contains both C and H, with nothing outside CHNOPS+halogens (i.e.
    not an inorganic salt that happens to contain both). Distinct organic
    isomers routinely share an empirical formula (e.g. cyclohexane and
    methylcyclopentane are both C6H12), unlike most inorganic polymorphs --
    so multiple MP candidates here can't be disambiguated by stoichiometry
    alone the way select_candidate()'s inorganic fallback tiers do.
    """
    comp = formula_to_composition(bare_formula)
    if comp is None:
        return False
    elements = {str(e) for e in comp.elements}
    return "C" in elements and "H" in elements and elements <= _ORGANIC_ALLOWED_ELEMENTS


def is_whole_molecule_multiple(bare_formula: str, candidate_composition: str) -> bool:
    """MP's formula= search matches by reduced formula, collapsing every
    CnH2n cycloalkane/alkene onto the same bucket (C6H12, C6H12[M],
    C7H14[M] are all indistinguishable to it). A candidate can only
    actually BE a multiple of the target if its exact per-cell atom counts
    are an integer multiple of the target's own formula (a C4H8 cell can't
    be built from whole C6H12 molecules) -- catches this even with only
    one MP candidate, so select_candidate()'s single_candidate tier can't
    be trusted blindly for organics either.
    """
    target = formula_to_composition(bare_formula)
    if target is None:
        return True  # can't check -- don't block on a formula we can't parse
    try:
        cand = Composition(candidate_composition)
    except Exception:
        return True
    target_amts = target.get_el_amt_dict()
    cand_amts = cand.get_el_amt_dict()
    if set(target_amts) != set(cand_amts):
        return False
    ratios = [cand_amts[el] / target_amts[el] for el in target_amts]
    k = round(ratios[0])
    if k < 1:
        return False
    return all(abs(r - k) < 1e-2 for r in ratios)


def strip_phase_tag(formula: str):
    """Return (bare_formula, tag) splitting off a trailing [TAG] annotation."""
    m = re.search(r"\[([^\]]*)\]$", formula)
    if not m:
        return formula, None
    return formula[: m.start()], m.group(1)


def strip_all_tags(formula: str):
    """Repeatedly strip trailing [TAG] groups -- some entries stack more than
    one, e.g. 'C2F2Cl2[1,1][g]' (isomer tag, then gas tag)."""
    cur = formula
    while True:
        bare, tag = strip_phase_tag(cur)
        if tag is None:
            return bare
        cur = bare


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


def rationalize_composition(comp: Composition, max_denominator: int = 20) -> Composition:
    """Approximate a composition with non-integer element amounts by the
    nearest whole-number-ratio composition.

    Barin includes non-stoichiometric defect compounds at an exact
    fractional ratio (e.g. Fe0.877S for pyrrhotite). MP indexes only
    whole-number formulas, and the real MP entry is a specific
    small-integer formula (e.g. Fe7S8), not the literal decimal -- so this
    finds that integer formula for search purposes; composition_ratio_close()
    below re-verifies the result against the true ratio with tolerance,
    since this is only an approximation.
    """
    amt_dict = comp.get_el_amt_dict()
    fracs = {el: Fraction(amt).limit_denominator(max_denominator) for el, amt in amt_dict.items()}
    denom_lcm = reduce(lambda a, b: a * b // gcd(a, b), (f.denominator for f in fracs.values()), 1)
    int_amts = {el: round(f * denom_lcm) for el, f in fracs.items()}
    common = reduce(gcd, [v for v in int_amts.values() if v > 0])
    int_amts = {el: amt // common for el, amt in int_amts.items()}
    return Composition(int_amts)


def is_nonstoichiometric(comp: Composition) -> bool:
    """True if any element amount is not a whole number."""
    return any(abs(amt - round(amt)) > 1e-6 for amt in comp.get_el_amt_dict().values())


def composition_ratio_close(target_bare_formula: str, candidate_composition: str, tol: float = 0.05) -> bool:
    """For a non-stoichiometric target, verify a candidate's element ratios
    are close to the *true* measured ratio (not the rationalized search
    formula, which is only an approximant) -- e.g. Fe0.877S's real ratio is
    0.877, and Fe7S8 (0.875) is within tolerance, but a candidate near a
    different simple ratio like Fe0.75S (3:4) should be rejected even though
    it might share some intermediate rationalization at a coarser
    max_denominator.
    """
    target = formula_to_composition(target_bare_formula)
    if target is None:
        return True
    try:
        cand = Composition(candidate_composition)
    except Exception:
        return True
    target_amts = target.get_el_amt_dict()
    cand_amts = cand.get_el_amt_dict()
    if set(target_amts) != set(cand_amts):
        return False
    anchor = next(iter(target_amts))
    if target_amts[anchor] == 0:
        return True
    target_ratios = {el: amt / target_amts[anchor] for el, amt in target_amts.items()}
    cand_ratios = {el: amt / cand_amts[anchor] for el, amt in cand_amts.items()}
    return all(abs(cand_ratios[el] - target_ratios[el]) <= tol for el in target_ratios)


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
        bare, tag = strip_phase_tag(formula_raw)
        # OCR sometimes mangles the '[g]' gas marker itself (e.g. '(g]',
        # '{g]', '/g]'), so also fall back to the Name column, which
        # reliably says "(GAS)" for every gas-phase entry.
        if tag == "g" or "(GAS)" in name_raw.upper():
            # A gas has no crystal structure to look up, so it's never a
            # matching candidate -- but if it's organic, that's the
            # clearest "no structure expected" case, worth recording rather
            # than vanishing silently. (Non-organic gases are left out
            # entirely: Barin's gas tables are exhaustive elemental/simple
            # entries, and recording all ~1000 isn't what's being asked here.)
            fully_bare = strip_all_tags(fix_ocr_letters(formula_raw))
            if is_organic_formula(fully_bare):
                rows.append(
                    {
                        "Formula": formula_raw,
                        "Name": name_raw,
                        "Page number": r["Page number"],
                        "bare_formula": fully_bare,
                        "phase_tag": tag,
                        "reduced_formula": None,
                        "parse_ok": False,
                        "suspected_ocr_corruption": False,
                        "is_gas": True,
                        "is_nonstoichiometric": False,
                    }
                )
            continue
        corrected = fix_ocr_letters(formula_raw)
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
        nonstoich = comp is not None and is_nonstoichiometric(comp)
        if nonstoich:
            # MP only indexes/searches whole-number formulas -- search using
            # the nearest simple integer ratio (e.g. Fe0.877S -> Fe7S8), and
            # keep the exact fractional bare_formula around so
            # composition_ratio_close() can re-verify candidates against the
            # true measured ratio rather than trusting the approximation.
            reduced = rationalize_composition(comp).reduced_formula
        else:
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
                "is_gas": False,
                "is_nonstoichiometric": nonstoich,
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
    fields = ["material_id", "formula_pretty", "symmetry", "energy_above_hull",
              "is_stable", "nsites", "theoretical", "composition", "database_IDs"]
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
            icsd_ids = d.database_IDs.get("icsd") if d.database_IDs else None
            all_docs.append(
                {
                    "reduced_formula": d.formula_pretty,
                    "material_id": str(d.material_id),
                    "spacegroup": d.symmetry.symbol if d.symmetry else None,
                    "energy_above_hull": d.energy_above_hull,
                    "is_stable": d.is_stable,
                    "nsites": d.nsites,
                    "theoretical": d.theoretical,
                    "composition": str(d.composition),
                    "icsd_n": len(icsd_ids) if icsd_ids else 0,
                }
            )
        tqdm.write(f"  queried {len(batch)} formulas, total candidates so far: {len(all_docs)}")
    return pd.DataFrame(all_docs)


def query_remarks(mpr, material_ids):
    """Batch-fetch MP's provenance 'remarks' (the "User remarks" field shown
    on a material's web page, e.g. ['Pyrrhotite 4C', 'Iron sulfide (7/8)'])
    for a list of material_ids. Returns {material_id: [remark strings]}."""
    lookup = {}
    if not material_ids:
        return lookup
    for batch in chunked(sorted(set(material_ids)), 100):
        for attempt in range(3):
            try:
                docs = mpr.materials.provenance.search(material_ids=batch, 
                                                       fields=["material_id", "remarks"])
                break
            except Exception as e:
                tqdm.write(f"  remarks query error ({e}), retrying...")
                time.sleep(5)
        else:
            tqdm.write(f"  FAILED remarks batch starting with {batch[0]}, skipping")
            continue
        for d in docs:
            lookup[str(d.material_id)] = d.remarks or []
    return lookup


def select_candidate(row, candidates, remarks_lookup=None):
    """Pick a material_id for one toc row from its candidate pool."""
    cands = candidates[candidates["reduced_formula"] == row["reduced_formula"]]

    organic = is_organic_formula(row["bare_formula"])
    if organic:
        # MP's formula= search matches by simplest ratio, so cyclohexane,
        # methylcyclopentane, and methylcyclohexane all land in the same
        # candidate pool. Keep only candidates whose exact composition
        # could actually be tiled from whole molecules of this formula --
        # otherwise even a lone candidate can be a different-sized molecule
        # (e.g. a C4H8 cell can't be built from C6H12).
        cands = cands[cands["composition"].apply(
            lambda c: is_whole_molecule_multiple(row["bare_formula"], c)
        )]

    nonstoich = bool(row.get("is_nonstoichiometric", False))
    if nonstoich:
        # reduced_formula here is only rationalize_composition()'s integer
        # approximation of Barin's exact measured ratio -- verify each
        # candidate's true composition is actually close to that ratio, not
        # just coincidentally sharing the same simplified search formula.
        cands = cands[cands["composition"].apply(
            lambda c: composition_ratio_close(row["bare_formula"], c)
        )]

    if len(cands) == 0:
        return None, "no_mp_match", None

    if nonstoich and remarks_lookup:
        # Non-stoichiometric defect compounds (pyrrhotite, wuestite, ...)
        # commonly have several real superstructure polytypes at the same
        # ratio. MP's remarks/tags field often names the specific mineral
        # (e.g. mp-542794's remarks include "Pyrrhotite 4C") -- prefer a
        # candidate whose remarks mention this row's name over a blind
        # lowest-energy guess.
        name_key = re.split(r"[\s(),]", row["Name"].upper())[0]
        name_matches = cands[cands["material_id"].apply(
            lambda mid: any(name_key in rem.upper() or rem.upper() in name_key
                             for rem in remarks_lookup.get(mid, []))
        )]
        name_matches_exp = name_matches[name_matches["theoretical"] == False]  # noqa: E712
        pool = name_matches_exp if len(name_matches_exp) >= 1 else name_matches
        if len(pool) >= 1:
            # Among several same-named candidates (distinct real polytypes,
            # e.g. pyrrhotite's 3T/4C/5C/... superstructures, can be nearly
            # energy-degenerate), prefer whichever has the most independent
            # ICSD structure determinations -- the more "canonical"/commonly
            # observed form -- before falling back to lowest energy.
            c = pool.sort_values(["icsd_n", "energy_above_hull"], ascending=[False, True]).iloc[0]
            return c["material_id"], "nonstoichiometric_remarks_match", c["spacegroup"]

    if len(cands) == 1:
        c = cands.iloc[0]
        return c["material_id"], "single_candidate", c["spacegroup"]

    if pd.notna(row["phase_tag"]) and row["phase_tag"]:
        hint_sg = POLYMORPH_SPACEGROUP_HINTS.get((row["bare_formula"], row["phase_tag"]))
        if hint_sg:
            match = cands[cands["spacegroup"] == hint_sg]
            if len(match) >= 1:
                match_exp = match[match["theoretical"] == False]  # noqa: E712
                pool = match_exp if len(match_exp) >= 1 else match
                c = pool.sort_values("energy_above_hull").iloc[0]
                return c["material_id"], "spacegroup_hint", c["spacegroup"]
    else:
        # No [TAG] in the formula, but some untagged rows still specify their
        # polymorph via a Name parenthetical alone (e.g. 'TIN (WHITE)') --
        # for those, "most stable by DFT energy" can silently pick a
        # different, colder-stable named polymorph instead.
        hint_sg = UNTAGGED_NAME_SPACEGROUP_HINTS.get(row["bare_formula"])
        if hint_sg:
            match = cands[cands["spacegroup"] == hint_sg]
            if len(match) >= 1:
                match_exp = match[match["theoretical"] == False]  # noqa: E712
                pool = match_exp if len(match_exp) >= 1 else match
                c = pool.sort_values("energy_above_hull").iloc[0]
                return c["material_id"], "name_spacegroup_hint", c["spacegroup"]

    if organic:
        # No spacegroup hint applies (that table is inorganic-only) and
        # multiple candidates survived the whole-molecule check -- for an
        # organic that most likely means distinct isomers of the same size,
        # which composition alone can't disambiguate further.
        return None, "organic_ambiguous_isomers", None

    # Barin's data is entirely experimental, and DFT energy_above_hull (a 0K
    # ground-state proxy) can miss finite-temperature entropic stabilization
    # of the polymorph actually tabulated. Prefer candidates MP has matched
    # to a real ICSD structure (theoretical == False) before falling back to
    # pure lowest-energy among all candidates.
    has_tag = pd.notna(row["phase_tag"]) and bool(row["phase_tag"])
    experimental = cands[cands["theoretical"] == False]  # noqa: E712
    if len(experimental) >= 1:
        c = experimental.sort_values("energy_above_hull").iloc[0]
        method = "most_stable_experimental_polymorph_unresolved" if has_tag else "most_stable_experimental"
        return c["material_id"], method, c["spacegroup"]

    c = cands.sort_values("energy_above_hull").iloc[0]
    method = "most_stable_guess_polymorph_unresolved" if has_tag else "most_stable_guess"
    return c["material_id"], method, c["spacegroup"]


def sanitize_filename(formula: str) -> str:
    # '(' ')' are kept as-is (valid on every filesystem here, just needing
    # quotes in a shell command) so formulas like 'Al2(SO4)3' read as
    # themselves in the filename. '*' (Barin's hydrate separator) is kept
    # too, matching read_all_thermodata_pdf.py's .json naming, so a
    # hydrate's .cif and .json share the same base name.
    return re.sub(r"[^A-Za-z0-9_.()\[\]*-]", "_", formula)


# match_method values where the *specific* polymorph Barin names (a [TAG]
# or a Name parenthetical) was actually cross-checked against MP's own
# data -- a researched spacegroup hint, or a remarks/tags match -- as
# opposed to a stoichiometry-only pick that confirms only *a* structure
# with the right formula, not that it's *this* named polymorph.
PHASE_NAME_VERIFIED_METHODS = {
    "spacegroup_hint",
    "name_spacegroup_hint",
    "nonstoichiometric_remarks_match",
}


def main():
    toc = load_toc()
    gas_organic = toc[toc["is_gas"]].copy()
    toc = toc[~toc["is_gas"]].copy()
    print(f"Non-gas toc rows: {len(toc)} (plus {len(gas_organic)} gas-phase organics, "
          f"recorded as no-structure-expected)")
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

        nonstoich_formulas = set(parseable.loc[parseable["is_nonstoichiometric"], "reduced_formula"].dropna())
        nonstoich_mids = candidates.loc[
            candidates["reduced_formula"].isin(nonstoich_formulas), "material_id"
        ].tolist()
        print(f"Fetching MP remarks for {len(set(nonstoich_mids))} non-stoichiometric-formula "
              f"candidates (for mineral-name disambiguation)...")
        remarks_lookup = query_remarks(mpr, nonstoich_mids)

        results = []
        for _, row in parseable.iterrows():
            mid, method, sg = select_candidate(row, candidates, remarks_lookup)
            results.append({
                **row.to_dict(),
                "material_id": mid,
                "match_method": method,
                "phase_name_verified": (method in PHASE_NAME_VERIFIED_METHODS) if mid else None,
                "spacegroup": sg,
            })
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
            # symprec finds the actual space group and writes the compact
            # conventional cell with real symmetry operations -- without it,
            # pymatgen's CifWriter defaults to unreduced P1 (every site listed
            # explicitly, identity symmetry only), which loses exactly the
            # spacegroup information this pipeline goes to such lengths to
            # resolve per polymorph.
            try:
                cif_text = struct.to(fmt="cif", symprec=0.1)
            except Exception:
                # A handful of structures (disordered occupancies, unusual
                # cells) can make symmetry-finding itself throw -- fall back
                # to the unreduced P1 write rather than losing the CIF.
                cif_text = struct.to(fmt="cif")
            cif_text = f"# Source: Materials Project {mid} (https://materialsproject.org/materials/{mid})\n" + cif_text
            (CIF_DIR / fname).write_text(cif_text)
            cif_paths.append(fname)
        matched["cif_file"] = cif_paths

    # Assemble final outputs
    def unparseable_label(row):
        return "suspected_ocr_corruption" if row["suspected_ocr_corruption"] else "unparseable_formula"

    unparseable = unparseable.assign(
        material_id=None,
        match_method=unparseable.apply(unparseable_label, axis=1),
        phase_name_verified=None,
        spacegroup=None,
    )
    gas_organic = gas_organic.assign(
        material_id=None, match_method="gas_phase_organic", phase_name_verified=None, spacegroup=None
    )
    unmatched = pd.concat(
        [results_df[results_df["material_id"].isna()], unparseable, gas_organic],
        ignore_index=True,
    )

    cols = ["Formula", "Name", "Page number", "bare_formula", "phase_tag",
            "reduced_formula", "material_id", "match_method", "phase_name_verified", "spacegroup", "cif_file"]
    matched.reindex(columns=cols).to_csv(SUMMARY_CSV, index=False)
    unmatched.reindex(columns=[c for c in cols if c != "cif_file"]).to_csv(UNMATCHED_CSV, index=False)

    print(f"\nDone.")
    print(f"  CIFs written: {matched['cif_file'].notna().sum()} -> {CIF_DIR}")
    print(f"  Match summary: {SUMMARY_CSV}")
    print(f"  Unmatched (need ICSD / manual lookup): {len(unmatched)} -> {UNMATCHED_CSV}")
    print(matched["match_method"].value_counts())


if __name__ == "__main__":
    main()
