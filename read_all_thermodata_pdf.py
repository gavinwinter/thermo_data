"""
OCR extraction of thermodynamic data tables from a scanned copy of Barin's
'Thermochemical Data of Pure Substances' (3rd Edition, 1995) -- see README
for the copyright note on the source PDF.

Setup: this is the one thing you need to provide -- your own local copy of
the Barin PDF, passed in via CLI:

    python read_all_thermodata_pdf.py --input-pdf /path/to/barin_1995.pdf

Run with a Python environment that has pdfplumber, opencv, pytesseract, and
the `tesseract` OCR engine itself installed (see README) -- pytesseract is
just a thin wrapper around the `tesseract` binary, which is looked up on
PATH (i.e. whichever environment you've activated), falling back to the
directory containing the current Python interpreter.

Pipeline:
  1. Render every PDF page to an image, clean it up (grayscale, sharpen,
     threshold) for OCR, and save it under barin_jpg_data/.
  2. Look up each formula in toc_barin.csv (a pre-built table of contents
     mapping formula -> page number, tracked in this repo) to find its page.
  3. OCR that page and parse out each phase's T/Cp/S/G/H/.../logKf table,
     validating rows against the thermodynamic identity G = H - T*S and
     monotonicity (dG/dT = -S < 0) to catch and repair OCR digit errors.
  4. Write one JSON file per formula into --output-dir (default:
     barin_json_data/), which query_mp_cifs_from_toc.py's sibling scripts
     do not depend on -- that pipeline only needs toc_barin.csv.

Nothing else about the internals needs to change to run this end to end.
"""
import argparse
import itertools
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from pytesseract import Output
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent


def configure_tesseract():
    """Point pytesseract at the `tesseract` binary/tessdata for whichever
    Python environment is currently active -- checks PATH first (the normal
    case after `conda activate`), then falls back to the directory
    containing this interpreter (the case when a script is run via an
    env's absolute python path without activating it)."""
    tesseract_path = shutil.which("tesseract")
    env_bin = Path(sys.executable).resolve().parent
    if not tesseract_path:
        candidate = env_bin / "tesseract"
        if candidate.exists():
            tesseract_path = str(candidate)
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    tessdata_dir = env_bin.parent / "share" / "tessdata"
    if tessdata_dir.exists():
        os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)

    print("tesseract version:", pytesseract.get_tesseract_version())


configure_tesseract()


# Functions for table of contents lookup
# --------------------------------------
def levenshtein_distance(s1, s2):
    """Edit distance (insertions, deletions, substitutions) between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]



def closest_match(input_string, string_list):
    """Return the entry in string_list with the smallest Levenshtein distance to input_string."""
    min_distance = float('inf')
    closest = None

    for candidate in string_list:
        distance = levenshtein_distance(input_string, candidate)

        if distance < min_distance:
            min_distance = distance
            closest = candidate

    return closest


def fix_zero_before_digit(formula: str) -> str:
    """Replace '0' with 'O' whenever it's immediately followed by a digit
    (e.g. "Ag2C03" -> "Ag2CO3", "Fe203" -> "Fe2O3"), except inside a
    legitimate decimal number like "0.702" or "1.05" -- those zeros are
    real stoichiometric digits, not a misread 'O', and blindly converting
    them would turn e.g. "NbC0.702" into the wrong "NbC0.7O2"."""
    decimal_spans = [m.span() for m in re.finditer(r"\d+\.\d+", formula)]

    def in_decimal(pos):
        return any(start <= pos < end for start, end in decimal_spans)

    chars = list(formula)
    for i, c in enumerate(chars):
        if c == "0" and i + 1 < len(chars) and chars[i + 1].isdigit() and not in_decimal(i):
            chars[i] = "O"
    return "".join(chars)
# --------------------------------------



# Functions for extracting thermodynamic data
# -------------------------------------------
def get_ocr_data(page_num, image_to_data=False, image_to_string=True):
    """OCR one rendered Barin page and return its text as a list of lines.

    Crops the page-number/binding-gutter margins before OCR. With
    image_to_string (the default), runs Tesseract's plain-text mode and
    repairs a few systematic Tesseract artifacts (spaces hallucinated
    after decimal points and minus signs, missing spaces before a merged
    negative number). With image_to_data instead, runs Tesseract's
    word-level bounding-box mode and bins each word into one of the
    table's 10 data columns by its horizontal position, reconstructing
    each row from those bins -- an alternative extraction path for pages
    where the plain-text mode's column spacing gets scrambled.
    """
    img_path = 'barin_jpg_data/Thermochemical_Data_of_Pure_Substances___1995___Barin_Page_{:04}.jpg'.format(page_num)
    img = Image.open(img_path)
    width, height = img.size

    # Setting the points for cropped image (upper left is origin)
    left = 0.05 * width
    top = 0
    right = 0.95 * width
    bottom = height
    img = img.crop(box=(left, top, right, bottom))

    if image_to_string:
        text = pytesseract.image_to_string(img, lang='eng+equ', config='--psm 6')

        # --- THE REGEX REPAIR ---
        # Fix spaces hallucinated immediately after a decimal point (e.g., "-91. 8320" -> "-91.8320")
        text = re.sub(r'(?<=\.)\s+(?=\d+)', '', text)
        
        # Fix spaces hallucinated immediately after ANY minus-like sign (e.g., "~ 918.320" -> "~918.320")
        text = re.sub(r'(?<=[-—~])\s+(?=\d+)', '', text)
        
        # Fix merged columns by injecting a space before ANY minus-like sign that follows a digit
        # Matches a digit, then captures -, —, or ~, and injects a space before it.
        text = re.sub(r'(?<=\d)([—~-])', r' \1', text)
        # ------------------------

        # Detect the sections of string with just numbers
        # (i.e. these are the page numbers) and parse by this
        lines = []
        line = []
        for char in text:
            line.append(char)
            if char == '\n':
                lines.append("".join(line).rstrip('\n'))
                line = []

    elif image_to_data:
        print('here')
        data = pytesseract.image_to_data(img, lang='eng+equ', config='--psm 6', output_type=Output.DICT)
    
        lines_data = {}
        
        # Group recognized words by their Tesseract-assigned line identifiers
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            
            # Skip empty strings or whitespace
            if not text:
                continue
                
            # Create a unique tuple ID for each line based on block, paragraph, and line number
            line_id = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            
            if line_id not in lines_data:
                lines_data[line_id] = []
                
            # Store the text and its horizontal boundaries
            lines_data[line_id].append({
                'text': text,
                'left': data['left'][i],
                'right': data['left'][i] + data['width'][i]
            })
            
        lines = []
        for line_id, words in lines_data.items():
        
            # Skip lines that clearly aren't full data rows to avoid mis-binning headers
            if len(words) < 5:
                lines.append(" ".join([w['text'] for w in words]))
                continue
                
            # Initialize 10 empty string bins for the 10 data columns:
            # T, Cp, S, G_H298_T, H, H_H298, G, dHf, dGf, logKf
            column_strings = [""] * 10
            
            for word in words:
                # Calculate the horizontal center of the word's bounding box
                center_x = (word['left'] + word['right']) / 2
                
                # Get the relative position across the cropped image width (0.0 to 1.0)
                relative_x = center_x / img.width
                
                # Map the relative X coordinate to the correct column index (0 to 9)
                # *Note: May need to slightly tune these threshold boundaries based on crop*
                if relative_x < 0.08: idx = 0      # T
                elif relative_x < 0.17: idx = 1    # Cp
                elif relative_x < 0.27: idx = 2    # S
                elif relative_x < 0.38: idx = 3    # -(G-H298)/T
                elif relative_x < 0.48: idx = 4    # H
                elif relative_x < 0.58: idx = 5    # H-H298
                elif relative_x < 0.69: idx = 6    # G
                elif relative_x < 0.80: idx = 7    # dHf
                elif relative_x < 0.90: idx = 8    # dGf
                else: idx = 9                      # logKf
                
                # Concatenate text into the appropriate bin
                column_strings[idx] += word['text']
                
            # Filter out empty bins and reconstruct the line with single spaces
            clean_row = [s for s in column_strings if s != ""]
            lines.append(" ".join(clean_row))

    return lines


# The table's second header line always starts with the temperature units
# label '[K]', which must not be mistaken for a phase name -- but OCR mangles
# its brackets inconsistently ('{K]', '{[K]', '(K]', etc., the same
# bracket-shape confusion seen throughout this OCR'd source), so matching
# the *shape* of one-or-more opening-bracket-like characters + 'K' +
# one-or-more closing-bracket-like characters catches every variant a
# literal '!= "[K]"' comparison misses -- including a doubled/lowercased
# 'K' (e.g. '[kK]'), which OCR produces just as readily as the single
# uppercase form, and which a strict single-'K' pattern would otherwise
# let through as if it were a genuine new phase name (see CHClI2[g]).
_TEMP_UNIT_LABEL_PATTERN = re.compile(r"^[\[{(]+[kK]+[\]})]+$")


def read_thermo_data(name, lines, debug=False, formula=None):
    """Parse OCR'd page lines into per-phase thermodynamic data tables for one compound.

    Locates the named entry, splits it into phases at each new phase-name
    row, and for each data row repairs common OCR artifacts (missing
    decimal points, digit/letter confusions, hallucinated spacing) before
    parsing its 10 columns (T, Cp, S, -(G-H298)/T, H, H-H298, G, dHf, dGf,
    logKf). Each candidate row is validated against the thermodynamic
    identity G = H - TS; if no combination of parsed floats satisfies it,
    the row falls back to a positional parse rather than being dropped,
    gated by the global constraint that G is monotonically non-increasing
    in T (dG/dT = -S < 0) across the entire table -- including phase
    transitions -- which is what prevents gaps at phase-transition
    boundaries that a per-phase-only check would miss.

    `formula` (optional) is tried as a second channel for locating the
    entry's own header window when `name` doesn't match anywhere on the
    page -- the compound's *name* can pick up an ordinary OCR letter drop
    just like anything else on the page (e.g. Cu3I3[g]'s name OCR'd as
    'TRICOPPER TRIODIDE', missing an I), and with name matching using zero
    fuzzy tolerance by design (see _fuzzy_match_tolerance), that alone was
    enough to miss the header entirely and fall through to parsing from the
    top of the page -- silently picking up whichever *other* entry's table
    happened to be first there instead (Cu3I3[g] pulling CuI[g]'s data).
    The formula is far less prone to this specific failure mode: it's
    shorter, and it sits right next to the name on the same header line.
    """

    reading_phase = False

    T, Cp, S, G_H298_T, H, H_H298, G, ΔHf, ΔGf, logKf = [], [], [], [], [], [], [], [], [], []
    phases = []

    # Tracks the last accepted row's (T, G, S) across the whole table,
    # including across phase boundaries. This is what lets us enforce
    # dG/dT = -S globally, not just within a single phase's list -- which is
    # what let garbage rows through at the very first row of each new phase
    # last time. The direction of that constraint flips once S itself has
    # gone negative (see last_S_global below): most substances never do,
    # but a few (e.g. Cd(OH)2, which decomposes) genuinely tabulate an S
    # that crosses zero, and from that point on G is *supposed* to rise
    # with T, not fall -- enforcing "G must not increase" past that point
    # would drop real data (see NSPT decomposition entries generally).
    last_T_global = None
    last_G_global = None
    last_S_global = None

    lines = [line for line in lines if line != ""]

    # Drop standalone footnote/annotation lines before they can
    # bleed into a neighboring row during OCR line reconstruction.
    # These are lines that are ONLY a single bare float (e.g. "0.860"),
    # which correspond to transition-property annotations, not data rows.
    def is_bare_footnote(line):
        toks = line.split()
        return len(toks) == 1 and re.fullmatch(r'-?\d+\.\d+', toks[0]) is not None

    lines = [line for line in lines if not is_bare_footnote(line)]

    # Barin packs multiple short entries onto one page (e.g. ALUMINIUM COBALT
    # and 5-ALUMINIUM 2-COBALT both fit on the same page), so this compound's
    # own table can start anywhere in `lines`, not at the top. A literal
    # `name in line` substring check is too brittle to find it reliably --
    # names with a numeric prefix ('5-ALUMINIUM 2-COBALT', '18-ALUMINIUM
    # 4-BORON 33-OXIDE', ...) routinely get that prefix's hyphen/spacing
    # OCR'd differently than it was typed here, so the check silently never
    # matches and `lines` is left unsliced -- meaning parsing falls through
    # to whichever *earlier* compound's table happens to be first on the
    # page instead of this one's own.
    #
    # Anchor to the 'Phase T Cp S ...' data-header line itself (same
    # detection _entry_header_windows() uses), not to whichever line the
    # name text happens to land on: a short preceding entry's leftover
    # References-section row can get OCR'd onto the *same line* as the next
    # entry's own name/formula (e.g. 'SOL W1/e e Hu1 MPT= 1918.252.774
    # 5-ALUMINIUM 2-COBALT Al5Co2'), and starting the parse there instead of
    # at the next clean header confuses the phase/data-row detection below.
    #
    # Try an exact normalized match against every header's lookback window
    # first, across the whole page, before considering any fuzzy match:
    # 'ALUMINIUM 3-NICKEL' and '3-ALUMINIUM NICKEL' normalize to strings
    # only one transposition apart (edit distance 2, right at the fuzzy
    # threshold below), so falling back to fuzzy matching per-header instead
    # of exhausting exact matches everywhere first could match one entry to
    # the other's header. Both names normally have their own exact match
    # (OCR usually preserves letters fine; it's the hyphen/spacing around a
    # numeric prefix that's unreliable), so fuzzy matching is only reached
    # for a name that doesn't cleanly appear anywhere as typed.
    name_key = _normalize_for_page_match(name)
    formula_key = _normalize_for_page_match(formula) if formula else ""
    if name_key or formula_key:
        max_dist = _fuzzy_match_tolerance(name_key) if name_key else 0
        header_indices = []
        for i, line in enumerate(lines):
            m = re.match(r"^[^A-Za-z]{0,3}Phase\b[^A-Za-z]{0,10}([A-Za-z])", line, re.IGNORECASE)
            if m and m.group(1).upper() != "H":
                header_indices.append(i)

        def window_key(h, lookback=3):
            return _normalize_for_page_match(" ".join(lines[max(0, h - lookback):h]))

        match_h = None
        if name_key:
            for h in header_indices:
                if name_key in window_key(h):
                    match_h = h
                    break
        if match_h is None and formula_key:
            for h in header_indices:
                if formula_key in window_key(h):
                    match_h = h
                    break
        if match_h is None and name_key:
            for h in header_indices:
                if _fuzzy_find(name_key, window_key(h), max_dist) != -1:
                    match_h = h
                    break
        if match_h is None and formula_key:
            for h in header_indices:
                if _fuzzy_find(formula_key, window_key(h), 0) != -1:
                    match_h = h
                    break
        if match_h is not None:
            lines = lines[match_h:]

    dropped_rows = []

    for i, line in enumerate(lines):

        tokens = line.split()
        if not tokens:
            continue

        first_token = tokens[0]

        # --- OCR PHASE NAME CLEANER ---
        # Normalize Tesseract typos: fix leading symbols (§, $)
        # and trailing lowercase 'l' misreads for numbers (e.g., -A1l -> -A1)
        cleaned_first_token = first_token.replace('§', 'S').replace('$', 'S')
        cleaned_first_token = re.sub(r'([A-Z]\d+)l$', r'\g<1>1', cleaned_first_token)

        # Check if the preceding lines indicate a phase table is active
        preceding_lines_str = " ".join(lines[max(0, i-3):i])
        is_phase_header_nearby = "Phase" in preceding_lines_str

        # 'References' itself gets OCR'd inconsistently (e.g. 'Referenzen'
        # on some pages, seemingly a font/ligature artifact) -- a literal
        # '!= "References"' comparison misses that, which then lets the
        # References section's own 'Phase H/S Cp' sub-header slip past as
        # if it were a genuine new data phase, since the exclusion below
        # never fires and the section-end check further down never breaks
        # out of the loop either. Tolerate up to 2 edits (rather than the
        # usual length-scaled tolerance) since this is a single well-known
        # word being checked in a low-risk context -- collateral false
        # positives on unrelated short tokens are effectively impossible.
        is_references_token = _fuzzy_find(
            "REFERENCES", _normalize_for_page_match(cleaned_first_token), 2
        ) != -1

        # A phase row starts with a non-numeric token (has letters) when a phase table is nearby
        is_phase_start = (
            any(c.isalpha() for c in cleaned_first_token)
            and not _TEMP_UNIT_LABEL_PATTERN.match(cleaned_first_token)
            and not is_references_token
            and (is_phase_header_nearby or reading_phase)
        )

        if is_phase_start:
            if reading_phase:
                phases.append([phase_name, T, Cp, S, G_H298_T, H, H_H298, G, ΔHf, ΔGf, logKf])
            T, Cp, S, G_H298_T, H, H_H298, G, ΔHf, ΔGf, logKf = [], [], [], [], [], [], [], [], [], []
            phase_name = cleaned_first_token  # Save the properly cleaned phase name!
            reading_phase = True

            # Strip the phase name off the line so the numerical parser doesn't choke on it
            line = line[line.find(first_token) + len(first_token):]

        elif is_references_token or _fuzzy_find("REFERENCES", _normalize_for_page_match(line), 2) != -1:
            if reading_phase:
                phases.append([phase_name, T, Cp, S, G_H298_T, H, H_H298, G, ΔHf, ΔGf, logKf])
            reading_phase = False
            break

        if reading_phase and len(line.split()) > 3:

            first_word = line.split()[0]
            if first_word.isupper() and not _TEMP_UNIT_LABEL_PATTERN.match(first_word):
                line = line[line.find(first_word) + len(first_word):]

            # Repair numbers that lost their decimal point in OCR
            # e.g. "34199" -> "34.199", "-964615" -> "-964.615", "-1144412" -> "-1144.412"
            # This is the actual root cause of every dropped row: a missing
            # decimal point makes the float regex skip the token entirely,
            # which shifts every subsequent column's index by one and makes
            # G get compared against what's really ΔHf or ΔGf.
            # Barin's tables use a fixed format: T has 2 decimal places,
            # every other column has 3 -- so this is fully deterministic,
            # not a guess.
            def repair_missing_decimal(token, is_first_token):
                if re.fullmatch(r'-?\d{4,7}', token):
                    sign = '-' if token.startswith('-') else ''
                    digits = token[1:] if sign else token
                    n_decimals = 2 if is_first_token else 3
                    if len(digits) > n_decimals:
                        return sign + digits[:-n_decimals] + '.' + digits[-n_decimals:]
                return token

            tokens = line.split()
            tokens = [repair_missing_decimal(tok, j == 0) for j, tok in enumerate(tokens)]
            line = " ".join(tokens)

            # '§' (section sign) is what Tesseract occasionally emits for
            # a leading '5' digit in this numeric-table font (e.g. InP's SOL-B
            # Cp '55.229' OCR'd as '§5.229') -- left unmapped, the leading digit
            # is silently lost entirely (the float regex below only matches the
            # '5.229' remainder), shifting the value down by an order of
            # magnitude before the G=H-TS/monotonicity checks ever see it.
            line = line.replace('§', '5')
            line = line.replace('B', '8').replace('S', '5').replace('O', '0').replace('l', '1')
            line = re.sub(r'[—–−~_]', '-', line)
            line = line.replace(',', '.')
            line = re.sub(r'(?<=\.)\s+(?=\d+)', '', line)
            line = re.sub(r'(?<=-)\s+(?=\d+)', '', line)
            line = re.sub(r'(?<=\d)-', ' -', line)

            raw_floats = [float(x) for x in re.findall(r'-?\d+\.\d+', line)]

            valid_row = None
            used_fallback = False

            if 9 <= len(raw_floats) <= 15:
                target_lengths = [10, 9] if len(raw_floats) >= 10 else [9]

                for L in target_lengths:
                    for combo in itertools.combinations(raw_floats, L):
                        T_test = abs(combo[0])
                        S_test = combo[2]
                        H_test = combo[4]
                        G_test = combo[6]

                        G_calc = H_test - (T_test * (S_test / 1000.0))

                        if abs(G_calc - G_test) <= 2.0:
                            valid_row = list(combo)
                            if L == 9:
                                valid_row.append(0.0)
                            break
                    if valid_row:
                        break

            # --- Fallback, validated with the actual physics constraint ---
            # dG/dT = -S, and S > 0 always => G must be monotonically
            # non-increasing as T increases, with NO exceptions -- not even
            # across a phase transition (G is continuous there; only the
            # slope kinks). Checked against the last row accepted GLOBALLY
            # (not per-phase), otherwise the first row of each new phase
            # sails through unchecked.
            if not valid_row and 9 <= len(raw_floats):
                candidate = raw_floats[:10] if len(raw_floats) >= 10 else raw_floats[:9]
                if len(candidate) == 9:
                    candidate.append(0.0)

                sane = True
                if last_T_global is not None:
                    T_cand, G_cand = abs(candidate[0]), candidate[6]

                    # T must not go backwards
                    if T_cand < last_T_global - 0.01:
                        sane = False

                    # G must not increase -- except once S has already gone
                    # negative, at which point dG/dT = -S > 0 and a rising G
                    # is the physically expected behavior, not corruption.
                    if (last_S_global is None or last_S_global >= 0) and G_cand > last_G_global + 1.0:
                        sane = False

                # G=H-TS coarse sanity floor, applied even on the very first
                # row of a phase (where there's no previous row to check
                # monotonicity against, so the checks above are skipped
                # entirely). The sieve above already tried this at strict
                # (2.0) tolerance and failed -- otherwise valid_row would
                # already be set -- so a *wide* tolerance here is only about
                # catching a positionally-assigned column that's grossly
                # wrong (e.g. a stray extra OCR'd digit inflating one value
                # by orders of magnitude), not normal per-digit noise. Without
                # this, a garbage first-row G silently becomes last_G_global
                # and fails every real row after it out of the gate.
                T_cand, S_cand, H_cand, G_cand = abs(candidate[0]), candidate[2], candidate[4], candidate[6]
                if abs((H_cand - T_cand * (S_cand / 1000.0)) - G_cand) > 50.0:
                    sane = False

                if sane:
                    valid_row = candidate
                    used_fallback = True

            if not valid_row:
                dropped_rows.append((i, line, raw_floats))
                if debug:
                    print(f"[DROPPED] line {i}: {raw_floats!r} -> {line!r}")
                continue

            # Final global sanity gate, applied even to sieve-passed rows:
            # a combination can satisfy G=H-TS with the wrong permutation of
            # columns and still violate G monotonicity. Reject those too --
            # except the G-must-not-increase half, once S has already gone
            # negative (see last_S_global).
            T_final, G_final, S_final = abs(valid_row[0]), valid_row[6], valid_row[2]
            if last_T_global is not None:
                G_rising_expected = last_S_global is not None and last_S_global < 0
                if T_final < last_T_global - 0.01 or (
                    not G_rising_expected and G_final > last_G_global + 1.0
                ):
                    dropped_rows.append((i, line, raw_floats))
                    if debug:
                        print(f"[DROPPED - failed global monotonicity] line {i}: {valid_row}")
                    continue

            last_T_global, last_G_global, last_S_global = T_final, G_final, S_final

            if debug and used_fallback:
                print(f"[FALLBACK] line {i}: sieve failed, used positional parse -> {valid_row}")

            new_line = valid_row

            T.append(abs(new_line[0]))
            Cp.append(new_line[1])
            S.append(new_line[2])
            G_H298_T.append(new_line[3])
            H.append(new_line[4])
            H_H298.append(new_line[5])
            G.append(new_line[6])
            ΔHf.append(new_line[7])
            ΔGf.append(new_line[8])
            logKf.append(new_line[9])

        else:
            continue

    if debug and dropped_rows:
        print(f"\n{len(dropped_rows)} row(s) still dropped entirely (raw_floats count out of [9,15] range):")
        for i, line, rf in dropped_rows:
            print(f"  line {i}: {rf!r} <- {line!r}")

    # Auto-correct single-digit OCR misreads in Cp/S using the same H/S
    # cross-check that flags them (see apply_consistency_corrections), so a
    # bad digit is fixed here at extraction time rather than needing a
    # separate pass over already-written JSON files.
    for phase_idx, phase_data in enumerate(phases):
        is_last_phase = phase_idx == len(phases) - 1
        corrected_Cp, corrected_S, corrections = apply_consistency_corrections(
            phase_data[1], phase_data[2], phase_data[3], phase_data[5], check_last_row=is_last_phase
        )
        phase_data[2], phase_data[3] = corrected_Cp, corrected_S
        for i, T_i, field, old, new in corrections:
            tqdm.write(f"  [AUTO-CORRECTED] {name} [{phase_data[0]}] T={T_i}: {field} {old} -> {new}")

    return phases


def _breaks_established_trend(T, Cp, i, rel_tol=0.4):
    """True if going from row i-1 to i changes Cp at a rate substantially
    different from the established rate from i-2 to i-1 (per unit T) --
    i.e. this row doesn't continue the trend its own immediate history was
    already following. Used as an extra gate on top of the H/S cross-check
    specifically at the last row of a phase: a backward-only secant estimate
    of Cp systematically approximates the *midpoint* of its span rather than
    the endpoint, so a steep but perfectly linear trend (e.g. CaSO4*2H2O's
    SOL phase increases by exactly the same ~31.8 J/(mol K) every 100 K,
    right through its real last row) can otherwise look identical to a
    genuine anomaly to the H/S check alone, which has no way to tell
    "unusually steep, but consistent" from "wrong." A real misread breaks
    that established local pattern outright (e.g. NH2[g]'s Cp increments by
    ~0.5 per 100 K for many rows, then by ~10.5 at the corrupted one).
    Conservatively returns True (does not block a flag/correction) when
    there isn't enough history (i < 2) or the recent steps are degenerate.
    """
    if i < 2:
        return True
    dT_prev, dT_last = T[i - 1] - T[i - 2], T[i] - T[i - 1]
    if dT_prev <= 0 or dT_last <= 0:
        return True
    rate_prev = (Cp[i - 1] - Cp[i - 2]) / dT_prev
    rate_last = (Cp[i] - Cp[i - 1]) / dT_last
    return abs(rate_last - rate_prev) > max(0.02, rel_tol * abs(rate_prev))


# Cp estimates implied by H's or S's local slope that fall outside this
# range are themselves numerically unreliable (typically because two
# adjacent tabulated values happen to be nearly equal, so dividing by their
# tiny difference amplifies ordinary 3-decimal rounding into a huge, bogus
# result) rather than evidence of anything -- no real substance in this
# dataset has a molar Cp outside it, so such an estimate is discarded
# instead of being compared against.
_PLAUSIBLE_CP_RANGE = (-20.0, 900.0)


def _sign_flip_improves_trend(T, S, i):
    """True if flipping S[i]'s sign brings it closer to the value implied
    by a linear extrapolation from its two preceding rows than leaving it
    as-is -- shared by check_phase_consistency (flagging) and
    apply_consistency_corrections (fixing) so the two can't disagree about
    which negative S values are real. Some entries (e.g. Cd(OH)2's SOL
    phase, which decomposes readily) genuinely tabulate an S that smoothly
    crosses zero and keeps decreasing -- 20.876, 5.787, -6.970 -- where the
    negative sign is real data, not a misread, and flipping it to +6.970
    would actually be the one breaking the trend it was already smoothly
    continuing. Conservatively returns True (treats it as a likely misread)
    when there isn't enough history to tell.
    """
    if i < 2:
        return True
    dT_prev = T[i - 1] - T[i - 2]
    if dT_prev <= 0:
        return True
    rate_prev = (S[i - 1] - S[i - 2]) / dT_prev
    predicted = S[i - 1] + rate_prev * (T[i] - T[i - 1])
    return abs(-S[i] - predicted) < abs(S[i] - predicted)


def _neighbor_index_pairs(i, n):
    """Two (a, b) index pairs spanning i from either side, for estimating a
    local slope at i without using row i's own value. Interior points use
    the immediate neighbor on each side; the first/last point of a phase
    (which only has neighbors on one side) instead uses that side's nearest
    and next-nearest neighbor, giving two pairs of different width so they
    aren't simply duplicates of each other."""
    if i == 0:
        return [(0, 1), (0, 2)] if n > 2 else ([(0, 1)] if n > 1 else [])
    if i == n - 1:
        return [(n - 2, n - 1), (n - 3, n - 1)] if n > 2 else ([(n - 2, n - 1)] if n > 1 else [])
    return [(i - 1, i), (i, i + 1)]


def check_phase_consistency(T, Cp, S, H, cp_margin=5.0, check_last_row=False):
    """Cross-check each row's tabulated Cp and S against the thermodynamic
    identities Cp = dH/dT and Cp = T*dS/dT, to catch a single-token OCR
    misread (a dropped/substituted digit, a flipped sign) in Cp, H, or S
    that leaves the *other* columns internally smooth -- the kind of error
    read_thermo_data()'s own-row G=H-TS sieve can't see, since a row can
    satisfy G=H-TS internally while still being wildly inconsistent with its
    neighbors' trend.

    Two independent checks, run per row:
      - Cp vs. neighbors: estimate Cp two ways from each of H's and S's
        local slope (see _neighbor_index_pairs), without using the row's own
        H/S/Cp values, and flag if the tabulated Cp falls outside *both* the
        H-implied and S-implied range (with cp_margin of slack). Requiring
        both to disagree, rather than either, is what keeps this from firing
        on genuine curvature: real curvature in Cp shows up matching in both
        H's and S's slope, and typically brackets the tabulated Cp between
        the two neighbor-pair estimates rather than missing it entirely.
        Skipped at the first row of a phase, and at the last row *unless*
        check_last_row says no further phase follows this one: a one-sided
        (backward- or forward-only) neighbor estimate systematically
        underestimates Cp specifically where it's curving fastest, and real
        substances routinely show a genuine Cp rise heading into their next
        phase transition (e.g. AgBr's last solid-phase row, just below its
        705 K melting point, is legitimately anomalous pre-melting behavior,
        not an OCR error) at a magnitude this can't distinguish from actual
        corruption. That risk is specific to a row sitting right before a
        transition, though -- the truly last row of a compound's last phase
        (nothing tabulated after it at all) has no such transition to
        explain a deviation, so the caller should pass check_last_row=True
        there (see NH2[g]'s last GAS row, a genuine OCR misread with no
        transition anywhere nearby to legitimize it).
      - S sign: within one phase, entropy is normally always positive, so a
        lone negative value surrounded by otherwise-positive neighbors is
        usually a sign flip, independent of the Cp check above (which can
        miss this, since Cp can still agree fine with H's slope while S
        alone is corrupted). Only flagged when flipping it would actually
        continue the established local trend better than leaving it
        negative (see _sign_flip_improves_trend) -- some entries (e.g.
        Cd(OH)2's SOL phase) genuinely tabulate an S that smoothly crosses
        zero, and flagging that as an error every run would be noise, not
        signal.

    Returns a list of (index, T_i, Cp_i, S_i, reasons) tuples, where reasons
    is a subset of ['Cp', 'S_sign'] naming which check(s) fired.
    """
    n = len(T)
    n_positive_S = sum(1 for v in S if v > 0)
    flagged = []
    for i in range(n):
        cp_wrong = False
        checkable = 0 < i < n - 1 or (i == n - 1 and check_last_row)
        pairs = _neighbor_index_pairs(i, n) if checkable else []
        if len(pairs) >= 2:
            cp_from_H, cp_from_S = [], []
            for a, b in pairs:
                if T[b] <= T[a]:
                    continue
                h_est = 1000.0 * (H[b] - H[a]) / (T[b] - T[a])
                s_est = (S[b] - S[a]) / (math.log(T[b]) - math.log(T[a]))
                if _PLAUSIBLE_CP_RANGE[0] <= h_est <= _PLAUSIBLE_CP_RANGE[1]:
                    cp_from_H.append(h_est)
                if _PLAUSIBLE_CP_RANGE[0] <= s_est <= _PLAUSIBLE_CP_RANGE[1]:
                    cp_from_S.append(s_est)
            h_disagrees = len(cp_from_H) == len(pairs) and not (
                min(cp_from_H) - cp_margin <= Cp[i] <= max(cp_from_H) + cp_margin
            )
            s_disagrees = len(cp_from_S) == len(pairs) and not (
                min(cp_from_S) - cp_margin <= Cp[i] <= max(cp_from_S) + cp_margin
            )
            cp_wrong = h_disagrees and s_disagrees
            if cp_wrong and i == n - 1:
                cp_wrong = _breaks_established_trend(T, Cp, i)

        # A lone negative S against an otherwise all-positive phase (allowing
        # for one other row already suspect for some unrelated reason) is
        # usually a sign flip -- unless flipping it would actually move
        # further from the established trend, which means the negative sign
        # is real (see _sign_flip_improves_trend).
        sign_wrong = (
            S[i] < 0 and n_positive_S >= n - 1 and n_positive_S > 0
            and _sign_flip_improves_trend(T, S, i)
        )

        if cp_wrong or sign_wrong:
            reasons = (["Cp"] if cp_wrong else []) + (["S_sign"] if sign_wrong else [])
            flagged.append((i, T[i], Cp[i], S[i], reasons))
    return flagged


def _invert_pair_estimate(est, i, a, b, T, Cp):
    """Convert a two-point secant estimate `est` (H's or S's implied
    average Cp over [T[a], T[b]], from _neighbor_index_pairs/H or S) into
    an estimate of Cp AT row i, one endpoint of that pair -- rather than
    treating the estimate as if it already approximated Cp[i] itself.

    For an *adjacent* pair (a single interval, i.e. i's immediate
    neighbor), `est` ~ (Cp[i] + Cp[neighbor]) / 2, so Cp[i] is recovered by
    inverting that average against the neighbor's own already-tabulated
    (trusted) Cp. For a *wide* pair that skips one interior point (two
    intervals -- the second, cross-check pair _neighbor_index_pairs hands
    back at the very first/last row of a phase), the skipped point's own
    trusted Cp resolves the second interval the same way.

    Without this inversion, a boundary row's correction ends up averaging
    together one estimate centered on the interval *between* i and its
    neighbor with one centered on a span that only reaches i at its very
    edge -- systematically biasing the "corrected" value toward the
    interior of that span rather than the true value at i. That's what
    made NH2[g]'s last GAS row (T=3000) resolve to 57.785 instead of the
    true 58.186: nothing wrong with the estimates themselves, just averaged
    as if they were both already centered on the row being corrected.
    """
    lo, hi = (a, b) if a < b else (b, a)
    if hi - lo == 1:
        neighbor = a if i == b else b
        return 2 * est - Cp[neighbor]
    mid = lo + 1
    d1, d2 = T[mid] - T[lo], T[hi] - T[mid]
    if i == hi:
        return (est * (d1 + d2) * 2 - d1 * Cp[lo] - (d1 + d2) * Cp[mid]) / d2
    return (est * (d1 + d2) * 2 - d2 * Cp[hi] - (d1 + d2) * Cp[mid]) / d1


def _snap_to_plausible_digit_fix(old_value, estimate, tol=0.1, allow_fallback=True):
    """If changing a single digit of old_value's own printed form lands
    within `tol` of the physics-derived `estimate`, return that exact
    value instead of the estimate itself.

    Every OCR corruption this correction step is designed to catch is a
    single misread digit (a '3' read as '8', a '5' as '6' or '9', etc. --
    see e.g. HgS[g]'s Cp 87.226 for a true 37.226, or NH2[g]'s 68.186 for a
    true 58.186), not a value drawn from nowhere. The physics estimate
    (from H's/S's local slope) is only ever an approximation -- exact for
    perfectly linear Cp, off by a curvature-dependent amount otherwise --
    so when one specific one-digit fix of the original token lands
    squarely on top of it, that fix almost certainly *is* the original
    tabulated value, and is preferable to the estimate's own approximate
    reconstruction of it.

    allow_fallback=False returns None instead of the raw estimate when no
    single-digit fix is found -- see apply_consistency_corrections for why
    a boundary row needs that distinction.
    """
    s = f"{old_value:.3f}"
    best, best_dist = None, tol
    for i, ch in enumerate(s):
        if not ch.isdigit():
            continue
        for d in "0123456789":
            if d == ch:
                continue
            candidate = float(s[:i] + d + s[i + 1:])
            dist = abs(candidate - estimate)
            if dist < best_dist:
                best, best_dist = candidate, dist
    if best is not None:
        return best
    return round(estimate, 3) if allow_fallback else None


def apply_consistency_corrections(T, Cp, S, H, cp_margin=5.0, check_last_row=False):
    """Auto-correct the same Cp/S problems check_phase_consistency() flags,
    using the same H/S-implied estimates that flag them as the correction
    itself -- rather than trying to pattern-match the specific OCR mechanism
    (a substituted digit, an inserted one, a dropped one all look different
    character-by-character but identical from this side: a value that
    doesn't match its neighbors' trend), this trusts the two independent
    physics-derived estimates directly. That's only safe when H's slope and
    S's slope agree with *each other* (not just both disagree with the
    tabulated value) -- e.g. HgS[g]'s GAS Cp goes 37.173, 37.092, [87.226],
    37.263, 37.289: both cp_from_H and cp_from_S land near 37.2 independently,
    which is real corroborating evidence, not a coincidence. If they don't
    agree with each other either, there's no confident correction to make,
    and the row is left for check_phase_consistency() to flag for manual
    review instead.

    S sign flips are corrected when a lone negative value sits among
    otherwise-positive ones *and* flipping it actually continues the
    established local trend better than leaving it negative -- checked
    against a linear extrapolation from the two preceding rows, the same
    kind of check _breaks_established_trend does for Cp. Some entries
    (e.g. Cd(OH)2's SOL phase, which decomposes readily) genuinely tabulate
    an S that smoothly crosses zero and keeps decreasing -- 20.876, 5.787,
    -6.970 -- where the negative sign is real data, not a misread, and
    blindly flipping it to +6.970 would actually be the one breaking the
    trend it was already smoothly continuing. check_last_row controls
    whether the last row is eligible for the *Cp* correction below too --
    pass True only when the caller knows no further phase follows this one
    (see check_phase_consistency).

    Returns (corrected_Cp, corrected_S, corrections), where corrections is a
    list of (index, T_i, field, old_value, new_value) tuples describing what
    changed, for logging/audit purposes.
    """
    n = len(T)
    Cp = list(Cp)
    S = list(S)
    corrections = []
    n_positive_S = sum(1 for v in S if v > 0)

    for i in range(n):
        if (
            S[i] < 0 and n_positive_S >= n - 1 and n_positive_S > 0
            and _sign_flip_improves_trend(T, S, i)
        ):
            corrections.append((i, T[i], "S", S[i], -S[i]))
            S[i] = -S[i]

    correctable_indices = list(range(1, n - 1)) + ([n - 1] if check_last_row and n > 1 else [])
    for i in correctable_indices:
        pairs = _neighbor_index_pairs(i, n)
        if len(pairs) < 2:
            continue
        cp_from_H, cp_from_S = [], []
        for a, b in pairs:
            if T[b] <= T[a]:
                continue
            h_est = 1000.0 * (H[b] - H[a]) / (T[b] - T[a])
            s_est = (S[b] - S[a]) / (math.log(T[b]) - math.log(T[a]))
            if _PLAUSIBLE_CP_RANGE[0] <= h_est <= _PLAUSIBLE_CP_RANGE[1]:
                cp_from_H.append(h_est)
            if _PLAUSIBLE_CP_RANGE[0] <= s_est <= _PLAUSIBLE_CP_RANGE[1]:
                cp_from_S.append(s_est)
        if len(cp_from_H) != len(pairs) or len(cp_from_S) != len(pairs):
            continue
        h_mid = sum(cp_from_H) / len(cp_from_H)
        s_mid = sum(cp_from_S) / len(cp_from_S)
        h_disagrees = not (min(cp_from_H) - cp_margin <= Cp[i] <= max(cp_from_H) + cp_margin)
        s_disagrees = not (min(cp_from_S) - cp_margin <= Cp[i] <= max(cp_from_S) + cp_margin)
        confident = h_disagrees and s_disagrees and abs(h_mid - s_mid) <= cp_margin
        if confident and i == n - 1:
            confident = _breaks_established_trend(T, Cp, i)
        if confident:
            is_boundary = i == 0 or i == n - 1
            if is_boundary:
                # Boundary row: one of the two pairs is "wide" (skips an
                # interior point), so a raw average of h_mid/s_mid is
                # biased toward that span's interior -- invert each pair's
                # estimate against its own known neighbor(s) instead (see
                # _invert_pair_estimate).
                h_final = sum(
                    _invert_pair_estimate(e, i, a, b, T, Cp) for e, (a, b) in zip(cp_from_H, pairs)
                ) / len(cp_from_H)
                s_final = sum(
                    _invert_pair_estimate(e, i, a, b, T, Cp) for e, (a, b) in zip(cp_from_S, pairs)
                ) / len(cp_from_S)
            else:
                h_final, s_final = h_mid, s_mid
            # A boundary row only has neighbors on one side, so its
            # physics estimate can't be cross-checked against a *trusted*
            # value on the other side the way an interior row's can -- it's
            # equally consistent with either "this row was misread" or
            # "this row is genuine curvature the estimate doesn't capture"
            # (e.g. As4S4's LIQ phase Cp accelerates for real approaching
            # its BPT=995K decomposition: 266.534, 268.638, 285.817,
            # 310.093 -- printed cleanly, no misread, just a steepening
            # trend the H/S check reads as anomalous). Requiring an actual
            # single-digit fix of the printed value -- not just the raw
            # estimate -- is what tells apart NH2[g]'s real corruption
            # (68.186 -> a clean one-digit fix landing on 58.186) from
            # that: no one-digit fix of "310.093" lands anywhere near its
            # physics estimate, because 310.093 was never wrong.
            corrected = _snap_to_plausible_digit_fix(
                Cp[i], (h_final + s_final) / 2, allow_fallback=not is_boundary
            )
            if corrected is None:
                continue
            corrections.append((i, T[i], "Cp", Cp[i], corrected))
            Cp[i] = corrected

    return Cp, S, corrections


def get_thermo_data(name, page_num, image_to_data=False, image_to_string=True):
    """OCR page_num and parse out name's phase table in one call."""
    lines = get_ocr_data(page_num, image_to_data=image_to_data, image_to_string=image_to_string)
    phases = read_thermo_data(name, lines)

    return phases


def _normalize_for_page_match(s):
    """Uppercase, strip everything but letters/digits/brackets, and unify
    the OCR confusions that legitimately vary *independently* between
    toc_barin.csv and a given page's own OCR pass -- each was cleaned
    separately, and their noise doesn't line up: 'O'/'0', lowercase
    'l'/capital 'I' (which, once uppercased, folds 'L' and 'I' together --
    a bonus of that is it also unifies 'Al'/'AI' and 'Cl'/'CI', the exact
    same confusion pair applied to those digraphs), and '{'/'}' standing in
    for '['/']' (e.g. 'KBr{g]') -- unified into square brackets *before*
    stripping, so they survive as the bracket characters the rest of this
    function (and page_contains_compound's polymorph-tag check) expect,
    rather than being dropped as punctuation. Every OTHER digit and letter
    stays distinct, since digit counts are exactly what discriminates
    between similarly-named compounds (e.g. 'CaTiO3' vs 'Ca3Ti2O7' both
    reduce to the same letters, but not the same digit-aware skeleton).
    This is a matching heuristic only -- it never touches the formula text
    that actually gets written out.

    Also collapses runs of repeated bracket characters down to one: OCR
    sometimes reads a single '[' as *both* a '{' and a '[' landing right
    next to each other (e.g. 'Cu3l3{[g]' for 'Cu3I3[g]'), which survives
    the {/}->[/] unification above as a doubled '[[' that a formula's own
    single-bracket key then fails to substring-match at all -- silently
    falling through to formula-based verification failing entirely, and
    from there to whatever page/entry happens to be first on the
    fallback page (see Cu3I3[g], which this let mismatch onto CuI[g]'s
    table instead of its own)."""
    s = str(s).replace("{", "[").replace("}", "]")
    s = re.sub(r"[^A-Za-z0-9\[\]]", "", s).upper()
    s = re.sub(r"\[+", "[", s)
    s = re.sub(r"\]+", "]", s)
    s = s.replace("O", "0")
    return s.replace("L", "I")


def _entry_header_windows(lines, lookback=3):
    """Barin packs multiple short entries onto one page when their tables
    are small (e.g. a page can hold AgCN's whole table, then Ag2CO3's
    right after it), so a compound's own header can be anywhere on the
    page, not just the first few lines. Each entry's data table is
    introduced by a 'Phase T Cp S ...' column-header line right after its
    name/formula line -- unlike the References section's differently-worded
    'Phase H/S ...' header, which must NOT be treated as an entry boundary,
    since compounds get *mentioned* there too (e.g. a decomposition remark
    like '(LIQ + CaTiO3)') without that being this page's own entry for
    them. Returns the text immediately preceding each genuine data-header
    line, which is where an entry's own name/formula actually lives.
    """
    windows = []
    for i, line in enumerate(lines):
        # Tolerate OCR noise before 'Phase' (e.g. a stray leading bullet/dot
        # character). Rather than requiring 'T' to follow -- the data
        # header's column-label row is itself OCR'd inconsistently enough
        # that '(Phase) T Cp S' sometimes loses its 'T' entirely, e.g.
        # 'Phase Cy S -(G-H298)/T ...' -- key off what's actually constant:
        # the References section's header is the only place 'Phase' is ever
        # followed by 'H' (from 'Phase H/S ...'), so reject just that.
        m = re.match(r"^[^A-Za-z]{0,3}Phase\b[^A-Za-z]{0,10}([A-Za-z])", line, re.IGNORECASE)
        if m and m.group(1).upper() != "H":
            windows.append(" ".join(lines[max(0, i - lookback) : i]))
    return windows


_TAG_AFTER_PATTERN = re.compile(r"^\[[A-Za-z0-9,]{1,6}\]")


def _looks_like_tag(text_after_match: str) -> bool:
    """True if a page-text position immediately following a formula match
    looks like a genuine short polymorph/isomer tag ('[B]', '[III]',
    '[1,1]') that closes within a few characters -- as opposed to a bare
    '[' that's actually unrelated OCR noise landing right after the match
    by coincidence (e.g. 'IRON' misread as '[RON' in the adjacent Name
    text, which never closes with ']' within a handful of characters)."""
    return bool(_TAG_AFTER_PATTERN.match(text_after_match))


def _fuzzy_match_tolerance(key: str) -> int:
    """Max edit distance to tolerate when fuzzy-matching a normalized name
    skeleton against OCR'd page text. Always 0 (exact match only): chemical
    names can be genuine minimal pairs just like formulas can -- 'AMIDOGEN'
    (NH2) and 'IMIDOGEN' (NH) differ by exactly one letter but are different
    real compounds, and edit-distance-1 tolerance let 'AMIDOGEN' match a
    page that only had 'IMIDOGEN' on it, verifying the *wrong* compound's
    page as if it were correct. Since page_contains_compound() tries the
    formula and the name as two independent channels, either one matching
    exactly is enough -- this only costs verification when *both* fields
    happen to have OCR noise on the same entry, which is rare, versus
    silently accepting a different compound's data, which is far worse.
    """
    return 0


def _fuzzy_find(needle: str, haystack: str, max_dist: int):
    """Index of a substring of haystack within max_dist edits of needle, or
    -1. needle/haystack are short normalized formula/name skeletons here
    (at most a few dozen characters), so trying every window length within
    +/-max_dist at every position is cheap -- this is what survives a
    single stray inserted/dropped/substituted character that isn't one of
    the specific confusions _normalize_for_page_match already unifies
    (e.g. a flat-out extra digit, as in 'Fe2MnO4' OCR'd as 'Fe2Mn0O4').
    """
    n = len(needle)
    if n == 0:
        return -1
    for length in range(max(1, n - max_dist), n + max_dist + 1):
        for start in range(0, len(haystack) - length + 1):
            if levenshtein_distance(needle, haystack[start : start + length]) <= max_dist:
                return start
    return -1


def page_contains_compound(lines, formula, name):
    """Best-effort check that a page plausibly contains this formula's own
    table (not just a passing mention of it elsewhere, e.g. in another
    entry's decomposition remarks). Checking this before trusting a page
    number is what catches a wrong page-offset guess landing on a
    *different, real* compound's table instead of erroring out.

    An untagged formula (e.g. 'Ca2SiO4') must NOT match a page whose header
    continues with a polymorph tag right after it (e.g. 'Ca2SiO4[B]',
    LARNITE, the beta form) -- that's a different, more specific entry with
    its own row in toc_barin.csv, and accepting it here would silently
    pull the wrong polymorph's data. Matching tolerates a small edit
    distance (not just exact substring containment) since the page's own
    OCR pass can introduce noise -- e.g. a stray inserted digit -- that
    _normalize_for_page_match's specific character unifications don't
    cover.
    """
    page_text = _normalize_for_page_match(" ".join(_entry_header_windows(lines)))
    formula_key = _normalize_for_page_match(formula)
    if formula_key:
        # Formulas get no fuzzy tolerance at all, unlike names: a name typo
        # from ordinary OCR letter noise is still recognizably the same
        # compound, but a single-digit difference in a formula is a
        # *different compound* by definition -- 'WBr5[g]' (tungsten
        # pentabromide) fuzzy-matched within edit-distance-1 of 'WBr[g]'
        # (tungsten monobromide, no '5') on a page that only had the latter,
        # which is exactly the wrong-page acceptance this check exists to
        # prevent. The name check below still catches genuine OCR noise
        # like 'Fe2MnO4' OCR'd as 'Fe2Mn0O4' -- an extra stray digit that
        # doesn't happen to spell out a different real stoichiometry.
        max_dist = 0
        idx = page_text.find(formula_key)
        match_len = len(formula_key)
        if idx == -1:
            idx = _fuzzy_find(formula_key, page_text, max_dist)
            # match_len is approximate after a fuzzy hit -- look at a window
            # a bit wider than the needle so a real tag right after isn't missed
            match_len = len(formula_key)
        if idx != -1:
            next_text = page_text[idx + match_len : idx + match_len + 8]
            if "[" in formula_key or not _looks_like_tag(next_text):
                return True
    name_key = _normalize_for_page_match(name)
    if name_key:
        max_dist = _fuzzy_match_tolerance(name_key)
        if name_key in page_text or _fuzzy_find(name_key, page_text, max_dist) != -1:
            return True
    return False


def find_verified_page(formula, name, guessed_page, search_radius=3):
    """Try the precomputed page number first (cheapest, and right most of
    the time); if its header doesn't match this compound, search outward
    (+/-1, +/-2, ...) for a nearby page that does, since Barin's TOC-page-
    to-PDF-page offset is a piecewise approximation that's occasionally off
    by one or two. Returns (page_num, lines, verified) -- if nothing in the
    search radius verifies, falls back to the original guess with
    verified=False so the caller can flag it instead of silently trusting
    unverified data.
    """
    offsets = [0]
    for r in range(1, search_radius + 1):
        offsets += [-r, r]
    for offset in offsets:
        candidate_page = guessed_page + offset
        if candidate_page < 1:
            continue
        try:
            lines = get_ocr_data(candidate_page)
        except FileNotFoundError:
            continue
        if page_contains_compound(lines, formula, name):
            return candidate_page, lines, True
    # Nothing verified -- fall back to the original guess, lines re-fetched
    # by the caller's own get_ocr_data call, flagged as unverified.
    return guessed_page, None, False
# -------------------------------------------



def main(input_pdf, output_dir="barin_json_data"):
    # Converting from Barin .pdf to individual .jpgs for OCR -- skips any
    # page whose .jpg already exists, since re-rendering at 300 DPI is by
    # far the slowest step here and a rerun usually only needs to pick up
    # where a previous one left off (or redo nothing at all).
    print(f"Opening {input_pdf}...")
    with pdfplumber.open(input_pdf) as pdf:
        print(f"Converting {len(pdf.pages)} PDF pages to JPG (skipping ones already rendered)...")
        skipped = 0
        for idx, page in enumerate(tqdm(pdf.pages, desc="Rendering pages")):
            jpg_path = 'barin_jpg_data/Thermochemical_Data_of_Pure_Substances___1995___Barin_Page_{:04}.jpg'.format(idx)
            if os.path.exists(jpg_path):
                skipped += 1
                continue

            # Extracts the page rendering as a PIL Image
            img = page.to_image(resolution=300).original

            if idx == 12:
                img.save('frontpage_og.jpg')

            img_cv2 = np.array(img)

            # Convert the color from RGB to BGR convention for cv2
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,20,-1], [-1,-1,-1]])
            img_cv2 = cv2.filter2D(img_cv2, -1, kernel)

            # Binary thresholding
            th, img_cv2 = cv2.threshold(img_cv2, 128, 255, cv2.THRESH_BINARY)

            img = Image.fromarray(img_cv2)

            if idx == 12:
                img.save('frontpage_processed.jpg')

            img.save(jpg_path)

        print(f"Rendered {len(pdf.pages) - skipped} pages, skipped {skipped} already-rendered pages.")


    # Extracting data from Barin using OCR
    # list of compounds of interest
    # lookup page number corresponding to compounds of interest
    print("Loading toc_barin.csv...")
    toc = pd.read_csv('toc_barin.csv')

    # Ex:
    # lookup_formulas = [
    #                    "Fe", "Fe0.947O", "FeO",
    #                    "Fe2O3", "Fe3O4", "O2[g]",
    #                    "O2[g]",
    #                    "SiO2", "Al2O3", "CaO",
    #                    "TiO2", "TiO2[A]", "Na2O",
    #                    "K2O", "MnO2", "V2O5"]

    # Generate all formulas in Barin
    lookup_formulas = []
    for value in toc['Formula']:
        lookup_formulas.append(value)

    # Get the corresponding page number in Barin
    print(f"Looking up page numbers for {len(lookup_formulas)} formulas...")
    names = []
    page_nums = []
    for lookup_formula in tqdm(lookup_formulas, desc="Looking up pages"):
        try:
            idx = toc.index[toc['Formula'] == lookup_formula].tolist()[0]
        except IndexError:
            lookup_formula = closest_match(lookup_formula,toc['Formula'].tolist())
            idx = toc.index[toc['Formula'] == lookup_formula].tolist()[0]

        names.append(toc['Name'].tolist()[idx])
        # These base offsets were empirically off by one across a diverse
        # sample spanning all three ranges (28/30 needed a -1 correction
        # from find_verified_page) -- corrected here so the first guess is
        # right for the common case again, instead of paying for an extra
        # OCR call on nearly every formula. find_verified_page remains as
        # the safety net for whatever's still off.
        if int(toc['Page number'].tolist()[idx]) <= 924:
            page_nums.append(int(toc['Page number'].tolist()[idx])+114)
        elif (int(toc['Page number'].tolist()[idx]) >= 925 
              and int(toc['Page number'].tolist()[idx]) <= 1200):
            page_nums.append(int(toc['Page number'].tolist()[idx])+115)
        elif (int(toc['Page number'].tolist()[idx]) >= 1200 
              and int(toc['Page number'].tolist()[idx]) >= 925):
            page_nums.append(int(toc['Page number'].tolist()[idx])+116)


    page_nums = [page_num for page_num in page_nums]

    # Create a dedicated directory to avoid dumping 2,500 files onto your root path
    os.makedirs(output_dir, exist_ok=True)

    # Actually extract the thermo data
    print(f"Extracting thermodynamic data for {len(lookup_formulas)} formulas into {output_dir}/...")
    unverified = []
    inconsistent_rows = []
    corrected_page_count = 0
    for lookup_formula, name, page_num in tqdm(
        zip(lookup_formulas, names, page_nums), total=len(lookup_formulas), desc="Extracting thermo data"
    ):

        lookup_formula = fix_zero_before_digit(lookup_formula)

        # Sanitize the formula name to ensure it is a valid, safe filepath.
        # '.' is kept (not stripped into the extension) so non-stoichiometric
        # formulas like "WO2.96" produce "WO2.96.json", not "WO296.json".
        safe_filename = "".join(c for c in lookup_formula if c.isalnum() or c in "_-.[]*") + ".json"

        if os.path.exists(os.path.join(output_dir, safe_filename)):
            continue

        # The precomputed page_num is a piecewise-offset approximation and
        # is occasionally off by a page or two -- verify it actually belongs
        # to this compound before trusting it (see find_verified_page), so a
        # bad guess doesn't silently extract a *different* real compound's
        # table under this formula's name.
        verified_page, lines, verified = find_verified_page(lookup_formula, name, page_num)
        if verified and verified_page != page_num:
            corrected_page_count += 1
            tqdm.write(f"  [page corrected] {lookup_formula}: {page_num} -> {verified_page}")
        if not verified:
            unverified.append((lookup_formula, name, page_num))
            tqdm.write(f"  [UNVERIFIED page] {lookup_formula} ({name}): using unconfirmed page {page_num}")
            lines = get_ocr_data(page_num)
        page_num = verified_page

        phases = read_thermo_data(name, lines, formula=lookup_formula)

        # Convert the raw list-of-lists into a cleanly labeled, nested dictionary
        structured_phases = {}
        for phase_idx, phase_data in enumerate(phases):
            # Skip empty or malformed phase blocks
            if not phase_data or len(phase_data) < 11:
                continue

            phase_name = phase_data[0]
            phase_T, phase_Cp, phase_S, phase_H = phase_data[1], phase_data[2], phase_data[3], phase_data[5]
            is_last_phase = phase_idx == len(phases) - 1
            for i, T_i, Cp_i, S_i, reasons in check_phase_consistency(
                phase_T, phase_Cp, phase_S, phase_H, check_last_row=is_last_phase
            ):
                inconsistent_rows.append((lookup_formula, name, phase_name, T_i, Cp_i, S_i, ",".join(reasons)))
                tqdm.write(
                    f"  [INCONSISTENT] {lookup_formula} [{phase_name}] T={T_i}: "
                    f"Cp={Cp_i}, S={S_i} -- flagged for {','.join(reasons)}"
                )

            structured_phases[phase_name] = {
                "T": phase_data[1],
                "Cp": phase_data[2],
                "S": phase_data[3],
                "-(G-H298)/T": phase_data[4],
                "H": phase_data[5],
                "H-H298": phase_data[6],
                "G": phase_data[7],
                "delta_Hf": phase_data[8],
                "delta_Gf": phase_data[9],
                "logKf": phase_data[10]
            }

        # Pack the metadata and the numerical data into a dictionary
        material_data = {
            "formula": lookup_formula,
            "name": name,
            "page_number": page_num,
            "phases": structured_phases
        }

        file_path = os.path.join(output_dir, safe_filename)

        # Open a unique file for this specific material and dump the dictionary
        with open(file_path, 'w') as f:
            # indent=4 formats the JSON with line breaks and spacing so it is human-readable
            json.dump(material_data, f, indent=4)

    print(f"\nDone. Page number auto-corrected for {corrected_page_count} formulas.")
    if unverified:
        unverified_path = os.path.join(output_dir, "_unverified_pages.csv")
        pd.DataFrame(unverified, columns=["formula", "name", "page_num_used"]).to_csv(unverified_path, 
                                                                                      index=False)
        print(
            f"WARNING: {len(unverified)} formulas could not be verified against any page "
            f"within the search radius -- their extracted data (if any) may belong to the "
            f"wrong compound. Review {unverified_path}."
        )
    if inconsistent_rows:
        inconsistent_path = os.path.join(output_dir, "_thermodynamic_inconsistencies.csv")
        pd.DataFrame(
            inconsistent_rows,
            columns=["formula", "name", "phase", "T", "Cp", "S", "reasons"],
        ).to_csv(inconsistent_path, index=False)
        print(
            f"WARNING: {len(inconsistent_rows)} row(s) fail a Cp/H/S thermodynamic consistency "
            f"check ('Cp' = tabulated Cp disagrees with both H's and S's local slope; 'S_sign' = "
            f"an isolated negative entropy value) -- a likely single-token OCR misread (digit or "
            f"sign) in Cp, H, or S at that row. Review {inconsistent_path}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-pdf", required=True, help="Path to your local copy of the Barin PDF")
    parser.add_argument("--output-dir",
                        default="barin_json_data",
                        help="Directory to write per-formula JSON files into (default: barin_json_data)")
    args = parser.parse_args()
    main(args.input_pdf, args.output_dir)