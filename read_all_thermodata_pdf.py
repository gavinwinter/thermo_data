"""
OCR extraction of thermodynamic data tables from a scanned copy of Barin's
'Thermochemical Data of Pure Substances' (3rd Edition, 1995) -- see README
for the copyright note on the source PDF.

Usage:
    python read_all_thermodata_pdf.py --input-pdf /path/to/barin_1995.pdf

Requires pdfplumber, opencv, pytesseract, and the `tesseract` binary on
PATH (see README).

Pipeline: render each PDF page to a cleaned-up image -> look up its page in
toc_barin.csv -> OCR the page and parse each phase's T/Cp/S/G/H/.../logKf
table, validating rows against G = H - T*S and monotonicity to catch OCR
digit errors -> write one JSON file per formula into --output-dir.
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
    """Point pytesseract at the active env's `tesseract` binary/tessdata --
    checks PATH first, then falls back to the interpreter's own directory
    (for running via an absolute python path without activating the env)."""
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
    """Replace '0' with 'O' when followed by a digit (e.g. "Ag2C03" ->
    "Ag2CO3"), except inside a decimal like "0.702" -- those zeros are
    real stoichiometric digits, not a misread 'O'."""
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
    """OCR one rendered Barin page into a list of lines, after cropping the
    page-number/gutter margins. image_to_string (default) uses Tesseract's
    plain-text mode with repairs for known spacing artifacts; image_to_data
    instead bins words into the table's 10 columns by x-position -- an
    alternate path for pages where plain-text column spacing gets scrambled.
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


# The header's temperature-unit label '[K]' must not be mistaken for a
# phase name -- OCR mangles its brackets inconsistently ('{K]', '[kK]',
# etc.), so match the *shape* (bracket-like chars + K's + bracket-like
# chars) instead of a literal '[K]' comparison (see CHClI2[g]).
_TEMP_UNIT_LABEL_PATTERN = re.compile(r"^[\[{(]+[kK]+[\]})]+$")


def read_thermo_data(name, lines, debug=False, formula=None):
    """Parse OCR'd page lines into per-phase thermodynamic data tables for
    one compound.

    Locates the entry, splits it into phases, repairs common OCR artifacts,
    and parses each row's 10 columns (T, Cp, S, -(G-H298)/T, H, H-H298, G,
    dHf, dGf, logKf). Rows are validated against G = H - TS; if no
    combination of parsed floats satisfies that, the row falls back to a
    positional parse instead of being dropped, gated by dG/dT = -S < 0
    checked globally rather than per-phase, so gaps at phase-transition
    boundaries aren't missed.

    `formula` (optional) is a second channel for locating the entry's
    header when `name` (zero fuzzy tolerance by design) fails to match due
    to an ordinary OCR letter drop -- otherwise parsing silently falls
    through to whichever other entry's table is first on the page (e.g.
    Cu3I3[g] pulling CuI[g]'s data).
    """

    reading_phase = False

    T, Cp, S, G_H298_T, H, H_H298, G, ΔHf, ΔGf, logKf = [], [], [], [], [], [], [], [], [], []
    phases = []

    # Tracks the last accepted row's (T, G, S) globally (not per-phase), so
    # dG/dT = -S is enforced across phase boundaries too. Flips direction
    # once S goes negative (see last_S_global): a few entries (e.g.
    # Cd(OH)2, which decomposes) genuinely tabulate S crossing zero, after
    # which G is supposed to rise with T -- enforcing "G must not
    # increase" there would drop real data.
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

    # Barin packs multiple short entries per page, so this compound's table
    # can start anywhere in `lines`. Anchor to each 'Phase T Cp S ...'
    # data-header line (same detection as _entry_header_windows()) rather
    # than a literal `name in line` check, which is too brittle against
    # OCR'd numeric-prefix spacing and can land on a preceding entry's
    # leftover References row instead. Exact header matches are tried
    # across the whole page before any fuzzy match, since two differently
    # ordered names can otherwise fall within each other's fuzzy threshold.
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

        # 'References' itself gets OCR'd inconsistently (e.g. 'Referenzen'),
        # so a literal comparison would miss it and let the References
        # section's own sub-header slip through as a new data phase.
        # Tolerate up to 2 edits (fixed, not length-scaled) since this is
        # one well-known word checked in a low-risk context.
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

            # Repair numbers that lost their decimal point in OCR (e.g.
            # "34199" -> "34.199"): a missing point makes the float regex
            # skip the token, shifting every later column's index. Barin's
            # format is fixed -- T has 2 decimal places, every other column
            # has 3 -- so this is deterministic, not a guess.
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

                # G=H-TS coarse sanity floor, applied even on a phase's
                # first row (no previous row to check monotonicity
                # against). The sieve already tried this at strict (2.0)
                # tolerance and failed, so this wide tolerance only catches
                # a grossly wrong positional column, not per-digit noise --
                # otherwise a garbage first-row G poisons last_G_global.
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
    """True if row i's Cp-vs-T rate deviates from the established rate at
    i-1 -- an extra gate used at a phase's last row, where a one-sided
    secant estimate systematically approximates the *midpoint* of its span
    rather than the endpoint, so a steep-but-linear trend (e.g. CaSO4*2H2O's
    steady ~31.8 J/(mol K) rise per 100 K) can otherwise look identical to a
    genuine anomaly to the H/S check alone. Conservatively returns True when
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


# Cp estimates outside this range are numerically unreliable (typically
# two near-equal tabulated values amplifying ordinary rounding through
# division), not evidence of anything real -- no substance here has a
# molar Cp outside it, so such estimates are discarded, not compared.
_PLAUSIBLE_CP_RANGE = (-20.0, 900.0)


def _sign_flip_improves_trend(T, S, i):
    """True if flipping S[i]'s sign fits the trend extrapolated from its
    two preceding rows better than leaving it as-is -- shared by
    check_phase_consistency (flagging) and apply_consistency_corrections
    (fixing) so the two can't disagree. Some entries (e.g. Cd(OH)2's
    decomposing SOL phase) genuinely tabulate an S that smoothly crosses
    zero, where the negative sign is real, not a misread. Conservatively
    returns True when there isn't enough history to tell.
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
    local slope at i without using row i's own value. Boundary points (only
    one-sided neighbors) get two different-width pairs from that side
    instead, so they aren't duplicates of each other."""
    if i == 0:
        return [(0, 1), (0, 2)] if n > 2 else ([(0, 1)] if n > 1 else [])
    if i == n - 1:
        return [(n - 2, n - 1), (n - 3, n - 1)] if n > 2 else ([(n - 2, n - 1)] if n > 1 else [])
    return [(i - 1, i), (i, i + 1)]


def check_phase_consistency(T, Cp, S, H, cp_margin=5.0, check_last_row=False):
    """Cross-check each row's tabulated Cp and S against Cp = dH/dT and
    Cp = T*dS/dT, to catch a single-token OCR misread (digit or sign) in
    Cp, H, or S that leaves the *other* columns internally smooth -- an
    error read_thermo_data()'s own-row G=H-TS check can't see, since a row
    can satisfy G=H-TS while still being inconsistent with its neighbors.

    Two independent checks per row:
      - Cp vs. neighbors: estimate Cp from H's and S's local slope on each
        side (see _neighbor_index_pairs), and flag if the tabulated Cp
        falls outside *both* estimates (with cp_margin slack) -- requiring
        both to disagree avoids false positives on genuine curvature, which
        typically shows up consistently in both. Skipped at a phase's first
        row, and at its last row unless check_last_row=True (no further
        phase follows): a one-sided estimate can't tell a real
        pre-transition Cp rise (e.g. AgBr nearing its melting point) from
        corruption, and that risk only exists right before a transition.
      - S sign: a lone negative S among otherwise-positive neighbors is
        flagged as a sign flip only when flipping it fits the local trend
        better (see _sign_flip_improves_trend) -- some entries (e.g.
        Cd(OH)2) genuinely tabulate S crossing zero.

    Returns a list of (index, T_i, Cp_i, S_i, reasons) tuples, reasons
    being a subset of ['Cp', 'S_sign'].
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
    average Cp over [T[a], T[b]]) into an estimate of Cp AT row i, an
    endpoint of that pair, rather than treating `est` as if it already
    approximated Cp[i] itself.

    For an adjacent pair, Cp[i] is recovered by inverting the average
    against the neighbor's own trusted Cp; for a wide pair (skips one
    interior point), the skipped point's trusted Cp resolves it the same
    way. Without this inversion, a boundary correction is biased toward
    the interior of the span rather than the true value at i -- what made
    NH2[g]'s last GAS row resolve to 57.785 instead of the true 58.186.
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
    """If changing a single digit of old_value's printed form lands within
    `tol` of the physics-derived `estimate`, return that exact value
    instead of the estimate itself -- OCR corruption here is consistently
    a single misread digit (e.g. HgS[g]'s 87.226 for a true 37.226), so a
    one-digit fix landing on the estimate is almost certainly the true
    tabulated value, more reliable than the estimate's own approximation.

    allow_fallback=False returns None instead of the raw estimate when no
    single-digit fix is found (see apply_consistency_corrections).
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
    using the same H/S-implied estimates as the correction itself. Only
    applied when H's and S's estimates agree with *each other* (not just
    both disagreeing with the tabulated value) -- e.g. HgS[g]'s cp_from_H
    and cp_from_S both land near 37.2 independently, real corroborating
    evidence. Otherwise the row is left for check_phase_consistency() to
    flag for manual review.

    S sign flips are corrected when a lone negative value sits among
    otherwise-positive ones and flipping it fits the local trend better
    (see _sign_flip_improves_trend) -- some entries (e.g. Cd(OH)2) genuinely
    tabulate S crossing zero. check_last_row controls whether the last row
    is eligible for the Cp correction too (see check_phase_consistency).

    Returns (corrected_Cp, corrected_S, corrections), corrections being a
    list of (index, T_i, field, old_value, new_value) tuples for audit.
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
            # A boundary row only has neighbors on one side, so its physics
            # estimate can't be cross-checked against a trusted value the
            # other side the way an interior row's can -- equally
            # consistent with "misread" or "genuine curvature the estimate
            # doesn't capture" (e.g. As4S4's LIQ Cp really does accelerate
            # near its decomposition point). Requiring an actual
            # single-digit fix of the printed value, not just the raw
            # estimate, is what tells NH2[g]'s real corruption apart from
            # that.
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
    """Uppercase, strip to letters/digits/brackets, and unify OCR
    confusions that vary independently between toc_barin.csv and a page's
    own OCR pass: 'O'/'0', lowercase 'l'/capital 'I' (also unifying
    'Al'/'AI' and 'Cl'/'CI'), and '{'/'}' standing in for '['/']'. Every
    other digit/letter stays distinct, since digit counts discriminate
    between similarly-named compounds (e.g. 'CaTiO3' vs 'Ca3Ti2O7').
    Matching heuristic only -- never touches the formula text written out.

    Also collapses repeated bracket characters to one: OCR sometimes reads
    a single '[' as both '{' and '[' landing next to each other (e.g.
    'Cu3l3{[g]' for 'Cu3I3[g]'), which otherwise survives as a doubled
    '[[' that a formula's single-bracket key fails to match, falling
    through to the wrong page's entry (see Cu3I3[g] mismatching onto
    CuI[g])."""
    s = str(s).replace("{", "[").replace("}", "]")
    s = re.sub(r"[^A-Za-z0-9\[\]]", "", s).upper()
    s = re.sub(r"\[+", "[", s)
    s = re.sub(r"\]+", "]", s)
    s = s.replace("O", "0")
    return s.replace("L", "I")


def _entry_header_windows(lines, lookback=3):
    """Barin packs multiple short entries onto one page (e.g. AgCN then
    Ag2CO3), so an entry's header can be anywhere, not just at the top.
    Each entry's table is introduced by a 'Phase T Cp S ...' header line,
    unlike the References section's 'Phase H/S ...' header, which must NOT
    count as an entry boundary (compounds get mentioned there too, e.g. in
    a decomposition remark, without owning that page). Returns the text
    immediately preceding each genuine data-header line, where an entry's
    own name/formula lives.
    """
    windows = []
    for i, line in enumerate(lines):
        # Tolerate OCR noise before 'Phase' (a stray bullet/dot) and don't
        # require 'T' to follow -- the header row is OCR'd inconsistently
        # enough to lose its 'T' (e.g. 'Phase Cy S -(G-H298)/T ...'). Key
        # off what's actually constant: only the References header ever
        # follows 'Phase' with 'H' (from 'Phase H/S ...'), so reject that.
        m = re.match(r"^[^A-Za-z]{0,3}Phase\b[^A-Za-z]{0,10}([A-Za-z])", line, re.IGNORECASE)
        if m and m.group(1).upper() != "H":
            windows.append(" ".join(lines[max(0, i - lookback) : i]))
    return windows


_TAG_AFTER_PATTERN = re.compile(r"^\[[A-Za-z0-9,]{1,6}\]")


def _looks_like_tag(text_after_match: str) -> bool:
    """True if the text right after a formula match looks like a genuine
    short polymorph/isomer tag ('[B]', '[III]') that closes within a few
    characters -- as opposed to unrelated OCR noise (e.g. 'IRON' misread
    as '[RON') that never closes with ']'."""
    return bool(_TAG_AFTER_PATTERN.match(text_after_match))


def _fuzzy_match_tolerance(key: str) -> int:
    """Max edit distance for fuzzy-matching a normalized name skeleton
    against OCR'd page text. Always 0: chemical names can be genuine
    minimal pairs ('AMIDOGEN' vs 'IMIDOGEN', different compounds one
    letter apart), so any tolerance risks verifying the wrong compound's
    page. Since page_contains_compound() tries formula and name as
    independent channels, either matching exactly is enough.
    """
    return 0


def _fuzzy_find(needle: str, haystack: str, max_dist: int):
    """Index of a substring of haystack within max_dist edits of needle, or
    -1. Cheap here since needle/haystack are short formula/name skeletons --
    catches a stray inserted/dropped/substituted character not already
    covered by _normalize_for_page_match (e.g. 'Fe2MnO4' OCR'd as
    'Fe2Mn0O4').
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
    table, not just a passing mention elsewhere (e.g. a decomposition
    remark) -- catches a wrong page-offset guess landing on a different
    real compound's table.

    An untagged formula (e.g. 'Ca2SiO4') must NOT match a page whose
    header continues with a polymorph tag (e.g. 'Ca2SiO4[B]') -- that's a
    separate toc entry, and accepting it would silently pull the wrong
    polymorph. Matching tolerates a small edit distance for page-specific
    OCR noise that _normalize_for_page_match doesn't already cover.
    """
    page_text = _normalize_for_page_match(" ".join(_entry_header_windows(lines)))
    formula_key = _normalize_for_page_match(formula)
    if formula_key:
        # Formulas get no fuzzy tolerance, unlike names: an ordinary OCR
        # letter typo still reads as the same compound, but a single-digit
        # formula difference is a *different compound* (e.g. 'WBr5[g]' vs
        # 'WBr[g]'). The name check below still catches genuine noise like
        # 'Fe2MnO4' OCR'd as 'Fe2Mn0O4'.
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
    """Try the precomputed page first; if its header doesn't match, search
    outward (+/-1, +/-2, ...) since Barin's TOC-to-PDF page offset is
    occasionally off by one or two. Returns (page_num, lines, verified) --
    falls back to the original guess with verified=False if nothing in
    the search radius matches, so the caller can flag it instead of
    silently trusting unverified data.
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
        # sample (28/30 needed a -1 correction from find_verified_page) --
        # corrected here so the common case doesn't pay for an extra OCR
        # call. find_verified_page remains the safety net for the rest.
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