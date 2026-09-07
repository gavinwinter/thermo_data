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

        # Detect the sections of string with just numbers (i.e. these are the page numbers) and parse by this
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



def read_thermo_data(name, lines, debug=False):
    """
    Fixed version: no longer silently drops rows when the G=H-TS sieve fails
    to find a valid combination. Falls back to a positional parse instead,
    which is what prevents the gaps at phase-transition boundaries.
    """
    
    reading_phase = False

    T, Cp, S, G_H298_T, H, H_H298, G, ΔHf, ΔGf, logKf = [], [], [], [], [], [], [], [], [], []
    phases = []

    # Tracks the last accepted row's (T, G) across the whole table, including
    # across phase boundaries. This is what lets us enforce dG/dT = -S < 0
    # (G must be monotonically non-increasing in T) globally, not just within
    # a single phase's list -- which is what let garbage rows through at the
    # very first row of each new phase last time.
    last_T_global = None
    last_G_global = None

    lines = [line for line in lines if line != ""]

    # Drop standalone footnote/annotation lines before they can
    # bleed into a neighboring row during OCR line reconstruction.
    # These are lines that are ONLY a single bare float (e.g. "0.860"),
    # which correspond to transition-property annotations, not data rows.
    def is_bare_footnote(line):
        toks = line.split()
        return len(toks) == 1 and re.fullmatch(r'-?\d+\.\d+', toks[0]) is not None

    lines = [line for line in lines if not is_bare_footnote(line)]

    for i, line in enumerate(lines):
        if name in line:
            lines = lines[i:]

    dropped_rows = []

    for i, line in enumerate(lines):

        tokens = line.split()
        if not tokens:
            continue

        first_token = tokens[0]
        
        # --- OCR PHASE NAME CLEANER ---
        # Normalize Tesseract typos: fix leading symbols (§, $) and trailing lowercase 'l' misreads for numbers (e.g., -A1l -> -A1)
        cleaned_first_token = first_token.replace('§', 'S').replace('$', 'S')
        cleaned_first_token = re.sub(r'([A-Z]\d+)l$', r'\g<1>1', cleaned_first_token)

        # Check if the preceding lines indicate a phase table is active
        preceding_lines_str = " ".join(lines[max(0, i-3):i])
        is_phase_header_nearby = "Phase" in preceding_lines_str

        # A phase row starts with a non-numeric token (has letters) when a phase table is nearby
        is_phase_start = (
            any(c.isalpha() for c in cleaned_first_token) 
            and cleaned_first_token != "[K]" 
            and cleaned_first_token != "References" 
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

        elif "References" in line:
            if reading_phase:
                phases.append([phase_name, T, Cp, S, G_H298_T, H, H_H298, G, ΔHf, ΔGf, logKf])
            reading_phase = False
            break

        if reading_phase and len(line.split()) > 3:

            first_word = line.split()[0]
            if first_word.isupper() and first_word != "[K]":
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

                    # G must not increase (small tolerance for OCR/rounding noise)
                    if G_cand > last_G_global + 1.0:
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
            # columns and still violate G monotonicity. Reject those too.
            T_final, G_final = abs(valid_row[0]), valid_row[6]
            if last_T_global is not None:
                if T_final < last_T_global - 0.01 or G_final > last_G_global + 1.0:
                    dropped_rows.append((i, line, raw_floats))
                    if debug:
                        print(f"[DROPPED - failed global monotonicity] line {i}: {valid_row}")
                    continue

            last_T_global, last_G_global = T_final, G_final

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

    return phases

def get_thermo_data(name, page_num, image_to_data=False, image_to_string=True):
    
    lines = get_ocr_data(page_num, image_to_data=image_to_data, image_to_string=image_to_string)
    phases = read_thermo_data(name, lines)
    
    return phases
# -------------------------------------------



def main(input_pdf, output_dir="barin_json_data"):
    # Converting from Barin .pdf to individual .jpgs for OCR
    print(f"Opening {input_pdf}...")
    with pdfplumber.open(input_pdf) as pdf:
        print(f"Converting {len(pdf.pages)} PDF pages to JPG...")
        for idx, page in enumerate(tqdm(pdf.pages, desc="Rendering pages")):
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

            img.save('barin_jpg_data/Thermochemical_Data_of_Pure_Substances___1995___Barin_Page_{:04}.jpg'.format(idx))


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
        if int(toc['Page number'].tolist()[idx]) <= 924:
            page_nums.append(int(toc['Page number'].tolist()[idx])+115)
        elif int(toc['Page number'].tolist()[idx]) >= 925 and int(toc['Page number'].tolist()[idx]) <= 1200:
            page_nums.append(int(toc['Page number'].tolist()[idx])+116)
        elif int(toc['Page number'].tolist()[idx]) >= 1200 and int(toc['Page number'].tolist()[idx]) >= 925:
            page_nums.append(int(toc['Page number'].tolist()[idx])+117)


    page_nums = [page_num for page_num in page_nums]

    # Create a dedicated directory to avoid dumping 2,500 files onto your root path
    os.makedirs(output_dir, exist_ok=True)

    # Actually extract the thermo data
    print(f"Extracting thermodynamic data for {len(lookup_formulas)} formulas into {output_dir}/...")
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

        phases = get_thermo_data(name, page_num)

        # Convert the raw list-of-lists into a cleanly labeled, nested dictionary
        structured_phases = {}
        for phase_data in phases:
            # Skip empty or malformed phase blocks
            if not phase_data or len(phase_data) < 11:
                continue

            phase_name = phase_data[0]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-pdf", required=True, help="Path to your local copy of the Barin PDF")
    parser.add_argument("--output-dir", default="barin_json_data", help="Directory to write per-formula JSON files into (default: barin_json_data)")
    args = parser.parse_args()
    main(args.input_pdf, args.output_dir)