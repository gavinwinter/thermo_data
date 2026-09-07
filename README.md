This repository contains parsing and data-structuring tools intended for researchers who hold a legal copy of Ihsan Barin's 'Thermochemical Data of Pure Substances' (3rd Edition, 1995). This repository does not distribute any copyrighted text, images, or extracted datasets.

## Setup

Install the dependencies (numpy, pandas, pdfplumber, opencv, pytesseract + the `tesseract` binary, Pillow, tqdm, pymatgen, mp-api) into a single environment, then export a free Materials Project API key (from https://next.materialsproject.org/api):

```
export MP_API_KEY=your_key_here
```

Nothing else needs to be edited in any of the scripts below -- paths are resolved relative to each script's own location.

## Pipeline

1. OCR the Barin PDF into per-formula thermodynamic data (the only required CLI input is your local copy of the PDF, this step can take about an hour):

   `python read_all_thermodata_pdf.py --input-pdf /path/to/local/barin_1995.pdf --output-dir ./barin_json_data/`

2. Retrieve CIF crystal structures for every phase in `toc_barin.csv`, first from Materials Project:

   `python query_mp_cifs_from_toc.py`

3. Then fill in what Materials Project didn't have from the free Crystallography Open Database:

   `python query_cod_cifs_from_toc.py`

   Whatever's left in `cif_unmatched_final.csv` needs a manual lookup in ICSD (no public/scriptable API exists for it).
