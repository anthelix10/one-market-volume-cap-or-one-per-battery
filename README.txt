# One market-volume cap or one per battery?

Python code used for the analysis in:

**“One market-volume cap or one per battery? A formal decomposition of multi-asset battery backtests in Great Britain.”**

The manuscript and full research package are archived on Zenodo:

**https://doi.org/10.5281/zenodo.21428411**

The Zenodo record contains the processed inputs, saved results, supporting material and the fixed version of the project. This GitHub repository is a smaller code-only companion so that the scripts can be viewed without downloading the full archive.

**Current status:** preprint; not peer reviewed.

## Files in this repository

- `code/` contains the analysis, checking and figure scripts.
- `requirements.txt` lists the Python packages used by the scripts.
- `code/README.txt` gives a short description of each script.
- `.github/workflows/code-check.yml` runs a basic code check on GitHub.
- `CITATION.cff` supplies the citation shown by GitHub.
- `LICENSE.txt` covers the source code in this repository.

The manuscript, data and saved outputs are not copied here because they are already included in the Zenodo package.

## Running the code

The scripts were prepared for Python 3.12.

### 1. Download the project files from Zenodo

Download `analysis_code_inputs_results_v1.1.1.zip` from the Zenodo record. After extracting it, copy these two folders into the root of this repository:

```text
inputs/
paper_of_record/
```

The folder layout should then look like this:

```text
one-market-volume-cap-or-one-per-battery/
├── code/
├── inputs/
├── paper_of_record/
├── README.md
└── requirements.txt
```

The two Zenodo folders are ignored by Git, so they stay on your computer and are not uploaded to this repository.

### 2. Create a Python environment

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Check the saved results

Run these commands from the repository root:

```bash
python code/run_paper_of_record.py --validate-stored
python code/run_scope_extensions.py --validate-stored
python code/run_cost_feasibility_extensions.py --validate-stored
python code/rebuild_phi_calibration.py --validate-stored
python code/run_partition_extensions.py --validate
```

### 4. Make the figures

```bash
python code/make_article_figures.py
```

### 5. Rebuild the analyses

The rebuilding commands take longer than the stored-result checks:

```bash
python code/run_paper_of_record.py --rebuild-all
python code/run_scope_extensions.py --rebuild
python code/run_cost_feasibility_extensions.py --rebuild-all
python code/rebuild_phi_calibration.py --write
python code/run_partition_extensions.py --rebuild
```

## Checking the processed inputs against the raw responses

The raw-response archive is a separate download on the same Zenodo record:

```text
elexon_mid_pn_b1610_raw_responses_2023_2025.zip
```

It can be checked without extracting it:

```bash
python code/verify_inputs_from_raw_archive.py \
  --raw-archive /path/to/elexon_mid_pn_b1610_raw_responses_2023_2025.zip
```

GitHub also reads `CITATION.cff` and shows the same citation through its **Cite this repository** menu.
