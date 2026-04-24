# Soil Analytics Platform

A small **Streamlit** app for **soil and cementitious / carbonated material** workflows: you upload spectroscopy or thermal data, compare it to **reference ranges** from the literature, and (for **FESEM**) work through a **morphology ↔ phase** checklist driven by a YAML table—not by image or ML models. Reference data lives under `config/reference_ranges/`.

## What it does

| Area | In the app |
|------|------------|
| **FTIR** | Upload a CSV, plot bands, run reference checks, download an **HTML** report. |
| **XRD** | Same idea for diffraction data: upload CSV, plots, peak/phase checks from config, report export. |
| **TGA** | Upload CSV, thermal curves, reference range checks, report export. |
| **FESEM** | Shows a **mineral / morphology table** from `config/reference_ranges/fesem_remarks.yaml`, **multiple-choice** questions (shape/texture → phase), and short **soil / microstructure** notes. You can download a text summary; the app does **not** read micrograph files or use vision models. |

The Python package in `src/soil_analytics/` contains parsers, path helpers, and the logic the UI calls.

## Requirements

- **Python 3.10+**
- Dependencies are listed in `pyproject.toml` (e.g. Streamlit, pandas, matplotlib, PyYAML, pydantic, numpy).

## Run locally

From the **repository root** (the folder that contains `pyproject.toml` and `streamlit_app/`):

1. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install the project** in editable mode:

   ```bash
   pip install -e .
   ```

   For development (tests):

   ```bash
   pip install -e ".[dev]"
   ```

3. **Start the app**:

   ```bash
   streamlit run streamlit_app/Home.py
   ```

   A browser window should open. Use the sidebar to switch between **FTIR**, **XRD**, **TGA**, and **FESEM**.

4. **Run the test suite** (if you installed `[dev]`):

   ```bash
   pytest
   ```

## Project layout (short)

- `streamlit_app/` — Streamlit entry (`Home.py`) and `pages/` for each technique
- `src/soil_analytics/` — importable package used by the app
- `config/reference_ranges/` — YAML ranges, bands, peaks, FESEM text
- `tests/` — `pytest` tests
