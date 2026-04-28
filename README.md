# Lime Analytics Platform

A small **Streamlit** app for **lime, cementitious, and carbonated material** workflows: you upload
spectroscopy, diffraction, or thermal **exports**, compare them to **reference ranges** from the
literature, and (for **FESEM**) follow a **progressive habit questionnaire** in YAML to suggest a
likely phase—not by image or ML models. Reference data lives under `config/reference_ranges/`.

## What it does

| Area | In the app |
|------|------------|
| **FTIR** | Upload **.txt** spectrum files (e.g. JCAMP-style or two-column wavenumber vs y), run band checks, download an **HTML** report. |
| **XRD** | Upload **.asc** 2θ–intensity, multi-file overlay, qualitative phase hints, **HTML** report. |
| **TGA** | Upload **CSV** TG data, reference windows, **HTML** report. |
| **FESEM** | Reference table plus **fesem_wizard.yaml** step flow to suggest a **mineral/phase** and a short **interpretation**; optional text download; the app does **not** read micrograph files. |

The Python package in `src/lime_analytics/` contains parsers, path helpers, and the logic the UI calls.

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
- `src/lime_analytics/` — importable package used by the app
- `config/reference_ranges/` — YAML ranges, bands, peaks, FESEM table, FESEM wizard flow
- `tests/` — `pytest` tests
