# FESEM catalog (micrographs + analysis)

Pair each FESEM image with a **single analysis file** (plain text or Markdown).

## Layout

```text
data/fesem_supervised/
  README.md          (this file)
  micrographs/
    sample_a.png
    sample_b.tif
  analysis/
    sample_a.txt     # same stem as the image
    sample_b.md      # .txt or .md
```

Rules:

- Put one image per file under **micrographs/** (e.g. `.png`, `.jpg`, `.tif`).
- Under **analysis/** use the **same base name** as the image: `foo.png` ↔ `foo.txt` or `foo.md`.
- If no matching analysis file exists, the Streamlit FESEM page still lists the micrograph but marks analysis as missing.

Previously this folder used a **`supervised/`** flat layout with CSV labels for ML training. That workflow was removed; rename **`supervised/` → `micrographs/`** and move qualitative write-ups into **analysis/** as described.

## Upload lookup

In the FESEM app, you can upload a file. The app finds a catalog entry by **file basename** and, if the bytes match the micrograph on disk, treats the pair as verified and shows the analysis text.
