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

## Upload matching in the app

The FESEM page does **not** match uploads by file name. It compares the **appearance** of your upload to each catalog micrograph using a **perceptual hash** (pHash) and picks the **closest** image. The analysis text shown is whatever is paired on disk with that closest micrograph (true 1:1 between each on-disk image and its analysis file).

Very different images can still produce a “best” match if the catalog is small—interpret distances accordingly.
