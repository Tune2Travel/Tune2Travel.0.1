# YouTube Comment Gathering + Emoji Processing — `process_youtube_comments.py`

This folder contains **`process_youtube_comments.py`**, which uses the **YouTube Data API v3** to gather comments for given video IDs, processes emoji-related fields and language, and appends results to **per-video CSV files**. It supports **resume** via a progress JSON.

**Environment:** `project-main-release-Yeliz-Yucel`  
Create/activate from the repo root:

    conda env create -f environment.yml -n project-main-release-Yeliz-Yucel
    conda activate project-main-release-Yeliz-Yucel

> **IMPORTANT — Secrets & Compliance**
> - **Do NOT commit API keys** to the repository. Remove any keys from the script before pushing.
> - Load keys from environment variables or a git-ignored config file (examples below).
> - Ensure YouTube API ToS and your project’s Ethics/TOS are followed (see `/ethics_tos/` if present).

---

## What the script does

1) Rotates through multiple API keys to reduce quota failures.  
2) Fetches comment threads page-by-page (max 100 per API call).  
3) For each comment, writes:
   - `original_comment` (raw text)
   - `comment_no_emojis` (emojis removed)
   - `emojis_in_comment` (only the emojis)
   - `emoji_text_representation` (e.g., `:red_heart:` via `emoji.demojize`)
   - `detected_language` (best-effort; very short texts → `und`/`error`)
   - Metadata: `comment_id`, `author_name`, `published_at`, `published_at_unix`, `like_count`
4) Appends rows into **`<video_id>_comments.csv`** (one CSV per video).  
5) Saves **resume state** in **`comment_progress.json`** (per video: `nextToken`, `pageCount`).

Default targets inside the script (editable):
- Despacito — `kJQP7kiw5Fk`
- See You Again — `RgKAFK5djSk`

---

## Dependencies

If any are missing in your environment, install them:

    pip install google-api-python-client langdetect emoji pandas

*(Optional for other parts of the repo: `pyarrow` for Parquet I/O.)*

---

## Secure API key setup (recommended)

**Option A — Environment variable**  
Store a comma-separated list of keys:

    # macOS/Linux
    export YT_API_KEYS="key1,key2,key3"

    # Windows (PowerShell)
    setx YT_API_KEYS "key1,key2,key3"

Modify the script to read `os.getenv("YT_API_KEYS")` and `split(",")` to fill `API_KEYS`.

**Option B — JSON file (git-ignored)**  
Create `secrets/keys.json`:

    {
      "api_keys": ["key1", "key2", "key3"]
    }

Ensure `secrets/` is listed in `.gitignore`. Modify the script to load `api_keys` from that file.

> If you cannot edit code now, at minimum **delete/obfuscate** hard-coded keys before committing, and keep a local copy only.

---

## How to run

From the **repo root**:

    python process_youtube_comments.py

- The script uses the `TARGET_VIDEO_IDS` list near the bottom—edit it to change which videos are processed.
- On quota errors (HTTP 403), it automatically rotates to the next key. If all keys fail, it stops for that video.

**Resume:**  
If interrupted, re-run the script; it reads `comment_progress.json` and continues from the next page for each video.

---

## Outputs

- **Per-video CSVs** in the current working directory, e.g.:
  
      kJQP7kiw5Fk_comments.csv
      RgKAFK5djSk_comments.csv

- **Progress file**:
  
      comment_progress.json

**CSV columns (in order):**

    comment_id, original_comment, comment_no_emojis, emojis_in_comment,
    emoji_text_representation, detected_language, author_name,
    published_at, published_at_unix, like_count

---

## Common issues & tips

- **Quota/403 errors:** indicate exhausted daily quota; add keys, request more quota, or wait for reset.  
- **HTML in `original_comment`:** YouTube may return HTML entities; strip/convert if needed downstream.  
- **Language detection:** very short/noisy texts may be `und` or `error`.  
- **File permissions:** ensure you can write in the working directory for CSV and progress JSON.  
- **Secrets:** confirm no API keys or PII are pushed to the repo.

---

## Minimal verification checklist (PGR)

- [ ] `comment_progress.json` is created and updates as pages are processed.  
- [ ] `<video_id>_comments.csv` files are produced with the columns listed above.  
- [ ] No API keys are committed; secrets are loaded securely via env/ignored file.  
- [ ] This README explains how to run/resume without external links.
