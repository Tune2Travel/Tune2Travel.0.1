# Emotion UGC Pipeline

Fully reproducible pipeline for the project repository.

This repository includes:
- English filtering
- spam filtering
- emoji-aware text representation
- transformer-based 7-class emotion classification
- full-corpus inference
- monthly normalized emotion distributions
- lag-1 cosine temporal persistence
- permutation-based null benchmark
- emoji-aware vs emoji-stripped sensitivity
- appendix tables and figures

## Repository structure

```text
emotion_ugc_pipeline/
├── data/
│   ├── comments_raw.csv
│   ├── youtube_spam_dataset.csv
│   ├── gold_annotations.csv
│   ├── gold_eval_predictions.csv                 # optional; script can generate if absent
│   ├── gold_pred_with_synth.csv                  # optional
│   ├── gold_pred_no_synth.csv                    # optional
│   ├── despacito_emotion_labelled.csv            # optional; generated if comments_raw has video column
│   ├── seeyouagain_emotion_labelled.csv          # optional; generated if comments_raw has video column
│   ├── despacito_pred_with_synth.csv             # optional
│   ├── despacito_pred_no_synth.csv               # optional
│   ├── sya_pred_with_synth.csv                   # optional
│   └── sya_pred_no_synth.csv                     # optional
├── outputs/
├── src/
│   ├── __init__.py
│   └── full_pipeline.py
├── requirements.txt
└── README.md
```

## Minimum required input files

### 1) `data/comments_raw.csv`
Required columns:
- `comment_text`
- `published_at`

Optional columns:
- `video_id`
- `video_name`
- `source`

If `video_id` or `video_name` is present, the script will automatically create video-specific outputs.

### 2) `data/youtube_spam_dataset.csv`
Required columns:
- `text`
- `label`

Expected label convention:
- `0` = not spam
- `1` = spam

### 3) `data/gold_annotations.csv`
Required columns:
- `text`
- `label`

Allowed labels:
- `anger`
- `disgust`
- `fear`
- `joy`
- `neutral`
- `sadness`
- `surprise`

Optional column:
- `text_noemoji`

If `text_noemoji` is absent, the script generates it automatically.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/full_pipeline.py \
  --raw-data data/comments_raw.csv \
  --spam-data data/youtube_spam_dataset.csv \
  --gold-data data/gold_annotations.csv \
  --output-dir outputs \
  --model-dir outputs/model \
  --seed 42 \
  --n-permutations 1000 \
  --batch-size 64 \
  --epochs 3 \
  --learning-rate 5e-5
```

## Outputs

The script writes:
- cleaned analytic corpus
- full emotion-labeled corpus
- emoji-stripped predictions
- monthly normalized vectors
- lag-1 values
- Table 3
- Appendix A1–A4
- figures
- model checkpoint
- log file
- Excel workbook with all tables
- run metadata JSON

## Transparency notes

- Month-order shuffling only is used in the permutation test.
- Emojis are treated as measurement cues, not as separate emotion classes.
- Lag-1 is the only inferential temporal test.
- Lag-2 / Lag-3 / Lag-6 are descriptive only.
- If exact manuscript-specific split files are not provided, the script generates reproducible approximations using the supplied seed.
