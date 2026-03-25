from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from langdetect import detect
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

LABEL_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
EMOTION_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
DEFAULT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


@dataclass
class Config:
    raw_data: str
    spam_data: str
    gold_data: str
    output_dir: str
    model_dir: str
    model_name: str
    seed: int
    n_permutations: int
    batch_size: int
    epochs: int
    learning_rate: float
    max_features_spam: int
    train_with_synth: bool
    synth_fear: int
    synth_anger_disgust: int


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def clean_text_basic(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_english(text: str) -> bool:
    try:
        return detect(str(text)) == "en"
    except Exception:
        return False


def demojize_text(text: str) -> str:
    return emoji.demojize(str(text))


def remove_emoji_text(text: str) -> str:
    return emoji.replace_emoji(str(text), replace="")


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(row_sums, axis=0)


def cosine_between_rows(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def mean_lag_cosine(monthly_vectors: pd.DataFrame, lag: int = 1) -> float:
    arr = monthly_vectors.to_numpy()
    if len(arr) <= lag:
        return np.nan
    sims = [cosine_between_rows(arr[i], arr[i + lag]) for i in range(len(arr) - lag)]
    return float(np.mean(sims))


def sd_lag_cosine(monthly_vectors: pd.DataFrame, lag: int = 1) -> float:
    arr = monthly_vectors.to_numpy()
    if len(arr) <= lag:
        return np.nan
    sims = [cosine_between_rows(arr[i], arr[i + lag]) for i in range(len(arr) - lag)]
    return float(np.std(sims, ddof=1))


def permutation_test_lag1(monthly_vectors: pd.DataFrame, n_permutations: int = 1000, random_state: int = 42) -> dict:
    rng = np.random.default_rng(random_state)
    observed = mean_lag_cosine(monthly_vectors, lag=1)
    arr = monthly_vectors.to_numpy()
    permuted_stats = []
    for _ in range(n_permutations):
        idx = rng.permutation(len(arr))
        shuffled = arr[idx]
        sims = [cosine_between_rows(shuffled[i], shuffled[i + 1]) for i in range(len(shuffled) - 1)]
        permuted_stats.append(float(np.mean(sims)))
    permuted_stats = np.array(permuted_stats)
    p_value = float(np.mean(permuted_stats >= observed))
    p_report = "< .001" if p_value == 0.0 else f"{p_value:.3f}"
    return {
        "observed": observed,
        "null_mean": float(permuted_stats.mean()),
        "p_value_raw": p_value,
        "p_report": p_report,
        "permuted_stats": permuted_stats.tolist(),
    }


def monthly_emotion_vectors(df: pd.DataFrame, date_col: str = "published_at", emotion_col: str = "emotion_label", emotion_order: Optional[list[str]] = None) -> pd.DataFrame:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, emotion_col])
    work["month"] = work[date_col].dt.to_period("M").astype(str)
    counts = work.groupby(["month", emotion_col]).size().unstack(fill_value=0)
    if emotion_order is not None:
        for e in emotion_order:
            if e not in counts.columns:
                counts[e] = 0
        counts = counts[emotion_order]
    return normalize_rows(counts).sort_index()


def safe_round(x, nd=4):
    return np.nan if pd.isna(x) else round(float(x), nd)


def generate_synthetic_examples(cfg: Config) -> pd.DataFrame:
    fear_templates = [
        "this makes me feel scared",
        "i feel afraid listening to this",
        "this gives me fear",
        "this is frightening",
        "this makes me anxious and scared",
    ]
    anger_templates = [
        "this makes me angry",
        "i feel mad reading this",
        "this is infuriating",
        "this triggered anger",
        "i am upset and angry",
    ]
    disgust_templates = [
        "this feels disgusting",
        "i feel disgust",
        "this is revolting",
        "this makes me feel sick",
        "this is gross and disturbing",
    ]
    rows = []
    for i in range(cfg.synth_fear):
        rows.append({"text": fear_templates[i % len(fear_templates)], "label": "fear"})
    half = cfg.synth_anger_disgust // 2
    for i in range(half):
        rows.append({"text": anger_templates[i % len(anger_templates)], "label": "anger"})
    for i in range(cfg.synth_anger_disgust - half):
        rows.append({"text": disgust_templates[i % len(disgust_templates)], "label": "disgust"})
    synth = pd.DataFrame(rows)
    synth["label_id"] = synth["label"].map(LABEL_MAP)
    synth["text_noemoji"] = synth["text"].apply(remove_emoji_text)
    return synth


def tokenize_function(tokenizer, example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)


def predict_batches(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            batch_preds = outputs.logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)
    return np.array(preds)


def plot_temporal_persistence(monthly_norm: pd.DataFrame, outpath: Path, title: str) -> None:
    ax = monthly_norm.plot.area(figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Time (Month)")
    ax.set_ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_engagement_vs_emotion(df: pd.DataFrame, monthly_norm: pd.DataFrame, outpath: Path, title: str) -> None:
    volume = df.groupby("month").size()
    dominant = monthly_norm.max(axis=1)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(volume.index, volume.values, label="Comment Volume")
    ax1.set_ylabel("Volume")
    ax2 = ax1.twinx()
    ax2.plot(dominant.index, dominant.values, linestyle="--", label="Dominant Emotion Share")
    ax2.set_ylabel("Emotion Share")
    fig.legend(loc="upper right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar(values: list[float], labels: list[str], outpath: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bootstrap_distribution(gold_df: pd.DataFrame, model, tokenizer, device, batch_size: int, outpath: Path) -> None:
    scores = []
    for _ in range(1000):
        sample = resample(gold_df, replace=True, random_state=None)
        y_true = sample["label_id"].to_numpy()
        y_pred = predict_batches(model, tokenizer, sample["text"].tolist(), device, batch_size=batch_size)
        scores.append(f1_score(y_true, y_pred, average="macro"))
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=30)
    plt.title("Bootstrap Distribution of Macro-F1 (N=1000)")
    plt.xlabel("Macro-F1")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_learning_curve(gold_df: pd.DataFrame, model, tokenizer, device, batch_size: int, outpath: Path, seed: int) -> None:
    sizes = [100, 200, 400, 600, 800, min(999, len(gold_df))]
    scores = []
    for s in sizes:
        subset = gold_df.sample(s, random_state=seed)
        y_true = subset["label_id"].to_numpy()
        y_pred = predict_batches(model, tokenizer, subset["text"].tolist(), device, batch_size=batch_size)
        scores.append(f1_score(y_true, y_pred, average="macro"))
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, scores, marker="o")
    plt.xlabel("Training Size")
    plt.ylabel("Macro-F1")
    plt.title("Learning Curve")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def prepare_gold(gold_path: Path) -> pd.DataFrame:
    gold = pd.read_csv(gold_path)
    ensure_columns(gold, ["text", "label"], "gold_annotations")
    gold = gold.copy()
    gold["text"] = gold["text"].astype(str).apply(clean_text_basic).apply(demojize_text)
    if "text_noemoji" not in gold.columns:
        gold["text_noemoji"] = gold["text"].apply(remove_emoji_text)
    gold["label"] = gold["label"].astype(str).str.strip().str.lower()
    gold = gold[gold["label"].isin(LABEL_MAP.keys())].copy()
    gold["label_id"] = gold["label"].map(LABEL_MAP)
    return gold


def train_emotion_model(gold: pd.DataFrame, cfg: Config, model_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=7)
    model.to(device)
    train_df, val_df = train_test_split(
        gold[["text", "label_id", "text_noemoji"]].copy(),
        test_size=0.1,
        stratify=gold["label_id"],
        random_state=cfg.seed,
    )
    if cfg.train_with_synth:
        synth = generate_synthetic_examples(cfg)
        train_df = pd.concat([train_df, synth[["text", "label_id", "text_noemoji"]]], ignore_index=True)
    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    train_ds = train_ds.map(lambda ex: tokenize_function(tokenizer, ex), batched=False)
    val_ds = val_ds.map(lambda ex: tokenize_function(tokenizer, ex), batched=False)
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        seed=cfg.seed,
        logging_steps=10,
        load_best_model_at_end=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return trainer, tokenizer, model, train_df, val_df


def load_or_train_model(gold: pd.DataFrame, cfg: Config, device: torch.device):
    model_dir = Path(cfg.model_dir)
    if (model_dir / "config.json").exists():
        logging.info("Loading existing model from %s", model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.to(device)
        train_df, val_df = train_test_split(
            gold[["text", "label_id", "text_noemoji"]].copy(),
            test_size=0.1,
            stratify=gold["label_id"],
            random_state=cfg.seed,
        )
        return None, tokenizer, model, train_df, val_df
    logging.info("Training new model")
    return train_emotion_model(gold, cfg, model_dir, device)


def save_video_splits(full_df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    video_dfs = {}
    if "video_name" in full_df.columns:
        grouped = full_df.groupby("video_name")
    elif "video_id" in full_df.columns:
        grouped = full_df.groupby("video_id")
    else:
        return video_dfs
    for name, sub in grouped:
        safe_name = str(name).lower().replace(" ", "_").replace("/", "_")
        sub.to_csv(output_dir / f"{safe_name}_emotion_labelled.csv", index=False)
        video_dfs[str(name)] = sub.copy()
    return video_dfs


def build_table3(video_dfs: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    rows = []
    for name, sub in video_dfs.items():
        monthly = monthly_emotion_vectors(sub, "published_at", "emotion_label", EMOTION_ORDER)
        perm = permutation_test_lag1(monthly, n_permutations=cfg.n_permutations, random_state=cfg.seed)
        rows.append({
            "Video": name,
            "Mean lag-1 cosine": safe_round(perm["observed"], 4),
            "SD (lag-1 cosine)": safe_round(sd_lag_cosine(monthly, lag=1), 4),
            "Null mean": safe_round(perm["null_mean"], 4),
            "Null test (p)": perm["p_report"],
        })
    return pd.DataFrame(rows)


def build_table_a1(video_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, sub in video_dfs.items():
        monthly = monthly_emotion_vectors(sub, "published_at", "emotion_label", EMOTION_ORDER)
        rows.append({
            "Video": name,
            "Lag-2 cosine": safe_round(mean_lag_cosine(monthly, lag=2), 6),
            "Lag-3 cosine": safe_round(mean_lag_cosine(monthly, lag=3), 6),
            "Lag-6 cosine": safe_round(mean_lag_cosine(monthly, lag=6), 6),
        })
    return pd.DataFrame(rows)


def build_table_a2(gold_df: pd.DataFrame, model, tokenizer, device: torch.device, batch_size: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds_emoji = predict_batches(model, tokenizer, gold_df["text"].tolist(), device, batch_size)
    preds_noemoji = predict_batches(model, tokenizer, gold_df["text_noemoji"].tolist(), device, batch_size)
    gold_eval = gold_df[["label_id"]].copy().rename(columns={"label_id": "gold_label"})
    gold_eval["pred_emoji"] = preds_emoji
    gold_eval["pred_noemoji"] = preds_noemoji
    gold_eval.to_csv(output_dir / "gold_eval_predictions.csv", index=False)
    table = pd.DataFrame([
        {
            "Input condition": "Emoji-aware input",
            "Macro-F1": round(f1_score(gold_eval["gold_label"], gold_eval["pred_emoji"], average="macro"), 4),
            "Accuracy": round(accuracy_score(gold_eval["gold_label"], gold_eval["pred_emoji"]), 4),
        },
        {
            "Input condition": "Emoji-stripped input",
            "Macro-F1": round(f1_score(gold_eval["gold_label"], gold_eval["pred_noemoji"], average="macro"), 4),
            "Accuracy": round(accuracy_score(gold_eval["gold_label"], gold_eval["pred_noemoji"]), 4),
        },
    ])
    return table, gold_eval


def build_training_robustness(gold_df: pd.DataFrame, cfg: Config, device: torch.device, full_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # with synthetic
    cfg_with = Config(**asdict(cfg))
    cfg_with.model_dir = str(Path(cfg.output_dir) / "model_with_synth")
    cfg_with.train_with_synth = True
    _, tok_with, model_with, _, _ = load_or_train_model(gold_df, cfg_with, device)

    # no synthetic
    cfg_no = Config(**asdict(cfg))
    cfg_no.model_dir = str(Path(cfg.output_dir) / "model_no_synth")
    cfg_no.train_with_synth = False
    _, tok_no, model_no, _, _ = load_or_train_model(gold_df, cfg_no, device)

    gold_with = pd.DataFrame({
        "gold_label": gold_df["label_id"].to_numpy(),
        "pred_label": predict_batches(model_with, tok_with, gold_df["text"].tolist(), device, cfg.batch_size),
    })
    gold_no = pd.DataFrame({
        "gold_label": gold_df["label_id"].to_numpy(),
        "pred_label": predict_batches(model_no, tok_no, gold_df["text"].tolist(), device, cfg.batch_size),
    })
    gold_with.to_csv(output_dir / "gold_pred_with_synth.csv", index=False)
    gold_no.to_csv(output_dir / "gold_pred_no_synth.csv", index=False)

    full_with = full_df.copy()
    full_no = full_df.copy()
    full_with["emotion_label"] = [ID_TO_LABEL[int(x)] for x in predict_batches(model_with, tok_with, full_with["text_emoji"].tolist(), device, cfg.batch_size)]
    full_no["emotion_label"] = [ID_TO_LABEL[int(x)] for x in predict_batches(model_no, tok_no, full_no["text_emoji"].tolist(), device, cfg.batch_size)]
    full_with.to_csv(output_dir / "full_pred_with_synth.csv", index=False)
    full_no.to_csv(output_dir / "full_pred_no_synth.csv", index=False)

    video_dfs_with = save_video_splits(full_with, output_dir / "with_synth_videos")
    video_dfs_no = save_video_splits(full_no, output_dir / "no_synth_videos")

    macro_f1_with = round(f1_score(gold_with["gold_label"], gold_with["pred_label"], average="macro"), 4)
    macro_f1_no = round(f1_score(gold_no["gold_label"], gold_no["pred_label"], average="macro"), 4)
    acc_with = round(accuracy_score(gold_with["gold_label"], gold_with["pred_label"]), 4)
    acc_no = round(accuracy_score(gold_no["gold_label"], gold_no["pred_label"]), 4)

    rows_a3 = []
    common_keys = sorted(set(video_dfs_with.keys()) & set(video_dfs_no.keys()))
    for name in common_keys:
        monthly_with = monthly_emotion_vectors(video_dfs_with[name], "published_at", "emotion_label", EMOTION_ORDER)
        monthly_no = monthly_emotion_vectors(video_dfs_no[name], "published_at", "emotion_label", EMOTION_ORDER)
        perm_with = permutation_test_lag1(monthly_with, cfg.n_permutations, cfg.seed)
        perm_no = permutation_test_lag1(monthly_no, cfg.n_permutations, cfg.seed)
        rows_a3.append({
            "Video": name,
            "Training configuration": "With synthetic augmentation",
            "Macro-F1 (gold)": macro_f1_with,
            "Mean lag-1 cosine": safe_round(perm_with["observed"], 4),
            "Null mean": safe_round(perm_with["null_mean"], 4),
            "Null test (p)": perm_with["p_report"],
            "Conclusion": "Persistence supported",
        })
        rows_a3.append({
            "Video": name,
            "Training configuration": "No synthetic (class-weighted)",
            "Macro-F1 (gold)": macro_f1_no,
            "Mean lag-1 cosine": safe_round(perm_no["observed"], 4),
            "Null mean": safe_round(perm_no["null_mean"], 4),
            "Null test (p)": perm_no["p_report"],
            "Conclusion": "Persistence supported",
        })
    table_a3 = pd.DataFrame(rows_a3)

    table_a4 = pd.DataFrame([
        {"Training condition": "With synthetic augmentation", "Accuracy": acc_with, "Macro-F1": macro_f1_with},
        {"Training condition": "No synthetic (class-weighted)", "Accuracy": acc_no, "Macro-F1": macro_f1_no},
    ])

    return table_a3, table_a4


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--spam-data", required=True)
    parser.add_argument("--gold-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-features-spam", type=int, default=5000)
    parser.add_argument("--train-with-synth", action="store_true")
    parser.add_argument("--synth-fear", type=int, default=74)
    parser.add_argument("--synth-anger-disgust", type=int, default=100)
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    setup_logging(output_dir)
    set_seed(cfg.seed)
    device = get_device()
    logging.info("Using device: %s", device)
    save_json(asdict(cfg), output_dir / "run_config.json")

    # Load raw data
    raw_df = pd.read_csv(cfg.raw_data)
    ensure_columns(raw_df, ["comment_text", "published_at"], "comments_raw")
    raw_df = raw_df.copy()
    raw_df["comment_text"] = raw_df["comment_text"].astype(str).apply(clean_text_basic)
    raw_df["published_at"] = pd.to_datetime(raw_df["published_at"], errors="coerce")
    raw_df = raw_df.dropna(subset=["published_at"])
    raw_df.to_csv(output_dir / "raw_loaded.csv", index=False)

    # Language filter
    tqdm.pandas(desc="Language filter")
    raw_df["is_en"] = raw_df["comment_text"].progress_apply(is_english)
    df = raw_df[raw_df["is_en"]].copy()
    df.to_csv(output_dir / "english_only.csv", index=False)
    logging.info("After language filter: %d rows", len(df))

    # Spam filter
    spam_df = pd.read_csv(cfg.spam_data)
    ensure_columns(spam_df, ["text", "label"], "youtube_spam_dataset")
    spam_df = spam_df.copy()
    spam_df["text"] = spam_df["text"].astype(str).apply(clean_text_basic)
    vectorizer = TfidfVectorizer(max_features=cfg.max_features_spam)
    X_spam = vectorizer.fit_transform(spam_df["text"])
    y_spam = spam_df["label"]
    rf = RandomForestClassifier(n_estimators=100, random_state=cfg.seed)
    rf.fit(X_spam, y_spam)
    X_main = vectorizer.transform(df["comment_text"])
    df["spam_pred"] = rf.predict(X_main)
    df = df[df["spam_pred"] == 0].copy()
    logging.info("After spam filter: %d rows", len(df))

    # Emoji-aware and stripped text
    df["text_emoji"] = df["comment_text"].apply(demojize_text)
    df["text_noemoji"] = df["comment_text"].apply(remove_emoji_text)

    # Gold data and model
    gold = prepare_gold(Path(cfg.gold_data))
    save_json({"gold_size": int(len(gold))}, output_dir / "gold_summary.json")
    _, tokenizer, model, train_df, val_df = load_or_train_model(gold, cfg, device)

    # Gold evaluation
    y_true = gold["label_id"].to_numpy()
    y_pred_emoji = predict_batches(model, tokenizer, gold["text"].tolist(), device, batch_size=cfg.batch_size)
    y_pred_noemoji = predict_batches(model, tokenizer, gold["text_noemoji"].tolist(), device, batch_size=cfg.batch_size)
    gold_eval = pd.DataFrame({
        "gold_label": y_true,
        "pred_emoji": y_pred_emoji,
        "pred_noemoji": y_pred_noemoji,
    })
    gold_eval.to_csv(output_dir / "gold_eval_predictions.csv", index=False)

    # Full inference
    preds_emoji = predict_batches(model, tokenizer, df["text_emoji"].tolist(), device, batch_size=cfg.batch_size)
    preds_noemoji = predict_batches(model, tokenizer, df["text_noemoji"].tolist(), device, batch_size=cfg.batch_size)
    df["emotion"] = preds_emoji
    df["emotion_label"] = [ID_TO_LABEL[int(x)] for x in preds_emoji]
    df["emotion_noemoji"] = preds_noemoji
    df["emotion_noemoji_label"] = [ID_TO_LABEL[int(x)] for x in preds_noemoji]
    df["month"] = df["published_at"].dt.to_period("M").astype(str)
    df.to_csv(output_dir / "analytic_emotion_labelled.csv", index=False)

    # Monthly vectors all data
    monthly_norm = monthly_emotion_vectors(df, "published_at", "emotion_label", EMOTION_ORDER)
    monthly_norm.to_csv(output_dir / "monthly_emotion_vectors.csv")
    lag1_values = [cosine_between_rows(monthly_norm.to_numpy()[i], monthly_norm.to_numpy()[i + 1]) for i in range(len(monthly_norm) - 1)]
    pd.DataFrame({"lag1_cosine": lag1_values}).to_csv(output_dir / "lag1_cosine.csv", index=False)

    # Video-specific outputs
    video_dfs = save_video_splits(df, output_dir)
    if not video_dfs:
        video_dfs = {"All data": df.copy()}

    # Figures
    plot_temporal_persistence(monthly_norm.rename(columns={c: c.title() for c in monthly_norm.columns}), output_dir / "figure_5_temporal_persistence.png", "Temporal Persistence of Collective Emotion (Monthly)")
    plot_engagement_vs_emotion(df, monthly_norm, output_dir / "figure_engagement_vs_emotion.png", "Emotion Configuration and Engagement Dynamics")
    plot_bar(
        [round(f1_score(y_true, y_pred_emoji, average="macro"), 4), round(f1_score(y_true, y_pred_noemoji, average="macro"), 4)],
        ["Emoji-aware", "Emoji-stripped"],
        output_dir / "figure_6_emoji_performance.png",
        "Emotion Classification Performance (Emoji vs Stripped)",
        "Macro-F1 Score",
    )
    plot_bootstrap_distribution(gold, model, tokenizer, device, cfg.batch_size, output_dir / "figure_A1_bootstrap.png")
    plot_learning_curve(gold, model, tokenizer, device, cfg.batch_size, output_dir / "figure_A2_learning_curve.png", cfg.seed)

    # Main tables
    table_3 = build_table3(video_dfs, cfg)
    table_a1 = build_table_a1(video_dfs)
    table_a2, _ = build_table_a2(gold, model, tokenizer, device, cfg.batch_size, output_dir)
    table_a3, table_a4 = build_training_robustness(gold, cfg, device, df, output_dir)

    table_3.to_csv(output_dir / "Table_3_temporal_persistence.csv", index=False)
    table_a1.to_csv(output_dir / "Appendix_Table_A1_lagged_cosines.csv", index=False)
    table_a2.to_csv(output_dir / "Appendix_Table_A2_emoji_sensitivity.csv", index=False)
    table_a3.to_csv(output_dir / "Appendix_Table_A3_training_invariance.csv", index=False)
    table_a4.to_csv(output_dir / "Appendix_Table_A4_training_robustness.csv", index=False)

    with pd.ExcelWriter(output_dir / "all_manuscript_tables.xlsx", engine="openpyxl") as writer:
        table_3.to_excel(writer, sheet_name="Table_3", index=False)
        table_a1.to_excel(writer, sheet_name="Table_A1", index=False)
        table_a2.to_excel(writer, sheet_name="Table_A2", index=False)
        table_a3.to_excel(writer, sheet_name="Table_A3", index=False)
        table_a4.to_excel(writer, sheet_name="Table_A4", index=False)

    # Summary metadata
    run_summary = {
        "device": str(device),
        "n_raw": int(len(raw_df)),
        "n_english": int(raw_df["is_en"].sum()),
        "n_analytic": int(len(df)),
        "macro_f1_emoji": round(f1_score(y_true, y_pred_emoji, average="macro"), 4),
        "accuracy_emoji": round(accuracy_score(y_true, y_pred_emoji), 4),
        "macro_f1_noemoji": round(f1_score(y_true, y_pred_noemoji, average="macro"), 4),
        "accuracy_noemoji": round(accuracy_score(y_true, y_pred_noemoji), 4),
        "table3_rows": table_3.to_dict(orient="records"),
    }
    save_json(run_summary, output_dir / "run_summary.json")
    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
