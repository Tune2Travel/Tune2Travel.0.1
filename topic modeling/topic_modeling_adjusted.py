import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# -------- CONFIG --------
BASE_INPUT_DIR = r"/home/ccipfm/despa/depsa_topic"  # Adjusted to the unzipped directory
BASE_OUTPUT_DIR = os.path.join(BASE_INPUT_DIR, "topic_modeling_output") # Adjusted output directory
YEARS = [2021, 2022, 2023, 2024, 2025]
MIN_TOPIC_SIZES = [5, 10, 25, 50]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# ------------------------

# Load embedding model (auto-uses GPU if available)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
tqdm.pandas()

for year in YEARS:
    input_path = os.path.join(BASE_INPUT_DIR, f"cleaned_no_emo_despa_{year}.csv") # Adjusted file name
    if not os.path.exists(input_path):
        print(f"[WARNING] Missing file for {year}: {input_path}")
        continue

    print(f"\n[INFO] Loading data for {year}...")
    df = pd.read_csv(input_path)
    if "cleaned_text" not in df.columns:
        print(f"[WARNING] 'cleaned_text' column not found in {input_path}")
        continue
    
    texts = df["cleaned_text"].astype(str).dropna().tolist()

    for min_size in MIN_TOPIC_SIZES:
        print(f"[INFO] Running BERTopic for year {year}, min_topic_size={min_size}...")

        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=min_size,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(texts)

        output_dir = os.path.join(BASE_OUTPUT_DIR, f"clean_granularity_{year}", f"min_topic_size_{min_size}_{year}")
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        df_result = pd.DataFrame({
            "Document": texts,
            "Topic": topics,
            "Probability": probs
        })
        df_result.to_csv(os.path.join(output_dir, f"bertopic_results_{min_size}s_{year}.csv"), index=False)

        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(os.path.join(output_dir, f"bertopic_topics_{min_size}s_{year}.csv"), index=False)

        print(f"[DONE] Saved results in {output_dir}")

print("\n🎯 All years processed across all topic granularities.") 