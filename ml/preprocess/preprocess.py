import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. PATH HANDLING
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_INPUT_PATH = os.path.join(ROOT_DIR, "data", "cleaned-dataset.csv")
DATA_OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2. CLEANING FUNCTION
# ==========================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "]",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==========================================================
# 3. MAIN
# ==========================================================
def main():
    print(f"\nMembaca dataset dari: {DATA_INPUT_PATH}")

    # ==========================================================
    # LOAD DATASET (AMAN UNTUK CSV RUSAK)
    # ==========================================================
    df = pd.read_csv(
        DATA_INPUT_PATH,
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python"
    )

    print("\nDataset Loaded!")
    print("Jumlah baris:", len(df))
    print("Kolom awal:", list(df.columns))

    # ==========================================================
    # FIX HEADER JIKA TERBACA SATU KOLOM
    # ==========================================================
    if len(df.columns) == 1 and "," in df.columns[0]:
        print("\n[INFO] Header terdeteksi satu kolom, memperbaiki delimiter...")

        col_name = df.columns[0]
        new_columns = col_name.split(",")

        df = df[col_name].str.split(",", n=len(new_columns)-1, expand=True)
        df.columns = new_columns

        print("Header setelah diperbaiki:", list(df.columns))

    # ==========================================================
    # DETEKSI KOLOM TEXT & LABEL
    # ==========================================================
    possible_text_columns = ["tweet", "Tweet", "text", "Text", "content", "Content"]
    possible_label_columns = ["sentiment", "Sentiment", "label", "Label"]

    text_col = next((c for c in possible_text_columns if c in df.columns), None)
    label_col = next((c for c in possible_label_columns if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError("Kolom text atau label tidak ditemukan setelah perbaikan header")

    print(f"\nKolom text  : {text_col}")
    print(f"Kolom label : {label_col}")

    # ==========================================================
    # NORMALISASI LABEL
    # ==========================================================
    print("\nNormalisasi label...")

    label_mapping = {
        "positive": 1,
        "positif": 1,
        "neutral": 0,
        "netral": 0,
        "negative": -1,
        "negatif": -1
    }

    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.lower().map(label_mapping)

    df = df[df[label_col].isin([-1, 0, 1])]

    print("Distribusi label:")
    print(df[label_col].value_counts())

    # ==========================================================
    # CLEAN TEXT
    # ==========================================================
    print("\nMembersihkan teks...")
    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    # ==========================================================
    # VALIDASI STRATIFIED SPLIT
    # ==========================================================
    if df[label_col].value_counts().min() < 2:
        raise ValueError("Data per label terlalu sedikit untuk stratified split")

    # ==========================================================
    # TRAIN TEST SPLIT
    # ==========================================================
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )

    # ==========================================================
    # SAVE OUTPUT
    # ==========================================================
    final_columns = ["clean_text", label_col]

    train_path = os.path.join(DATA_OUTPUT_DIR, "train.csv")
    test_path = os.path.join(DATA_OUTPUT_DIR, "test.csv")

    train_df[final_columns].to_csv(train_path, index=False)
    test_df[final_columns].to_csv(test_path, index=False)

    print("\nPreprocessing selesai!")
    print("Train:", train_path)
    print("Test :", test_path)

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main()
