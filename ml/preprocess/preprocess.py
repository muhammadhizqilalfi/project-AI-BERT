import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. PATH HANDLING
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ml/preprocess/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # bert-ai-project/

DATA_INPUT_PATH = os.path.join(ROOT_DIR, "data", "INA_TweetsPPKM_Labeled_Pure.csv")
DATA_OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 2. CLEANING FUNCTION (lebih aman, tidak menghapus "!?,.")
# ==========================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www.\S+", "", text)  # URL
    text = re.sub(r"@\w+", "", text)             # Mention
    text = re.sub(r"#\w+", "", text)             # Hashtag

    # Remove emoji
    emoji_pattern = re.compile(
        "["                     # Start class
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport
        "\U0001F700-\U0001F77F"  # Alchemical
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    # JANGAN buang tanda baca penting untuk sentiment
    # KEEP: . , ! ?
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", text)

    # Clean spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================================
# 3. MAIN
# ==========================================================
def main():

    print(f"\nMembaca dataset dari: {DATA_INPUT_PATH}")

    # ==================================================
    # Load dataset dengan fallback ke koma bila TAB gagal
    # ==================================================
    try:
        df = pd.read_csv(
            DATA_INPUT_PATH,
            sep="\t",
            quotechar='"',
            on_bad_lines="skip",
            encoding="utf-8",
            engine="python"
        )
    except:
        print("Gagal load dengan TAB — mencoba koma...")
        df = pd.read_csv(
            DATA_INPUT_PATH,
            sep=",",
            quotechar='"',
            on_bad_lines="skip",
            encoding="utf-8",
            engine="python"
        )

    print("\nDataset Loaded!")
    print("Jumlah baris:", len(df))
    print("5 baris pertama:")
    print(df.head())
    print("Kolom tersedia:", list(df.columns))

    # ==========================================================
    # DETEKSI OTAOMATIS KOLOM TEXT & LABEL
    # ==========================================================
    possible_text_columns = ["tweet", "Tweet", "text", "Text", "content", "Content"]
    possible_label_columns = ["sentiment", "label", "Sentiment", "Label"]

    text_col = next((c for c in possible_text_columns if c in df.columns), None)
    label_col = next((c for c in possible_label_columns if c in df.columns), None)

    if text_col is None:
        raise ValueError(f"Tidak ditemukan kolom text! Kolom tersedia: {df.columns}")

    if label_col is None:
        raise ValueError(f"Tidak ditemukan kolom label! Kolom tersedia: {df.columns}")

    print(f"\nKolom text  terdeteksi: {text_col}")
    print(f"Kolom label terdeteksi: {label_col}")

    # ==========================================================
    # FIX LABEL YANG TERBALIK
    # ==========================================================
    print("\nDistribusi label awal:")
    print(df[label_col].value_counts())

    # Deteksi jika label terbalik (0=pos, 1=neg)
    # Cara deteksi otomatis: cek sampling
    sample = df[[text_col, label_col]].head(20)

    # Mapping standar:
    # 0 = negative, 1 = neutral, 2 = positive
    print("\nMemperbaiki label mapping ke format standar (neg=0, neu=1, pos=2)...")

    # Anda bisa mengubah ini sesuai dataset asli
    # DATASET INI DIDUGA MENGGUNAKAN:
    # 0 = positive
    # 1 = negative
    # 2 = neutral

    mapping_dataset = {
        0: "positive",
        1: "negative",
        2: "neutral"
    }

    reverse_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    df["sentiment_fixed"] = df[label_col].map(mapping_dataset).map(reverse_mapping)

    print("\n5 sampel setelah mapping:")
    print(df[[text_col, label_col, "sentiment_fixed"]].head())

    # Ganti label original dengan label_fixed
    df[label_col] = df["sentiment_fixed"]
    df.drop(columns=["sentiment_fixed"], inplace=True)

    print("\nDistribusi label setelah diperbaiki:")
    print(df[label_col].value_counts())

    # ==========================================================
    # CLEANING
    # ==========================================================
    print("\nMembersihkan teks...")
    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    # ==========================================================
    # TRAIN - TEST SPLIT
    # ==========================================================
    print("\nMembuat train-test split...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )

    # ==========================================================
    # SAVE OUTPUT
    # ==========================================================
    train_path = os.path.join(DATA_OUTPUT_DIR, "train.csv")
    test_path = os.path.join(DATA_OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nPreprocessing selesai!")
    print("Train disimpan di :", train_path)
    print("Test  disimpan di :", test_path)


if __name__ == "__main__":
    main()
