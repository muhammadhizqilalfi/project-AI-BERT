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
# 2. CLEANING FUNCTION
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
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # Keep letters/numbers/spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Clean spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================================
# 3. MAIN
# ==========================================================
def main():

    print(f"Membaca dataset dari: {DATA_INPUT_PATH}")

    # ==================================================
    # Attempt load with TAB delimiter (sesuai dataset Anda!)
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

    print("Dataset Loaded!")
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

    # Jika dataset Anda bentuk kolomnya seperti:
    # Date | User | Tweet | sentiment
    # maka ini pasti terdeteksi

    if text_col is None:
        raise ValueError(f"Tidak ditemukan kolom text! Kolom tersedia: {df.columns}")

    if label_col is None:
        raise ValueError(f"Tidak ditemukan kolom label! Kolom tersedia: {df.columns}")

    print(f"Kolom text  terdeteksi: {text_col}")
    print(f"Kolom label terdeteksi: {label_col}")

    # ==========================================================
    # CLEANING
    # ==========================================================
    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    # ==========================================================
    # TRAIN - TEST SPLIT
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
    train_df.to_csv(os.path.join(DATA_OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_OUTPUT_DIR, "test.csv"), index=False)

    print("Preprocessing selesai!")
    print("Train disimpan di :", os.path.join(DATA_OUTPUT_DIR, "train.csv"))
    print("Test  disimpan di :", os.path.join(DATA_OUTPUT_DIR, "test.csv"))


if __name__ == "__main__":
    main()
