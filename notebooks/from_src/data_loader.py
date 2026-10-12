"""Загрузка данных, категоризация и подготовка признаков (TF-IDF)."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import (
    CSV_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
)

CATEGORY_KEYWORDS = {
    "dessert": ["cake", "pie", "cookie", "dessert", "sweet", "candy", "chocolate", "brownie", "pudding"],
    "soup": ["soup", "broth", "stew", "chowder", "bisque"],
    "salad": ["salad", "slaw"],
    "meat": ["chicken", "beef", "pork", "steak", "meat", "lamb"],
    "bread": ["bread", "bun", "bagel", "muffin", "roll", "biscuit"],
}


def get_category(title: str) -> str:
    title = str(title).lower()
    for category, words in CATEGORY_KEYWORDS.items():
        if any(w in title for w in words):
            return category
    return "other"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=";")
    df.columns = ["number", "title", "instructions"]
    df["text"] = df["title"].astype(str) + " " + df["instructions"].astype(str)
    df["category"] = df["title"].apply(get_category)

    counts = df["category"].value_counts()
    valid = counts[counts >= 2].index
    df = df[df["category"].isin(valid)].reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
    X = vectorizer.fit_transform(df["text"]).toarray()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["category"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, vectorizer, encoder