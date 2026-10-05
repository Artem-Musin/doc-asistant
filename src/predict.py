"""Предсказание категории для нового текста."""
import pickle
import torch

from src.config import MODEL_PATH, TFIDF_PATH, ENCODER_PATH
from src.model import TextClassifier


def _load_artifacts():
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return vectorizer, encoder


def predict(title: str, instructions: str = "") -> str:
    vectorizer, encoder = _load_artifacts()
    text = f"{title} {instructions}"
    X = vectorizer.transform([text]).toarray()

    model = TextClassifier(X.shape[1], len(encoder.classes_))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        pred = torch.argmax(logits, dim=1).item()

    return encoder.inverse_transform([pred])[0]


if __name__ == "__main__":
    print(predict("Chocolate cake with berries"))