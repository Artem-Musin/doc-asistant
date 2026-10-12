"""Обучение классификатора: цикл, early stopping, сохранение артефактов."""
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from config import (
    MODEL_PATH, TFIDF_PATH, ENCODER_PATH,
    BATCH_SIZE, EPOCHS, PATIENCE, LR, RANDOM_STATE,
)
from data_loader import load_data, prepare_features
from model import TextClassifier

torch.manual_seed(RANDOM_STATE)


def train():
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer, encoder = prepare_features(df)

    num_classes = len(encoder.classes_)
    print("Классов:", num_classes)
    print("Классы:", list(encoder.classes_))
    print("Train:", len(X_train), "| Test:", len(X_test))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TextClassifier(X_train.shape[1], num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc, wait = 0, 0
    train_losses, test_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for X_b, y_b in train_loader:
            logits = model(X_b)
            loss = loss_fn(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                logits = model(X_b)
                _, pred = torch.max(logits, 1)
                correct += (pred == y_b).sum().item()
                total += y_b.size(0)
        test_acc = correct / total
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1:>3}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc, wait = test_acc, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses)
    axes[0].set_title("Loss")
    axes[0].grid(True)
    axes[1].plot(test_accs, color="red")
    axes[1].set_title("Test Accuracy")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("models/training_curves.png")
    print("Модель сохранена в", MODEL_PATH)


if __name__ == "__main__":
    train()