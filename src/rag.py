import faiss
import json
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys

# Добавляем путь, чтобы импортировать llm_client
sys.path.append(str(Path(__file__).parent))
import llm_client
from openai import OpenAI

ROOT = Path(__file__).parent.parent
INDEX_PATH = ROOT / "models" / "faiss_recipes_l2.index"
META_PATH = ROOT / "models" / "faiss_recipes_l2_meta.json"
CSV_PATH = ROOT / "data" / "cooking_clean.csv"


def chunk_text(text, size=500, overlap=50):
    text = str(text).strip()
    if not text: return []
    chunks, start, step = [], 0, size - overlap
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text): break
        start += step
    return chunks


def build_index():
    """Загружает индекс и энкодер один раз при старте"""
    print("Loading index and encoder...")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunk_meta = json.load(f)["meta"]

    # Восстанавливаем тексты чанков
    df = pd.read_csv(CSV_PATH, sep=";")
    df["full_text"] = "Recipe: " + df["title"].astype(str) + "\nInstructions: " + df["instructions"].astype(str)
    chunks = []
    for _, row in df.iterrows():
        chunks.extend(chunk_text(row["full_text"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)

    client = OpenAI(base_url=llm_client.OLLAMA_BASE_URL, api_key=llm_client.OLLAMA_API_KEY)

    return {
        "index": index, "chunk_meta": chunk_meta, "chunks": chunks,
        "encoder": encoder, "client": client
    }


RAG_SYSTEM = """Ты повар-ассистент. Отвечай кратко, только фактами из контекста. Если нет - пиши 'Нет данных'."""


def rag_answer(question, state, k=10):
    index = state["index"]
    encoder = state["encoder"]
    client = state["client"]

    # Поиск
    q_emb = encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, k)

    context_parts = []
    sources = []
    for i, idx in enumerate(indices[0]):
        meta = state["chunk_meta"][idx]
        text = state["chunks"][idx]
        context_parts.append(f"[{i + 1}] {meta['recipe_title']}\n{text}")
        if meta['recipe_title'] not in sources:
            sources.append(meta['recipe_title'])

    context = "\n\n".join(context_parts)

    # Генерация
    response = client.chat.completions.create(
        model=llm_client.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
        ],
        temperature=0,
    )
    return response.choices[0].message.content, sources