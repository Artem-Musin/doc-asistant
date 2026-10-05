"""Конфигурация проекта: пути и загрузка ключей из .env"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Корень проекта
ROOT = Path(__file__).resolve().parent.parent

# Пути
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
MODELS_DIR.mkdir(exist_ok=True)

# Файлы
CSV_PATH = DATA_DIR / "cooking_clean.csv"
MODEL_PATH = MODELS_DIR / "best_model.pth"
TFIDF_PATH = MODELS_DIR / "tfidf.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LORA_DIR = MODELS_DIR / "lora-doc-assistant"

# Ключи API
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Предупреждения
if not OLLAMA_API_KEY:
    print("OLLAMA_API_KEY не найден в .env")
if not HF_TOKEN:
    print("HF_TOKEN не найден в .env")

# URL и модель Ollama
OLLAMA_BASE_URL = "https://ollama.com/v1"
OLLAMA_MODEL = "gpt-oss:120b"

# Модель для LoRA
LORA_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Гиперпараметры классификатора
TFIDF_MAX_FEATURES = 1000
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 10
LR = 1e-3

# System prompt для LLM
SYSTEM_PROMPT = """Ты честный ассистент. Отвечай только валидным JSON.

Структура ответа:
{
  "answer": "твой ответ или 'Я не знаю'",
  "confidence": 0-100,
  "reason": "почему ты уверен или не уверен"
}

Правила:
1. Если не знаешь ответ, пиши в answer: "Я не знаю".
2. Если не уверен, пиши: "Я не уверен, но предполагаю: ..." и ставь confidence ниже 50.
3. Не выдумывай факты, даты, имена, цитаты, события.
4. Если вопрос о событиях после даты твоего обучения, скажи что не знаешь.
5. Если вопрос бессмысленный, так и скажи.
6. Не добавляй текст вне JSON."""


if __name__ == "__main__":
    print("ROOT:", ROOT)
    print("CSV:", CSV_PATH, "| существует:", CSV_PATH.exists())
    print("MODELS:", MODELS_DIR, "| существует:", MODELS_DIR.exists())
    print("OLLAMA key:", "OK" if OLLAMA_API_KEY else "НЕТ")
    print("HF token:", "OK" if HF_TOKEN else "НЕТ")