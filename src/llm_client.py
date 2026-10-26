"""Клиент к Ollama Cloud с structured output (JSON)."""
import json
from openai import OpenAI
from config import (
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
)

if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY не найден. Проверьте .env")

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


def ask_strict(question: str, model: str = OLLAMA_MODEL) -> dict:
    """
    Задать вопрос и получить структурированный ответ.
    Возвращает dict с полями:
      - data: {answer, confidence, reason}
      - usage: {prompt, completion, total}
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "answer": content,
            "confidence": 0,
            "reason": "JSON не распарсен",
        }

    return {
        "data": data,
        "usage": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
    }


if __name__ == "__main__":
    result = ask_strict("рецепт печенья")
    d = result["data"]

    print("Ответ:", d["answer"])
    print("Уверенность:", d["confidence"])
    print("Причина:", d["reason"])

    if d["confidence"] < 30:
        print("Модель не уверена в ответе.")

    print()
    print("Токены:", result["usage"])