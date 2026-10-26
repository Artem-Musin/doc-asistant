from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag import build_index, rag_answer

app = FastAPI(title="Doc Assistant")

# Хранилище состояния (индекс, модель)
index_state = None

@app.on_event("startup")
def load():
    global index_state
    index_state = build_index()

class Query(BaseModel):
    question: str
    k: int = 10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    answer, sources = rag_answer(q.question, index_state, k=q.k)
    return {"answer": answer, "sources": sources}