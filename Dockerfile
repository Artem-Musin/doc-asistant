FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
ENV OLLAMA_MODEL="gpt-oss:120b"
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
