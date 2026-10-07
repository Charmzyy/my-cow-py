# ---------------------------------------------------------------------------
# my-cow-py: FastAPI breed-prediction service (TensorFlow, CPU only).
# Build:  docker build -t cow-py:dev .
# Python 3.12: TF 2.21 ships wheels for 3.10-3.13, Keras 3.13 needs >= 3.11, Colab trained on 3.12.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Logs appear in `docker logs` immediately; no .pyc files in the image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Dependencies first: this slow layer stays cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Model (changes only when you retrain)
COPY cow_breed_v2.keras model_meta.json ./

# 3. Code last (changes most often)
COPY app/ ./app/

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
