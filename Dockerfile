FROM python:3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app
COPY . /app

RUN pip install uvicorn fastapi numpy pillow opencv-python python-multipart

RUN pip install --no-cache-dir --upgrade tensorflow[and-cuda]

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8003"]