FROM python:3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app
COPY . /app

RUN pip install uvicorn fastapi numpy pillow opencv-python tensorflow[and-cuda] python-multipart

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8003"]
