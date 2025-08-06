FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY app/model.pkl app/model.pkl

CMD ["python", "app/main.py"]
