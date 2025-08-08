# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
COPY models ./models
COPY logs ./logs

ENV PYTHONPATH=/app/src

EXPOSE 5000

CMD ["python", "api/app.py"]
