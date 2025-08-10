FROM python:3.10-slim

WORKDIR /app

# Copy dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY src ./src
COPY models ./models
COPY logs ./logs

ENV PYTHONPATH=/app/src

EXPOSE 5000

# Start FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
