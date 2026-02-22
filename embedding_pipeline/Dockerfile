FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY *.py .

# Cloud Run injects PORT; default to 8080
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
