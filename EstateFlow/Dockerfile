# Use a small, stable Python image
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and use app directory
WORKDIR /app

# System deps (optional but useful for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project into the container
COPY . .

# Create outputs dir (safe even if it exists)
RUN mkdir -p outputs

# Default command:
# 1) build/refresh DB from CSV into SQLite
# 2) run evaluation to produce outputs/*.csv
CMD ["sh", "-lc", "python scripts/db_setup.py && python scripts/evaluate.py"]
