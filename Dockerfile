FROM python:3.9.21-slim-bullseye

WORKDIR /app

COPY requirements.txt .

# For missing package
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app

# Expose the application port
EXPOSE 8080

# Define entry point (adjust to your needs)
CMD ["python", "app.py"]
