FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 datafetcher && chown -R datafetcher:datafetcher /app
USER datafetcher

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command
CMD ["python", "fetch_binance_ohlc.py"]