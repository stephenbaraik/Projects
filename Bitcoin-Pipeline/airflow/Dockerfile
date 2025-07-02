FROM apache/airflow:2.7.1-python3.9

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements first for better caching
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Create directories
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins

# Copy DAGs and other files
COPY dags/ /opt/airflow/dags/

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"
ENV AIRFLOW_HOME=/opt/airflow

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"