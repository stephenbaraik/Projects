FROM apache/airflow:2.8.1

# Install extra Python packages with the airflow user
USER airflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
