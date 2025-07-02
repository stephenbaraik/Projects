FROM apache/airflow:2.8.1

USER root
COPY requirements.txt .
RUN pip install -r requirements.txt

USER airflow
COPY airflow/dags/ /opt/airflow/dags
