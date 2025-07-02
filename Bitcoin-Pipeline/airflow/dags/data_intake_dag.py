from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='btc_data_fetch_dag',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:

    fetch_task = DockerOperator(
        task_id='fetch_btc_data',
        image='bitcoin-pipeline_fetcher',  # Confirm image name post-build
        auto_remove=True,
        environment={
            "DB_HOST": "postgres",
            "DB_NAME": "bitcoin_db",
            "DB_USER": "bitcoinuser",
            "DB_PASS": "bitcoinpass"
        }
    )
