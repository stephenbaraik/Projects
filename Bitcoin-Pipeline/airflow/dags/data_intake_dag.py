from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.docker_operator import DockerOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'bitcoin_data_intake',
    default_args=default_args,
    schedule_interval='@daily',  # Runs daily at midnight UTC
    catchup=False
) as dag:

    fetch_binance_data = DockerOperator(
        task_id='fetch_binance_ohlc',
        image='binance_fetcher',
        api_version='auto',
        auto_remove=True,
        environment={
            'DB_HOST': 'postgres',
            'DB_NAME': 'bitcoin_db',
            'DB_USER': 'bitcoinuser',
            'DB_PASS': 'bitcoinpass'
        }
    )
