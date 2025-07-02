from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='btc_data_fetch_dag',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@hourly',
    catchup=False
) as dag:

    fetch_data = BashOperator(
        task_id='fetch_binance_data',
        bash_command='docker-compose run --rm data_fetcher'
    )

    fetch_data
