from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'btc_data_pipeline',
    default_args=default_args,
    description='Fetch BTC OHLC and store in DB',
    schedule_interval='@hourly',
)

fetch_task = BashOperator(
    task_id='fetch_binance_data',
    bash_command='python /app/fetch_binance_ohlc.py',
    dag=dag,
)

fetch_task