#!/usr/bin/env python3
"""
Bitcoin Data Pipeline DAG
Orchestrates the fetching and processing of cryptocurrency data from Binance
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

# Add data_fetcher to Python path
sys.path.append('/opt/airflow/data_fetcher')

try:
    from fetch_binance_ohlc import BinanceDataFetcher
except ImportError:
    # Fallback for development/testing
    BinanceDataFetcher = None

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
    'max_active_runs': 1,
}

# DAG configuration
dag = DAG(
    'btc_data_pipeline',
    default_args=default_args,
    description='Bitcoin and cryptocurrency data pipeline',
    schedule_interval=timedelta(hours=1),  # Run every hour
    tags=['bitcoin', 'crypto', 'data-pipeline'],
    max_active_tasks=10,
    max_active_runs=1,
)

# Configuration variables
SYMBOLS = Variable.get("crypto_symbols", default_var="BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT", deserialize_json=False).split(',')
TIMEFRAMES = Variable.get("crypto_timeframes", default_var="1h,4h,1d", deserialize_json=False).split(',')
POSTGRES_CONN_ID = Variable.get("postgres_conn_id", default_var="postgres_default")

def check_database_connection():
    """Check if database connection is working"""
    try:
        postgres_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        connection = postgres_hook.get_conn()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result and result[0] == 1:
            print("✅ Database connection successful")
            return True
        else:
            raise Exception("Database connection test failed")
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

def fetch_crypto_data(symbol: str, timeframe: str, **context) -> Dict[str, Any]:
    """
    Fetch cryptocurrency data for a specific symbol and timeframe
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        timeframe: Time interval ('1h', '4h', '1d')
        context: Airflow context
        
    Returns:
        Dictionary with execution results
    """
    if not BinanceDataFetcher:
        raise ImportError("BinanceDataFetcher not available")
    
    print(f"🚀 Starting data fetch for {symbol} {timeframe}")
    
    # Initialize fetcher
    fetcher = BinanceDataFetcher()
    
    # Override database config for Airflow environment
    fetcher.db_config = {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'bitcoin_data'),
        'user': os.getenv('POSTGRES_USER', 'airflow'),
        'password': os.getenv('POSTGRES_PASSWORD', 'airflow')
    }
    
    try:
        # Connect to database
        if not fetcher.connect_db():
            raise Exception("Failed to connect to database")
        
        # Fetch and store data
        result = fetcher.fetch_and_store_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=1000,
            backfill_hours=24
        )
        
        print(f"✅ Data fetch completed for {symbol} {timeframe}")
        print(f"📊 Result: {result}")
        
        # Push result to XCom for downstream tasks
        context['task_instance'].xcom_push(key='fetch_result', value=result)
        
        return result
        
    except Exception as e:
        print(f"❌ Data fetch failed for {symbol} {timeframe}: {e}")
        raise
        
    finally:
        fetcher.close_db()

def validate_data_quality(symbol: str, timeframe: str, **context) -> bool:
    """
    Validate data quality for a specific symbol and timeframe
    
    Args:
        symbol: Trading pair symbol
        timeframe: Time interval
        context: Airflow context
        
    Returns:
        Boolean indicating if data quality is acceptable
    """
    try:
        postgres_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        
        # Check if we have recent data
        recent_data_query = """
            SELECT COUNT(*) as count, MAX(close_time) as latest_time
            FROM bitcoin.ohlc_data 
            WHERE symbol = %s AND timeframe = %s 
            AND close_time >= NOW() - INTERVAL '2 hours'
        """
        
        recent_data = postgres_hook.get_first(recent_data_query, parameters=(symbol, timeframe))
        
        if not recent_data or recent_data[0] == 0:
            print(f"⚠️  No recent data found for {symbol} {timeframe}")
            return False
        
        # Check for data gaps
        gap_check_query = """
            WITH time_gaps AS (
                SELECT 
                    close_time,
                    LAG(close_time) OVER (ORDER BY close_time) as prev_close_time,
                    EXTRACT(EPOCH FROM (close_time - LAG(close_time) OVER (ORDER BY close_time))) / 3600 as gap_hours
                FROM bitcoin.ohlc_data 
                WHERE symbol = %s AND timeframe = %s 
                AND close_time >= NOW() - INTERVAL '24 hours'
                ORDER BY close_time
            )
            SELECT COUNT(*) as gap_count
            FROM time_gaps 
            WHERE gap_hours > 2
        """
        
        gap_result = postgres_hook.get_first(gap_check_query, parameters=(symbol, timeframe))
        gap_count = gap_result[0] if gap_result else 0
        
        if gap_count > 0:
            print(f"⚠️  Found {gap_count} data gaps for {symbol} {timeframe}")
        
        # Check for data anomalies (price changes > 50%)
        anomaly_check_query = """
            WITH price_changes AS (
                SELECT 
                    close_time,
                    close_price,
                    LAG(close_price) OVER (ORDER BY close_time) as prev_close_price,
                    ABS((close_price - LAG(close_price) OVER (ORDER BY close_time)) / LAG(close_price) OVER (ORDER BY close_time)) as price_change_pct
                FROM bitcoin.ohlc_data 
                WHERE symbol = %s AND timeframe = %s 
                AND close_time >= NOW() - INTERVAL '24 hours'
                ORDER BY close_time
            )
            SELECT COUNT(*) as anomaly_count
            FROM price_changes 
            WHERE price_change_pct > 0.5
        """
        
        anomaly_result = postgres_hook.get_first(anomaly_check_query, parameters=(symbol, timeframe))
        anomaly_count = anomaly_result[0] if anomaly_result else 0
        
        if anomaly_count > 0:
            print(f"⚠️  Found {anomaly_count} price anomalies for {symbol} {timeframe}")
        
        # Overall quality assessment
        quality_score = 100
        if gap_count > 0:
            quality_score -= min(gap_count * 10, 50)
        if anomaly_count > 0:
            quality_score -= min(anomaly_count * 5, 30)
        
        print(f"📊 Data quality score for {symbol} {timeframe}: {quality_score}%")
        
        # Consider data quality acceptable if score >= 70%
        is_acceptable = quality_score >= 70
        
        context['task_instance'].xcom_push(key='quality_score', value=quality_score)
        context['task_instance'].xcom_push(key='is_acceptable', value=is_acceptable)
        
        return is_acceptable
        
    except Exception as e:
        print(f"❌ Data quality validation failed for {symbol} {timeframe}: {e}")
        return False

def generate_daily_summary(**context) -> Dict[str, Any]:
    """
    Generate daily summary statistics for all symbols and timeframes
    
    Returns:
        Dictionary with summary statistics
    """
    try:
        postgres_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        
        # Get summary statistics
        summary_query = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as record_count,
                MIN(close_time) as first_record,
                MAX(close_time) as last_record,
                AVG(volume) as avg_volume,
                MIN(low_price) as min_price,
                MAX(high_price) as max_price
            FROM bitcoin.ohlc_data 
            WHERE DATE(close_time) = CURRENT_DATE - INTERVAL '1 day'
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """
        
        summary_data = postgres_hook.get_records(summary_query)
        
        summary_results = []
        for row in summary_data:
            summary_results.append({
                'symbol': row[0],
                'timeframe': row[1],
                'record_count': row[2],
                'first_record': row[3].isoformat() if row[3] else None,
                'last_record': row[4].isoformat() if row[4] else None,
                'avg_volume': float(row[5]) if row[5] else 0,
                'min_price': float(row[6]) if row[6] else 0,
                'max_price': float(row[7]) if row[7] else 0,
            })
        
        print(f"📈 Generated daily summary for {len(summary_results)} symbol-timeframe combinations")
        
        # Store summary in XCom
        context['task_instance'].xcom_push(key='daily_summary', value=summary_results)
        
        return {'summary_count': len(summary_results), 'summary_data': summary_results}
        
    except Exception as e:
        print(f"❌ Daily summary generation failed: {e}")
        raise

def cleanup_old_data(**context) -> Dict[str, int]:
    """
    Clean up old data to manage storage
    
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        postgres_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        
        # Delete data older than 90 days for minute-level data
        # Keep longer history for hourly and daily data
        cleanup_queries = [
            ("1m data", "DELETE FROM bitcoin.ohlc_data WHERE timeframe = '1m' AND close_time < NOW() - INTERVAL '7 days'"),
            ("5m data", "DELETE FROM bitcoin.ohlc_data WHERE timeframe = '5m' AND close_time < NOW() - INTERVAL '30 days'"),
            ("15m data", "DELETE FROM bitcoin.ohlc_data WHERE timeframe = '15m' AND close_time < NOW() - INTERVAL '60 days'"),
            ("old logs", "DELETE FROM bitcoin.pipeline_logs WHERE created_at < NOW() - INTERVAL '30 days'"),
        ]
        
        cleanup_results = {}
        
        for description, query in cleanup_queries:
            result = postgres_hook.run(query)
            # Note: row count might not be available depending on PostgreSQL version
            cleanup_results[description] = "completed"
            print(f"🧹 Cleaned up {description}")
        
        print(f"✅ Cleanup completed for {len(cleanup_results)} categories")
        
        return cleanup_results
        
    except Exception as e:
        print(f"❌ Data cleanup failed: {e}")
        raise

# Start task
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# Database connection check
db_check_task = PythonOperator(
    task_id='check_database_connection',
    python_callable=check_database_connection,
    dag=dag,
)

# Create task groups for each symbol
symbol_task_groups = []

for symbol in SYMBOLS:
    symbol = symbol.strip()
    
    with TaskGroup(group_id=f'process_{symbol}', dag=dag) as symbol_group:
        
        # Tasks for each timeframe
        timeframe_tasks = []
        validation_tasks = []
        
        for timeframe in TIMEFRAMES:
            timeframe = timeframe.strip()
            
            # Data fetching task
            fetch_task = PythonOperator(
                task_id=f'fetch_{symbol}_{timeframe}',
                python_callable=fetch_crypto_data,
                op_kwargs={'symbol': symbol, 'timeframe': timeframe},
                dag=dag,
            )
            
            # Data validation task
            validate_task = PythonOperator(
                task_id=f'validate_{symbol}_{timeframe}',
                python_callable=validate_data_quality,
                op_kwargs={'symbol': symbol, 'timeframe': timeframe},
                dag=dag,
            )
            
            # Set up dependencies within the timeframe
            fetch_task >> validate_task
            
            timeframe_tasks.append(fetch_task)
            validation_tasks.append(validate_task)
        
        # All validations must pass before considering symbol complete
        symbol_complete = DummyOperator(
            task_id=f'{symbol}_complete',
            dag=dag,
        )
        
        validation_tasks >> symbol_complete
    
    symbol_task_groups.append(symbol_group)

# Daily summary task (runs after all symbols are processed)
daily_summary_task = PythonOperator(
    task_id='generate_daily_summary',
    python_callable=generate_daily_summary,
    dag=dag,
)

# Cleanup task
cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_data,
    dag=dag,
)

# End task
end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Set up main DAG dependencies
start_task >> db_check_task >> symbol_task_groups
symbol_task_groups >> daily_summary_task >> cleanup_task >> end_task

# Add documentation
dag.doc_md = """
# Bitcoin Data Pipeline

This DAG orchestrates the collection and processing of cryptocurrency data from Binance.

## Features:
- Fetches OHLC data for configured symbols and timeframes
- Validates data quality
- Generates daily summaries
- Cleans up old data
- Comprehensive error handling and logging

## Configuration:
- **Symbols**: Configured via Airflow Variable `crypto_symbols`
- **Timeframes**: Configured via Airflow Variable `crypto_timeframes`
- **Schedule**: Runs every hour
- **Retries**: 3 attempts with 5-minute delays

## Monitoring:
- Check pipeline logs in the database `bitcoin.pipeline_logs` table
- Monitor data quality scores in task logs
- View daily summaries in XCom data
"""