from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pred_crypto_memes import process_pipeline

with DAG('crypto_monitoring', start_date=datetime(2024, 1, 1), schedule_interval='@hourly') as dag:
    monitor_task = PythonOperator(
        task_id='monitor_social_media',
        python_callable=process_pipeline
    )