from datetime import datetime
from airflow import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'mnist',
    default_args=default_args,
    description='mnist',
    schedule_interval=None,
)

read_train_data = DockerOperator(
    task_id='read_train_data',
    image='sasha151299/second_pipeline:1.0',
    command='python /data/read_train_data.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

read_train_data