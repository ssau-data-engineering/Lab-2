from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount  # Добавленный импорт
from datetime import datetime, timedelta

# Определите параметры DAG
default_args = {
    'owner': 'bogdann',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'air_ml_bn',
    default_args=default_args,
    description='A simple DAG with a ml',
    schedule_interval=None,
)

read_data = DockerOperator(
    task_id='read_data',
    image= 'bogdann63/fian_sens:fian_sens',
    command= 'python /data/scripts/read_data.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

train_model = DockerOperator(
    task_id='train_model',
    image='bogdann63/fian_sens:fian_sens',
    command='python /data/scripts/training_model.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

read_data >> train_model


if __name__ == "__main__":
    dag.cli()
