from datetime import datetime
from airflow import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 12, 3),
}

dag = DAG(
    'train_mnist_model',
    default_args=default_args,
    schedule_interval=None,
)

train_model = DockerOperator(
    task_id='train_model',
    image='vapaov/custom_container:1.0',
    command='python /data/nn_mnist.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

train_model