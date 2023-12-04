from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'data_engineering_lab_2',
    default_args=default_args,
    description='DAG for data engineering lab 2: training a neural network',
    schedule_interval=None,
)

load_data_train_model = DockerOperator(
    task_id='load_data_and_train_model',
    image='tensorflow/tensorflow:latest',
    command='python /data/train_model.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

# train_model = DockerOperator(
#     task_id='train_model',
#     image='train_model_image',
#     command='python train_model.py --input /prepared_data --output /trained_model',
#     mounts=[Mount(source='/data', target='/data', type='bind')],
#     docker_url="tcp://docker-proxy:2375",
#     dag=dag,
# )
load_data_train_model
# read_data >> train_model