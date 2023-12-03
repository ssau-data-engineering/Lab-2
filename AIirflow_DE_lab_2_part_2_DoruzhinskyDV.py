from datetime import datetime
from airflow import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'Airflow_MNIST_DoruzhinskyDV',
    default_args=default_args,
    description='DAG fit MNIST part 2 lab',
    schedule_interval=None,
)

fit_MNIST = DockerOperator(
    task_id='transform_audio_to_text',
    image='devashishupadhyay/scikit-learn-docker:latest',
    command='python /data/MNIST_DoruzhinskyDV.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)


fit_MNIST