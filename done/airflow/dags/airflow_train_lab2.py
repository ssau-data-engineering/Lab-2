import os
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}


dag = DAG(
    'train_nn',
    default_args=default_args,
    description='DAG train NN',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_train_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/lab2_nn_train/data',  # Target folder to monitor
    fs_conn_id='file_train_connection',
    dag=dag,
)

train_nn = DockerOperator(
    task_id='train_nn_on_updated_data',
    image='huggy_face_image',
    docker_url="tcp://docker:2375", # For Dind usage case
    mount_tmp_dir=False,
    network_mode='bridge',
    entrypoint='bash',
    command=['-c', "python /data/lab2_nn_train/train_nn_mnist.py"],
    mounts=[
        Mount(source='/data', target='/data', type='bind'), 
    ],
    dag=dag,
)


wait_for_new_file >> train_nn