from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator, Mount
from docker.types import DeviceRequest
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator


default_args = {
    'owner': 'admin',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
    'dagrun_timeout': timedelta(0, 0, 0, 0, 15, 0, 0)
}

dag = DAG(
    'data_engineering_lab_2_train_model',
    default_args=default_args,
    description='DAG for data engineering lab 2: training a neural network',
    schedule_interval=None,
)
# пушу в регистри
# пишу такой докер оператор
# профит
# разобраться как включить гпу

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=5,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/lab2/train_data',  # Target folder to monitor
    fs_conn_id='default_conn_id',
    dag=dag,
)


train_model = DockerOperator(
    task_id='train_model',
    container_name = "ssau-lab2-model-train",
    image='anteii/ssau-lab2-train-model',
    command='python scripts/main.py',
    mounts=[
        # /data/
        Mount(source='/data/lab2/train_data/',                  target='/wd/data/',     type='bind'),
        Mount(source='/data/lab2/model/classify_text/',         target='/wd/model/',    type='bind'),
        Mount(source='/data/lab2/train_results/',               target='/wd/results/',  type='bind'),
        Mount(source='/data/lab2/train_model/wd/scripts/',      target='/wd/scripts/',  type='bind'),
        ],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
    auto_remove="force",
    #device_requests=[DeviceRequest(device_ids=["0"], capabilities=[['gpu']])]
)


wait_for_new_file >> train_model

if __name__ == "__main__":
    dag.test()