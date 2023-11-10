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
    'data_engineering_lab_2_inference',
    default_args=default_args,
    description='DAG for data engineering lab 2: training a neural network',
    schedule_interval=None,
)
# пушу в регистри
# пишу такой докер оператор
# профит
# разобраться как включить гпу

# Wait for new videos
wait_for_new_files = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=5,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/lab2/inference_data/videos',  # Target folder to monitor
    fs_conn_id='default_conn_id',
    dag=dag,
)

# Extract audios from videos
extract_audio_task = DockerOperator(
    task_id='extract_audio_task',
    container_name = "ssau-lab-video2audio",
    image='anteii/ssau-lab-video2audio',
    command='python scripts/main.py',
    mounts=[
        Mount(source='/data/lab2/inference_data/videos/',       target='/wd/videos/',   type='bind'),
        Mount(source='/data/lab2/inference_data/audios/',       target='/wd/audios/',   type='bind'),
        Mount(source='/data/lab2/extract_audio/wd/scripts/',    target='/wd/scripts/',  type='bind'),
        ],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
    auto_remove="force",
)


# Extract texts from audios
extract_texts_task = DockerOperator(
    task_id='extract_text_task',
    container_name = "ssau-lab2-audio2text-inference",
    image='anteii/ssau-lab2-audio2text-inference',
    command='python scripts/main.py',
    mounts=[
        Mount(source='/data/lab2/inference_data/audios/',   target='/wd/audios/',   type='bind'),
        Mount(source='/data/lab2/inference_data/texts/',    target='/wd/texts/',    type='bind'),
        Mount(source='/data/lab2/model/extract_text/',      target='/wd/model/',    type='bind'),
        Mount(source='/data/lab2/extract_text/wd/scripts/', target='/wd/scripts/',  type='bind'),
        ],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
    auto_remove="force",
    #device_requests=[DeviceRequest(device_ids=["0"], capabilities=[['gpu']])]
)

# Create summaries from texts
create_report_task = DockerOperator(
    task_id='summarize_text_task',
    container_name = "ssau-lab2-text2text-inference",
    image='anteii/ssau-lab2-text2text-inference',
    command='python scripts/main.py',
    mounts=[
        Mount(source='/data/lab2/inference_data/texts/',        target='/wd/texts/',    type='bind'),
        Mount(source='/data/lab2/inference_data/reports/',      target='/wd/reports/',  type='bind'),
        Mount(source='/data/lab2/model/summarize_text/',        target='/wd/model/',    type='bind'),
        Mount(source='/data/lab2/summarize_text/wd/scripts//',  target='/wd/scripts/',  type='bind`'),
        ],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
    auto_remove="force",
    #device_requests=[DeviceRequest(device_ids=["0"], capabilities=[['gpu']])]
)


wait_for_new_file >> extract_audio_task >> extract_texts_task >> create_report_task

if __name__ == "__main__":
    dag.test()