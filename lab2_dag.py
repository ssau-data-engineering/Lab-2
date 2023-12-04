from datetime import datetime
from airflow import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    'owner': 'vlados',
    'start_date': datetime(2023, 1, 3),
    'retries': 1,
}

dag = DAG(
    'audio_to_text_converter',
    default_args=default_args,
    description='DAG for extracting audio, transforming to text, summarizing, and saving as PDF',
    schedule_interval=None,
)

waiting_file = FileSensor(
    task_id='waiting_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data',  # Target folder to monitor
    fs_conn_id='fs_default',
    dag=dag,
)

audio_extraction = DockerOperator(
    task_id='audio_extraction',
    image='jrottenberg/ffmpeg',
    command='-i /data/video.mp4 -vn -acodec copy /data/audio.aac',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

text_transformation = DockerOperator(
    task_id='text_transformation',
    image='nyurik/alpine-python3-requests',
    command='python /data/text_transformation.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

text_summarizing = DockerOperator(
    task_id='text_summarizing',
    image='nyurik/alpine-python3-requests',
    command='python /data/text_summarizing.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

pdf_saving = DockerOperator(
    task_id='pdf_saving',
    image='blarney/tensorflow_learner:1.0',
    command='python /data/save_to_pdf.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

waiting_file >> audio_extraction >> text_transformation >> text_summarizing >> pdf_saving