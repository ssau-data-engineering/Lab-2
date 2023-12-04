from datetime import datetime
from airflow import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 12, 3),
}

dag = DAG(
    "video_to_summary_pdf",
    default_args=default_args,
    schedule_interval=None,
)

wait_video = FileSensor(
    task_id='wait_video',
    poke_interval=15, 
    filepath='/opt/airflow/data',
    fs_conn_id='connection_for_lab2', 
    dag=dag,
)

audio_from_video = DockerOperator(
    task_id='audio_from_video',
    image='jrottenberg/ffmpeg',
    command='-i /data/input_video.mp4 -vn -acodec copy /data/audio.aac',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

audio_to_text = DockerOperator(
    task_id='audio_to_text',
    image='nyurik/alpine-python3-requests',
    command='python /data/1_audio_to_text.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

text_to_summary = DockerOperator(
    task_id='text_to_summary',
    image='nyurik/alpine-python3-requests',
    command='python /data/2_text_to_summary.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

summary_to_pdf = DockerOperator(
    task_id='summary_to_pdf',
    image='vapaov/custom_container:1.0',
    command='python /data/3_summary_to_pdf.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

wait_video >> audio_from_video >> audio_to_text >> text_to_summary >> summary_to_pdf