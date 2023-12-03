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
    'translate_words_into_text',
    default_args=default_args,
    description='DAG , which allows you to translate words into text',
    schedule_interval=None,
)

monitoring = FileSensor(
    task_id='monitoring',
    poke_interval=20,
    filepath='/opt/airflow/data',  
    fs_conn_id='connection_inference',
    dag=dag,
)

converting_audio_mp4_to_aac = DockerOperator(
    task_id='converting_audio_mp4_to_aac',
    image='jrottenberg/ffmpeg',
    command='-i /data/anakin_and_obi_wan.mp4 -vn -acodec copy /data/received_video.aac',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

converting_audio_to_text = DockerOperator(
    task_id='converting_audio_to_text',
    image='nyurik/alpine-python3-requests',
    command='python /data/sound_to_txt.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

converting_text_to_pdf = DockerOperator(
    task_id='converting_text_to_pdf',
    image='nyurik/alpine-python3-requests',
    command='python /data/txt_to_pdf.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

save_to_pdf = DockerOperator(
    task_id='save_to_pdf',
    image='sasha151299/my_pdf:1.0',
    command='python /data/save_to_pdf.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

monitoring >> converting_audio_mp4_to_aac >> converting_audio_to_text >> converting_text_to_pdf >> save_to_pdf