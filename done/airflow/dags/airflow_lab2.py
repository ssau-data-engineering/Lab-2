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
    'audio_to_text_to_summary_to_pdf',
    default_args=default_args,
    description='DAG for extracting audio, transforming to text, summarizing, and saving as PDF',
    schedule_interval=None,
)

# TODO: Connection could be done via PythonAPI - but I didnt found HOW - so, do this in Web instead...
#file_connection = Connection(
#    conn_id="file_connection",
#    conn_type="fs",
#    description="Connection to file-path",
#)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/lab2',  # Target folder to monitor
    fs_conn_id='file_connection',
    dag=dag,
)

extract_audio = DockerOperator(
    task_id='extract_audio',
    image='jrottenberg/ffmpeg',
    docker_url="tcp://docker:2375", # For Dind usage case
    mount_tmp_dir=False,
    network_mode='bridge',
    entrypoint='bash',
    command=['-c', 'cd /data/lab2 && for single_video in ./*.mp4; do ffmpeg -y -i "${single_video}" -ss 1 -to 5 -vn "./../lab2_output/${single_video}.wav"; done'],
    mounts=[
        Mount(source='/data', target='/data', type='bind'), 
    ],
    dag=dag,
)

audo2text = DockerOperator(
    task_id='audio2text',
    image='huggy_face_image',
    docker_url="tcp://docker:2375", # For Dind usage case
    mount_tmp_dir=False,
    network_mode='bridge',
    entrypoint='bash',
    command=['-c', "python /data/audio2text.py"],
    mounts=[
        Mount(source='/data', target='/data', type='bind'), 
    ],
    dag=dag,
)

text2summary = DockerOperator(
    task_id='text2summary',
    image='huggy_face_image',
    docker_url="tcp://docker:2375", # For Dind usage case
    mount_tmp_dir=False,
    network_mode='bridge',
    entrypoint='bash',
    command=['-c', "python /data/text2summary.py"],
    mounts=[
        Mount(source='/data', target='/data', type='bind'), 
    ],
    dag=dag,
)

wait_for_new_file >> extract_audio >> audo2text >> text2summary