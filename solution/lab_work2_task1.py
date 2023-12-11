from datetime import datetime
from airflow import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    'owner': 'black_viking',
    'start_date': datetime(2023, 12, 2),
}

dag = DAG(
    'convert_audiotrack_from_video_to_summary_text_in_pdf',
    default_args=default_args,
    schedule_interval=None,
)

wait_get_new_videofile = FileSensor(
    task_id='wait_get_new_vidoefile',
    poke_interval=15, 
    filepath='/opt/airflow/data',
    fs_conn_id='fs_connection_default', 
    dag=dag,
)

extract_audiotrack_from_video = DockerOperator(
    task_id='extract_audiotrack_from_video',
    image='jrottenberg/ffmpeg',
    command='-i /data/input_video.mp4 -vn -acodec copy /data/audio.aac',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

transform_audiotrack_to_text = DockerOperator(
    task_id='transform_audiotrack_to_text',
    image='nyurik/alpine-python3-requests',
    command='python /data/transform_audiotrack_to_text.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

resume_text_from_audiotrack = DockerOperator(
    task_id='resume_text_from_audiotrack',
    image='nyurik/alpine-python3-requests',
    command='python /data/resume_text_from_audiotrack.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

save_get_text_from_txt_to_pdf = DockerOperator(
    task_id='save_to_pdf',
    image='melik163/our_tensorflow_container:1.0',
    command='python /data/save_text_to_pdf_file.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

wait_get_new_videofile >> extract_audiotrack_from_video >> transform_audiotrack_to_text >> resume_text_from_audiotrack >> save_get_text_from_txt_to_pdf