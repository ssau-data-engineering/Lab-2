from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount  # Добавленный импорт
from datetime import datetime, timedelta

# Определите параметры DAG
default_args = {
    'owner': 'bogdann',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'air_sens',
    default_args=default_args,
    description='A simple DAG with a FileSensor',
    schedule_interval=None,
)

# Определите путь к файлу, который вы ожидаете
file_path = '/opt/airflow/data/SPEECH.mp4'

# Создайте FileSensor, указав путь к файлу
file_sensor_task = FileSensor(
    task_id='file_sensor_task',
    filepath=file_path,
    fs_conn_id='airflow_cn',
    mode='poke',
    poke_interval=60,
    timeout=600,
    dag=dag,
)

# Задача extract_audio в том же DAG
extract_audio = DockerOperator(
    task_id='extract_audio',
    image='jrottenberg/ffmpeg',
    command='-i /data/SPEECH.mp4 -vn -acodec copy /data/output/audio.aac',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

audio_to_text = DockerOperator(
    task_id='audio_to_text',
    image='nyurik/alpine-python3-requests',
    command='python /data/scripts/audio_conv.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

text_convers = DockerOperator(
    task_id='text_convers',
    image='nyurik/alpine-python3-requests',
    command='python /data/scripts/text_conv.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

save_to_pdf = DockerOperator(
    task_id='save_to_pdf',
    image='mashupmill/text2pdf',
    command='text2pdf /data/output/summ.txt > /data/output/summ.pdf',
    mounts=[Mount(source='/data/output', target='/data/output', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

# Определите порядок выполнения задач
file_sensor_task >> extract_audio >> audio_to_text >> text_convers >> save_to_pdf

if __name__ == "__main__":
    dag.cli()
