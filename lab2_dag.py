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

# test = DockerOperator(
#     task_id='test',
#     image='paperspace/whisper-deploy:v1.11',
#     command=['python','/data/test.py','&&','python','/data/test2.py'],
#     mounts=[Mount(source='/data', target='/data', type='bind')],
#     docker_url="tcp://docker-proxy:2375",
#     dag=dag,
# )
# test

wait_for_new_file = FileSensor(
    task_id='wait_for_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/',  # Target folder to monitor
    fs_conn_id='file_connection',
    dag=dag,
)

extract_audio = DockerOperator(
    task_id='extract_audio',
    image='ermkeg/lab2:latest',
    command='ffmpeg -i /data/input_video.mp4 -vn /data/audio.mp3 -y',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

transform_audio_to_text = DockerOperator(
    task_id='transform_audio_to_text_summarize_and_save_to_pdf',
    image='ermkeg/lab2:latest',
    command='python /data/do_the_thing.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

# summarize_text = DockerOperator(
#     task_id='',
#     image='ml_model_image',
#     command='python summarize_text.py --input text.txt --output summary.txt',
#     mounts=[Mount(source='/data', target='/data', type='bind')],
#     docker_url="tcp://docker-proxy:2375",
#     dag=dag,
# )

# save_to_pdf = DockerOperator(
#     task_id='save_to_pdf',
#     image='ml_model_image',
#     command='python save_to_pdf.py --input summary.txt --output result.pdf',
#     mounts=[Mount(source='/data', target='/data', type='bind')],
#     docker_url="tcp://docker-proxy:2375",
#     dag=dag,
# )

wait_for_new_file >> extract_audio >> transform_audio_to_text #>> summarize_text >> save_to_pdf
