#!/bin/bash

video_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/videos/
audio_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/audios/
container_name=ssau-lab-video2audio-job

running_image_name=ssau-lab-video2audio

docker run \
    -it \
    --rm \
    --gpus "device=0" \
    -v ${video_path}:"/wd/videos/" \
    -v ${audio_path}:"/wd/audios/" \
    -v "/home/anteii/projects/ssau-data-engineering/lab2/extract_audio/wd/scripts/":"/wd/scripts/" \
    --name ${container_name} \
    ${running_image_name} \
    python scripts/main.py