#!/bin/bash

audio_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/audios/
text_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/texts/
model_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/model/
container_name=ssau-lab2-audio2text-inference

running_image_name=ssau-lab2-audio2text-inference

docker run \
    -it \
    --rm \
    --gpus "device=0" \
    -v ${audio_path}:"/wd/audios/" \
    -v ${text_path}:"/wd/texts/" \
    -v ${model_path}:"/wd/model/" \
    -v "/home/anteii/projects/ssau-data-engineering/lab2/extract_text/wd/scripts/":"/wd/scripts/" \
    --name ${container_name} \
    ${running_image_name} \
    python scripts/main.py