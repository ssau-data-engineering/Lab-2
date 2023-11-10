#!/bin/bash

text_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/texts/
report_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/reports/
model_path=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/model/summarize_text/
container_name=ssau-lab2-text2text-inference

running_image_name=ssau-lab2-text2text-inference

docker run \
    -it \
    --rm \
    --gpus "device=0" \
    -v ${text_path}:"/wd/texts/" \
    -v ${report_path}:"/wd/reports/" \
    -v ${model_path}:"/wd/model/" \
    -v "/home/anteii/projects/ssau-data-engineering/lab2/summarize_text/wd/scripts/":"/wd/scripts/" \
    --name ${container_name} \
    ${running_image_name} \
    python scripts/main.py