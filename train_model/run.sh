#!/bin/bash

train_data=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/train_data/
model=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/model/classify_text/
results=/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/train_results/
container_name=ssau-lab2-model-train

running_image_name=ssau-lab2-train-model

docker run \
    -it \
    --rm \
    --gpus "device=0" \
    -v ${train_data}:"/wd/data/" \
    -v ${model}:"/wd/model/" \
    -v ${results}:"/wd/results/" \
    -v "/home/anteii/projects/ssau-data-engineering/lab2/train_model/wd/scripts/":"/wd/scripts/" \
    --name ${container_name} \
    ${running_image_name} \
    python scripts/main.py