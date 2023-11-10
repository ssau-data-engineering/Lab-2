#!/bin/bash

image_name=ssau-lab2-train-model

docker build \
 --tag ${image_name} \
    .