#!/bin/bash

image_name=ssau-lab2-audio2text-inference

docker build \
 --tag ${image_name} \
    .