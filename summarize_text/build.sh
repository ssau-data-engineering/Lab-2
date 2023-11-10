#!/bin/bash

image_name=ssau-lab2-text2text-inference

docker build \
 --tag ${image_name} \
    .