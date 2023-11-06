#!/bin/bash

image_name=ssau-lab-video2audio

docker build \
 --tag ${image_name} \
 .