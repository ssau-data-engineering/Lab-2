#Deriving the latest base image
FROM python:latest

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install numpy seaborn scikit-learn pandas opencv-python matplotlib remotezip tqdm einops

#Labels as key value pair
LABEL Maintainer="dim4098.me17"


# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /usr/app/src