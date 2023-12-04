FROM python:3.9
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install numpy pandas torch===2.1.0 openai-whisper transformers fpdf 

LABEL Maintainer="ermkeg"

WORKDIR /usr/app/src