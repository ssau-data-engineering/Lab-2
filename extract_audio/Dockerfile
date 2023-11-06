FROM python:3.11.6-alpine3.18

RUN mkdir /wd

RUN mkdir /wd/videos
RUN mkdir /wd/audios
RUN mkdir /wd/scripts

RUN apk update
RUN apk add --no-cache vim ffmpeg

RUN pip install --upgrade pip

WORKDIR /wd

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
