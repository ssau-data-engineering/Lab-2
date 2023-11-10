FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN mkdir /wd

RUN mkdir /wd/reports
RUN mkdir /wd/texts
RUN mkdir /wd/model
RUN mkdir /wd/scripts

RUN apt update; apt install vim -y

RUN update-alternatives --install /usr/bin/python python /opt/conda/bin/python 1
RUN update-alternatives --install /usr/bin/pip pip /opt/conda/bin/pip 1

RUN pip install --upgrade pip

WORKDIR /wd

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib/python3.10/site-packages/torch/lib"