FROM python:3.10-slim
# Set timeout otherwise long build will crash in pip
RUN apt update && apt install -y ffmpeg && pip install --default-timeout=100 'transformers[torch]' torchvision