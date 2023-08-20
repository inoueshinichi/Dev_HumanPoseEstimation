# Dockerfile for web api server

# Image
From python:3.9.17-slim

Label author="inoue.shinichi.1800@gmail.com"

RUN apt-get update && apt-get upgrade -y
# OpenGL? https://cocoinit23.com/docker-opencv-importerror-libgl-so-1-cannot-open-shared-object-file/
RUN apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0

# Flask 2.3.2 (Python >= 3.8)
# Sanic 23.3.0
# Redis 3.3.8
RUN pip install \
    flask==2.3.2 \
    sanic==23.3.0 \
    redis==3.3.8 \
    numpy==1.25.1 \
    scipy==1.11.1 \
    pillow==10.0.0

# Pytorch 2.0.0
# torchvision 0.15
RUN pip install \
    torch==2.0.0 \
    torchvision==0.15 \
    --index-url https://download.pytorch.org/whl/cpu

# NOT install full Tensorflow 2.13.0 framework, only install packages for Tensorflow Lite
# TFLite Runtime 2.13.0
# TFLite Support 0.4.4
RUN pip install \
    tflite-runtime==2.13.0 \
    tflite-support==0.4.4
    
# opencv-python 4.8.0.74
RUN pip install \
    opencv-python==4.8.0.74

WORKDIR /hpe_web_api_app
COPY ./src/ /hpe_web_api_app/

ENV APP_PORT 80

# Default db conf
ENV REDIS_HOST 127.0.0.1
ENV REDIS_PORT 6379
ENV REDIS_DB 0

# Deep learning model path
ENV POSENET_MODEL /hpe_web_api_app/posenet/model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite

# Test connection to redis db
CMD ["python", "-u", "/hpe_web_api_app/web_api_srv.py"]