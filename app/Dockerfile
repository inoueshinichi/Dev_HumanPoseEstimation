# Dockerfile for web api server

# Image
From python:3.9.17-slim

Label author="inoue.shinichi.1800@gmail.com"

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

# Pytorch 2.0
# torchvision
RUN pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

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

# Test connection to redis db
CMD ["python", "-u", "/hpe_web_api_app/web_api_srv.py"]