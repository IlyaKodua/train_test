FROM snakepacker/python:3.12 AS builder


RUN apt-get update && apt install -y python3.12-venv
RUN python3.12 -m venv /app && /app/bin/pip install -U pip

RUN mkdir -p /workspace
COPY ./src /workspace/src

WORKDIR /workspace/src

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y



COPY requirements.txt /requirements.txt
RUN /app/bin/python3.12 -m pip install -r /requirements.txt



CMD ["/app/bin/python3.12", "main.py"]
