FROM python:3.11.3-slim

WORKDIR /app 

COPY experiment_requirements.txt requirements.txt

RUN buildDeps='gcc ' \
    && apt-get update \
    && apt-get install -y $buildDeps\
    && pip3 install -r requirements.txt --no-cache-dir \
    && apt-get purge -y --auto-remove $buildDeps

COPY . . 