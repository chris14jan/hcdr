FROM python:3.8.6-buster

COPY app /app
COPY hcdr /hcdr

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
