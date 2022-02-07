FROM python:3.8-slim-buster

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV FLASK_APP="app.py"

COPY . .
ENV PATH="/code/RandomForest:${PATH}"