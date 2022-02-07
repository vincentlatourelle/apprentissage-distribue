FROM python:3.8-slim-buster

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV FLASK_APP="./RandomForest/client/app.py"
ENV FLASK_ENV="development"

# COPY . .
ENV PATH="/code/RandomForest:${PATH}"