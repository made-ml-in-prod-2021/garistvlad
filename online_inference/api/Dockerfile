# pull official base image
FROM python:3.9-slim

# set working directory
RUN mkdir -p /usr/src/api
WORKDIR /usr/src/api

# set environment variables
ENV HEART_DISEASE_S3_URL="https://madeawsbucket.s3.eu-north-1.amazonaws.com/datasets/heart.csv"

# install python dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy the current directory
COPY . .