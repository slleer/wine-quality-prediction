# syntax=docker/dockerfile:1
FROM ubuntu:18.04

LABEL version="1.0"
LABEL description="Project 2 for CS 442, Author"
LABEL maintainer="Stephen Leer"

WORKDIR /sleerProject2

RUN apt-get update
RUN apt-get install -y openjdk-8-jdk

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

RUN apt-get install -y python3-pip

RUN pip3 install pandas
RUN pip3 install pyspark

COPY winequality-white.csv winequality-white.csv
COPY project2.py project2.py

CMD ["python3", "project2.py"]
