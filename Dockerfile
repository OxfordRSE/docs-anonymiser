FROM python:3.6-slim-stretch

ADD src/requirements.txt requirements.txt
ADD src/anonymise.py /usr/local/bin/anonymise.py

ENV CORENLP_VERSION=2017-06-09

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y imagemagick gcc g++ libsm6 openjdk-8-jre-headless curl unzip
RUN apt-get install -y autoconf automake libtool libleptonica-dev libpango1.0-dev
RUN apt-get clean

RUN pip install --no-cache-dir spacy
RUN python -m spacy download en
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -LO http://nlp.stanford.edu/software/stanford-corenlp-full-${CORENLP_VERSION}.zip
RUN unzip stanford-corenlp-full-${CORENLP_VERSION}.zip && \
	  rm stanford-corenlp-full-${CORENLP_VERSION}.zip

RUN curl -LO https://codeload.github.com/tesseract-ocr/tesseract/zip/3.05.01
RUN unzip tesseract-3.05.01.zip && rm tesseract-3.05.01.zip
RUN cd tesseract-3.05.01 && \
    ./autogen.sh && \
  	./configure && \
  	make && \
    make install && \
RUN ldconfig
RUN rm -rf /tesseract-3.05.01

WORKDIR /docs-anonymiser
