FROM python:3.6-slim-stretch

ENV CORENLP_VERSION=2017-06-09
ENV TESSERACT_VERSION=4.1.1

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y imagemagick gcc g++ libsm6 openjdk-8-jre-headless curl unzip python3-tk
RUN apt-get install -y autoconf automake libleptonica-dev libtool libpango1.0-dev libicu-dev libcairo2-dev
RUN apt-get clean
RUN apt-get autoclean
RUN apt-get autoremove
RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 
RUN pip install --no-cache-dir spacy==2.0.10
RUN python -m spacy download en

COPY src/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -LO http://nlp.stanford.edu/software/stanford-corenlp-full-${CORENLP_VERSION}.zip
RUN unzip stanford-corenlp-full-${CORENLP_VERSION}.zip && \
	  rm stanford-corenlp-full-${CORENLP_VERSION}.zip

RUN curl -L https://github.com/tesseract-ocr/tesseract/archive/${TESSERACT_VERSION}.zip -o tesseract-${TESSERACT_VERSION}.zip
RUN unzip tesseract-${TESSERACT_VERSION}.zip && rm tesseract-${TESSERACT_VERSION}.zip

RUN cd tesseract-${TESSERACT_VERSION} && ./autogen.sh
RUN	cd tesseract-${TESSERACT_VERSION} && ./configure
RUN	make -C tesseract-${TESSERACT_VERSION}
RUN make -C tesseract-${TESSERACT_VERSION} install
RUN make -C tesseract-${TESSERACT_VERSION} training-install

RUN ldconfig
RUN rm -rf /tesseract-${TESSERACT_VERSION}

RUN cd /usr/local/share/tessdata && \
    curl -LO https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata

COPY src/anonymise.py /usr/local/bin/anonymise.py
COPY policy.xml /etc/ImageMagick-6/policy.xml

WORKDIR /docs-anonymiser
