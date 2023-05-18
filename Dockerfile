FROM python:3

WORKDIR /app
COPY . .

RUN set -xe \
    && apt-get update -y \
    && apt-get install -y openjdk-11-jdk \
    && apt-get install -y libenchant-2-dev \
    && pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk
ENV PATH $JAVA_HOME/bin:$PATH

EXPOSE 5000

CMD ["gunicorn", "src:app"]
