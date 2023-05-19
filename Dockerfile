FROM python:3

WORKDIR /app
COPY . .

RUN set -xe \
    && apt-get update -y \
    && apt-get install -y openjdk-11-jdk \
    && apt-get install -y libenchant-2-dev \
    && apt-get install -y unzip \
    && pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

ADD https://languagetool.org/download/LanguageTool-stable.zip /tmp/LanguageTool-stable.zip
RUN set -xe \
    && mkdir -p /tmp/language_tool \
    && unzip /tmp/LanguageTool-stable.zip -d /tmp/language_tool \
    && mkdir /root/.cache/language_tool_python \
    && mv /tmp/language_tool/LanguageTool-6.1 /root/.cache/language_tool_python

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk
ENV PATH $JAVA_HOME/bin:$PATH

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:443", "src:app"]
