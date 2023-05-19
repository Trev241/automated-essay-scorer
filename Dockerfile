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

ADD https://languagetool.org/download/LanguageTool-5.7.zip /tmp/LanguageTool-5.7.zip
RUN set -xe \
    && mkdir -p /tmp/language_tool \
    && unzip /tmp/LanguageTool-5.7.zip -d /tmp/language_tool \
    && mkdir -p /root/.cache/language_tool_python \
    && mv /tmp/language_tool/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk
ENV PATH $JAVA_HOME/bin:$PATH

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:443", "src:app"]
