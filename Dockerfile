FROM python:3

WORKDIR /app
COPY . .

RUN set -xe \
    && apt-get update -y \
    && apt-get install -y libenchant-2-dev \
    && pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "src:app"]
