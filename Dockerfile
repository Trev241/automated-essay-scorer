FROM python:3

WORKDIR /app
COPY . .

RUN set -xe \
    && apt-get update -y \
    && apt-get install -y libenchant1c2a \
    && pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "src:app"]
