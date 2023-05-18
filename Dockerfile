FROM python:3

WORKDIR /app
COPY . .

RUN set -xe \
    && apt-get update -y \
    && pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "src:app"]
