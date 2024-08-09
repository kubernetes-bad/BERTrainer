FROM python:3.11

WORKDIR /app

COPY . .
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

ENTRYPOINT ["python", "-m", "bertrainer.train"]
