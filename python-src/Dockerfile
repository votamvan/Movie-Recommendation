FROM python:3.13-slim

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY ./app /app/app
COPY ./data /app/data
ENTRYPOINT [ "python" ]
CMD [ "app/main.py" ]