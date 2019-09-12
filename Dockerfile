FROM python:3.7

WORKDIR /app
COPY Pipfile Pipfile.lock ./ app/modules/core/model/*.json app/modules/core/model/*.model app/modules/core/model/lexicons/
RUN pip install -U pipenv
RUN pipenv install --system
ADD . ./

# Server
STOPSIGNAL SIGINT
CMD ["python", "main.py"]
