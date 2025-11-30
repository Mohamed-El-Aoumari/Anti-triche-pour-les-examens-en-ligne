# python:3.10
FROM python:3.10.19-slim

WORKDIR /app
COPY ./ ./

RUN pip install -r requirements.txt
CMD ["python", "main.py"]
