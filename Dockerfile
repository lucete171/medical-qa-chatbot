FROM python:3.10-slim

# Python 출력 버퍼링 비활성화
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY fastapi/requirements.txt ./fastapi_requirements.txt

RUN pip install -r requirements.txt \
 && pip install -r fastapi_requirements.txt

 COPY model ./model
COPY fastapi/app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]