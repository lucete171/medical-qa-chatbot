FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Python 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
 && rm -rf /var/lib/apt/lists/*

# requirements
COPY requirements.txt ./requirements.txt
COPY fastapi/requirements.txt ./fastapi_requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r fastapi_requirements.txt

# PyTorch GPU (CUDA 12.1)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 코드 복사
COPY model ./model
COPY fastapi/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
