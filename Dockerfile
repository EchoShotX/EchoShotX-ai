# 멀티 스테이지 빌드를 사용한 최적화된 Dockerfile

# Stage 1: 빌드 스테이지
FROM python:3.10-slim AS builder

# 빌드 인자 정의 (기본값: CPU)
ARG BUILD_FOR=cpu
ARG CUDA_VERSION=cu121

# 빌드 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /build

# Python 의존성 복사 및 설치
COPY echoshot_ai_server/requirements.txt /build/requirements.txt

# PyTorch 및 의존성 설치 (GPU/CPU 조건부)
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$BUILD_FOR" = "gpu" ]; then \
        echo "Installing PyTorch with CUDA support..." && \
        pip install --no-cache-dir \
        torch==2.4.1+${CUDA_VERSION} \
        torchvision==0.19.1+${CUDA_VERSION} \
        torchaudio==2.4.1+${CUDA_VERSION} \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION}; \
    else \
        echo "Installing PyTorch CPU version..." && \
        pip install --no-cache-dir \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

# 나머지 의존성 설치 (BuildKit pip 캐시 활용)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: 런타임 스테이지
FROM python:3.10-slim

# 런타임 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 비root 사용자 생성
RUN useradd -m -u 1000 worker && \
    mkdir -p /app /tmp/video_processing && \
    chown -R worker:worker /app /tmp/video_processing

# 작업 디렉토리 설정
WORKDIR /app

# 빌드 스테이지에서 Python 패키지 복사
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 코드 복사
COPY echoshot_ai_server/ /app/echoshot_ai_server/

# 모델 가중치 디렉토리 생성 및 모델 파일 다운로드
RUN mkdir -p /app/weights /app/echoshot_ai_server/tasks/weights && \
    echo "Downloading model files..." && \
    curl -L -o /app/weights/FSRCNN_x2.pb \
        https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb && \
    curl -L -o /app/weights/EDSR_x2.pb \
        https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb && \
    curl -L -o /app/weights/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    cp -r /app/weights/* /app/echoshot_ai_server/tasks/weights/ && \
    echo "Model files downloaded successfully"

# 소유권 변경
RUN chown -R worker:worker /app

# 비root 사용자로 전환
USER worker

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TEMP_DIR=/tmp/video_processing \
    APP_ENV=prod

# 임시 디렉토리 설정
VOLUME ["/tmp/video_processing"]

# 애플리케이션 실행
ENTRYPOINT ["python", "-m", "echoshot_ai_server.main"]

