# FSRCNN GPU 업스케일 전용 Dockerfile
# 베이스 이미지를 사용하여 빌드 시간을 단축합니다.
# 베이스 이미지: echoshot/opencv-cuda-t4:4.10.0 (Docker Hub에 사전 빌드됨)

# =========================
# 베이스 이미지 사용
# =========================
# 베이스 이미지는 OpenCV CUDA 빌드가 포함되어 있습니다.
# 베이스 이미지 빌드 방법은 scripts/build-base-image.sh 참조
ARG DOCKERHUB_USERNAME=echoshot
ARG OPENCV_VERSION=4.10.0
FROM ${DOCKERHUB_USERNAME}/opencv-cuda-t4:${OPENCV_VERSION}

# =========================
# App
# =========================
WORKDIR /app

COPY echoshot_ai_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY echoshot_ai_server/ /app/echoshot_ai_server/

# =========================
# Model weights (FSRCNN)
# =========================
RUN mkdir -p /app/weights /app/echoshot_ai_server/tasks/weights && \
    echo "Downloading FSRCNN model file..." && \
    curl -L -o /app/weights/FSRCNN_x2.pb \
      https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb && \
    cp /app/weights/FSRCNN_x2.pb /app/echoshot_ai_server/tasks/weights/ && \
    echo "FSRCNN model file downloaded successfully"

# =========================
# Non-root user
# =========================
RUN useradd -m -u 1000 worker && \
    mkdir -p /tmp/video_processing && \
    chown -R worker:worker /app /tmp/video_processing

USER worker

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TEMP_DIR=/tmp/video_processing \
    APP_ENV=prod \
    PYTHONPATH=/app

VOLUME ["/tmp/video_processing"]

ENTRYPOINT ["python", "-m", "echoshot_ai_server.main"]
