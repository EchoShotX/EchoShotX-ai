# FSRCNN GPU 업스케일 전용 Dockerfile
# NVIDIA T4 GPU (CUDA Compute 7.5)를 위한 OpenCV CUDA 빌드

# =========================
# Build args
# =========================
ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04
ARG CUDNN=8
ARG OPENCV_VERSION=4.10.0
ARG CUDA_ARCH=7.5

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN}-devel-ubuntu${UBUNTU_VERSION}

ARG OPENCV_VERSION
ARG CUDA_ARCH
ARG DEBIAN_FRONTEND=noninteractive

ENV OPENCV_VERSION=${OPENCV_VERSION}
ENV CUDA_ARCH=${CUDA_ARCH}

# =========================
# System deps
# =========================
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libavcodec-dev \
    libavutil-dev \
    libpostproc-dev \
    libeigen3-dev \
    libgtk2.0-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python defaults + pip
# =========================
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ✅ OpenCV Python 바인딩 빌드/런타임 버전 일치 위해 pip로 numpy 설치
RUN pip install --no-cache-dir "numpy>=1.26,<2.0"

# =========================
# Build OpenCV + opencv_contrib (CUDA)
# =========================
RUN set -eux; \
    cd /opt; \
    wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip; \
    unzip -q opencv.zip; \
    rm -f opencv.zip; \
    wget -q https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib.zip; \
    unzip -q opencv_contrib.zip; \
    rm -f opencv_contrib.zip; \
    mkdir -p /opt/opencv-${OPENCV_VERSION}/build; \
    cd /opt/opencv-${OPENCV_VERSION}/build; \
    cmake \
      -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DWITH_CUBLAS=ON \
      -DCUDA_ARCH_BIN=${CUDA_ARCH} \
      -DCUDA_ARCH_PTX=${CUDA_ARCH} \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_opencv_python3=ON \
      -DPYTHON3_EXECUTABLE=/usr/bin/python3.10 \
      -DPYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
      -DPYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
      -DPYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      ..; \
    # ✅ g4dn.xlarge 안정 1순위: OOM 방지 위해 병렬 빌드 제한
    make -j 2; \
    make install; \
    ldconfig; \
    rm -rf /opt/opencv-${OPENCV_VERSION} /opt/opencv_contrib-${OPENCV_VERSION}

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
