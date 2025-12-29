# FSRCNN GPU 업스케일 전용 Dockerfile
# NVIDIA T4 GPU (CUDA Compute 7.5)를 위한 OpenCV CUDA 빌드

# 빌드 인자 정의
ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04
ARG CUDNN=8
ARG OPENCV_VERSION=4.10.0
ARG CUDA_ARCH=7.5

# CUDA 베이스 이미지
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN}-devel-ubuntu${UBUNTU_VERSION}

# 환경 변수 설정
ENV OPENCV_VERSION=$OPENCV_VERSION
ENV CUDA_ARCH=$CUDA_ARCH
ARG DEBIAN_FRONTEND=noninteractive

# 빌드 도구 및 의존성 설치
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 필수: OpenCV 빌드 전에 NumPy 설치 (Python 바인딩 빌드에 필요)
RUN apt-get update && apt-get install -y python3-numpy && rm -rf /var/lib/apt/lists/*

# OpenCV 및 opencv_contrib 다운로드 및 빌드
RUN cd /opt/ && \
    # OpenCV 다운로드
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm ${OPENCV_VERSION}.zip && \
    # opencv_contrib 다운로드
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm ${OPENCV_VERSION}.zip && \
    # 빌드 디렉토리 생성
    mkdir /opt/opencv-${OPENCV_VERSION}/build && \
    cd /opt/opencv-${OPENCV_VERSION}/build && \
    # CMake 설정 (CUDA 지원, NVIDIA T4 아키텍처 7.5, DNN CUDA 명시적 활성화)
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=${CUDA_ARCH} \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_opencv_python3=ON \
        -DPYTHON3_EXECUTABLE=/usr/bin/python3.10 \
        -DPYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
        -DPYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
        -DPYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages \
        -DOPENCV_DNN_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_CUBLAS=ON \
        .. && \
    # 빌드 (모든 CPU 코어 사용)
    make -j $(nproc --all) && \
    make install && \
    ldconfig && \
    # 빌드 파일 정리
    rm -rf /opt/opencv-${OPENCV_VERSION} && \
    rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 Python 패키지 설치
COPY echoshot_ai_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY echoshot_ai_server/ /app/echoshot_ai_server/

# 모델 가중치 디렉토리 생성 및 FSRCNN 모델 파일 다운로드
RUN mkdir -p /app/weights /app/echoshot_ai_server/tasks/weights && \
    echo "Downloading FSRCNN model file..." && \
    curl -L -o /app/weights/FSRCNN_x2.pb \
        https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb && \
    cp /app/weights/FSRCNN_x2.pb /app/echoshot_ai_server/tasks/weights/ && \
    echo "FSRCNN model file downloaded successfully"

# 비root 사용자 생성
RUN useradd -m -u 1000 worker && \
    mkdir -p /tmp/video_processing && \
    chown -R worker:worker /app /tmp/video_processing

# 소유권 변경
RUN chown -R worker:worker /app

# 비root 사용자로 전환
USER worker

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TEMP_DIR=/tmp/video_processing \
    APP_ENV=prod \
    PYTHONPATH=/app

# 임시 디렉토리 설정
VOLUME ["/tmp/video_processing"]

# 애플리케이션 실행
ENTRYPOINT ["python", "-m", "echoshot_ai_server.main"]
