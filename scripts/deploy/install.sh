#!/bin/bash
# Install 스크립트
# Docker 이미지 Pull 및 준비 (빌드는 GitHub Actions에서 수행)

set -e

echo "=== Install: Docker 이미지 준비 ==="

APP_DIR="/opt/echoshot-worker"
cd "$APP_DIR"

# Docker Compose 설치 확인
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose가 설치되어 있지 않습니다. 설치를 진행합니다..."
    # Docker Compose 설치 (최신 버전)
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose가 설치되었습니다."
fi

# docker-compose.yml 파일 확인
if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
    echo "ERROR: docker-compose.yml 파일이 없습니다."
    exit 1
fi

# Docker 네트워크 확인 (없으면 생성)
if ! docker network ls | grep -q echoshot-network; then
    echo "Docker 네트워크를 생성합니다..."
    docker network create echoshot-network || true
fi

# 기존 컨테이너 정리 (있는 경우)
if [ -f "$APP_DIR/docker-compose.yml" ]; then
    echo "기존 컨테이너를 정리합니다..."
    cd "$APP_DIR"
    docker-compose down || true
fi

# 환경 변수 확인 (.env.prod에서 DOCKER_IMAGE 설정 확인)
ENV_FILE="$APP_DIR/.env.prod"
if [ -f "$ENV_FILE" ]; then
    # .env.prod에서 DOCKER_IMAGE 추출
    DOCKER_IMAGE=$(grep "^DOCKER_IMAGE=" "$ENV_FILE" | cut -d '=' -f2 | tr -d '"' | tr -d "'" || echo "")
    if [ -n "$DOCKER_IMAGE" ]; then
        echo "환경 변수에서 이미지 태그 확인: $DOCKER_IMAGE"
        export DOCKER_IMAGE
    fi
fi

# 이미지가 명시되지 않았으면 기본값 사용
if [ -z "$DOCKER_IMAGE" ]; then
    echo "DOCKER_IMAGE가 설정되지 않았습니다. 기본값을 사용합니다: echoshot/echoshot-worker:gpu-fsrcnn-t4"
    export DOCKER_IMAGE="echoshot/echoshot-worker:gpu-fsrcnn-t4"
fi

echo "사용할 Docker 이미지: $DOCKER_IMAGE"

# Docker 이미지 Pull (GitHub Actions에서 빌드되어 Docker Hub에 푸시된 이미지)
echo "Docker Hub에서 이미지를 pull합니다..."
echo "이 작업은 약 1-3분이 소요될 수 있습니다."
docker pull "$DOCKER_IMAGE" || {
    echo "ERROR: Docker 이미지 pull에 실패했습니다: $DOCKER_IMAGE"
    echo "이미지가 Docker Hub에 있는지 확인하거나, GitHub Actions 빌드가 완료되었는지 확인하세요."
    exit 1
}

echo "이미지 pull 완료!"

# 이미지 태그 확인
docker images | grep "$DOCKER_IMAGE" || echo "WARNING: 이미지가 로컬에 없습니다."

echo "=== Install 완료 ==="

