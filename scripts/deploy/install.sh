#!/bin/bash
# Install 스크립트
# Docker Compose 준비

set -e

echo "=== Install: Docker Compose 준비 ==="

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

echo "=== Install 완료 ==="

