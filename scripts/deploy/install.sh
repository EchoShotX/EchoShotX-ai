#!/bin/bash
# Install 스크립트
# Docker 이미지 Pull 및 컨테이너 준비

set -e

echo "=== Install: Docker 이미지 Pull ==="

APP_DIR="/opt/echoshot-worker"
cd "$APP_DIR"

# 환경 변수 로드 (있는 경우)
if [ -f "$APP_DIR/.env.prod" ]; then
    source "$APP_DIR/.env.prod"
fi

# ECR 정보 확인
if [ -z "$ECR_REPOSITORY" ] || [ -z "$AWS_REGION" ]; then
    echo "ERROR: ECR_REPOSITORY 또는 AWS_REGION이 설정되지 않았습니다."
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID:-}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPOSITORY}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "ECR Registry: $ECR_REGISTRY"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# ECR 로그인
echo "ECR에 로그인합니다..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# 기존 컨테이너 중지 및 제거 (있는 경우)
if [ "$(docker ps -aq -f name=echoshot-worker)" ]; then
    echo "기존 컨테이너를 중지합니다..."
    docker stop echoshot-worker || true
    docker rm echoshot-worker || true
fi

# 기존 이미지 제거 (선택적, 디스크 공간 절약)
# docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" || true

# 최신 이미지 Pull
echo "Docker 이미지를 Pull합니다..."
docker pull "${IMAGE_NAME}:${IMAGE_TAG}"

# 이미지 태그 확인
if ! docker images | grep -q "${ECR_REPOSITORY}"; then
    echo "ERROR: 이미지 Pull에 실패했습니다."
    exit 1
fi

echo "=== Install 완료 ==="

