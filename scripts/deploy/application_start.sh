#!/bin/bash
# ApplicationStart 스크립트
# Docker 컨테이너 실행

set -e

echo "=== ApplicationStart: 컨테이너 시작 ==="

APP_DIR="/opt/echoshot-worker"
ENV_FILE="$APP_DIR/.env.prod"

# 환경 변수 로드
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# ECR 정보 확인
if [ -z "$ECR_REPOSITORY" ] || [ -z "$AWS_REGION" ]; then
    echo "ERROR: ECR_REPOSITORY 또는 AWS_REGION이 설정되지 않았습니다."
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID:-}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPOSITORY}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# 기존 컨테이너가 실행 중이면 중지
if [ "$(docker ps -q -f name=echoshot-worker)" ]; then
    echo "기존 컨테이너를 중지합니다..."
    docker stop echoshot-worker
    docker rm echoshot-worker
fi

# 환경 변수 파일을 컨테이너에 전달
ENV_ARGS=""
if [ -f "$ENV_FILE" ]; then
    ENV_ARGS="--env-file $ENV_FILE"
fi

# Docker 컨테이너 실행
echo "Docker 컨테이너를 시작합니다..."
docker run -d \
    --name echoshot-worker \
    --restart unless-stopped \
    --network echoshot-network \
    $ENV_ARGS \
    -e AWS_REGION="${AWS_REGION}" \
    -e SQS_QUEUE_URL="${SQS_QUEUE_URL}" \
    -e S3_BUCKET_NAME="${S3_BUCKET_NAME}" \
    -e SPRING_API_BASE_URL="${SPRING_API_BASE_URL}" \
    -e WORKER_COUNT="${WORKER_COUNT:-4}" \
    -e REDIS_HOST="${REDIS_HOST}" \
    -e REDIS_PORT="${REDIS_PORT:-6379}" \
    -e APP_ENV=prod \
    -v /tmp/video_processing:/tmp/video_processing \
    -v "$APP_DIR/logs:/app/logs" \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    "${IMAGE_NAME}:${IMAGE_TAG}"

# 컨테이너 시작 확인
sleep 5
if ! docker ps | grep -q echoshot-worker; then
    echo "ERROR: 컨테이너가 시작되지 않았습니다."
    docker logs echoshot-worker
    exit 1
fi

echo "컨테이너가 성공적으로 시작되었습니다."
docker ps | grep echoshot-worker

echo "=== ApplicationStart 완료 ==="

