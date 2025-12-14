#!/bin/bash
# AfterInstall 스크립트
# 환경 변수 설정, 디렉토리 생성, 모델 가중치 확인

set -e

echo "=== AfterInstall: 환경 설정 ==="

APP_DIR="/opt/echoshot-worker"
ENV_FILE="$APP_DIR/.env.prod"
CONFIG_DIR="$APP_DIR/EchoShotX-ai-private"

# 서브모듈에서 환경 변수 파일 복사
if [ -d "$CONFIG_DIR" ] && [ -f "$CONFIG_DIR/.env.prod" ]; then
    echo "서브모듈에서 환경 변수 파일을 복사합니다..."
    cp "$CONFIG_DIR/.env.prod" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "환경 변수 파일이 서브모듈에서 복사되었습니다: $ENV_FILE"
elif [ -f "$APP_DIR/EchoShotX-ai-private/.env.prod" ]; then
    # CodeDeploy 배포 시 서브모듈이 포함된 경우
    echo "배포된 서브모듈에서 환경 변수 파일을 복사합니다..."
    cp "$APP_DIR/EchoShotX-ai-private/.env.prod" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "환경 변수 파일이 복사되었습니다: $ENV_FILE"
else
    echo "WARNING: 서브모듈에서 .env.prod 파일을 찾을 수 없습니다."
    echo "기본 템플릿을 생성합니다..."
    
    # 환경 변수는 CodeDeploy 환경 변수나 Systems Manager Parameter Store에서 가져옴
    # 프로덕션 환경만 사용
    cat > "$ENV_FILE" <<EOF
# AWS 설정
AWS_REGION=${AWS_REGION:-ap-northeast-2}
SQS_QUEUE_URL=${SQS_QUEUE_URL}
S3_BUCKET_NAME=${S3_BUCKET_NAME}

# Spring API 설정
SPRING_API_BASE_URL=${SPRING_API_BASE_URL}
SPRING_API_TIMEOUT=${SPRING_API_TIMEOUT:-30}

# Worker 설정
WORKER_COUNT=${WORKER_COUNT:-4}
MAX_RETRIES=${MAX_RETRIES:-3}
VISIBILITY_TIMEOUT=${VISIBILITY_TIMEOUT:-300}

# Redis 설정
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_PASSWORD=${REDIS_PASSWORD:-}
REDIS_DB=${REDIS_DB:-0}

# 비디오 처리 설정
TEMP_DIR=/tmp/video_processing
MAX_VIDEO_SIZE_MB=${MAX_VIDEO_SIZE_MB:-500}

# 로깅 설정
LOG_LEVEL=${LOG_LEVEL:-INFO}
APP_ENV=prod
EOF
    chmod 600 "$ENV_FILE"
    echo "기본 템플릿이 생성되었습니다. 실제 값으로 업데이트가 필요합니다."
fi

# 디렉토리 생성
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/data"
mkdir -p "/tmp/video_processing"
chmod 755 "/tmp/video_processing"

# 모델 가중치 디렉토리 확인
WEIGHTS_DIR="$APP_DIR/weights"
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "모델 가중치 디렉토리를 생성합니다..."
    mkdir -p "$WEIGHTS_DIR"
fi

# 모델 가중치가 없으면 경고 (선택적)
if [ -z "$(ls -A $WEIGHTS_DIR 2>/dev/null)" ]; then
    echo "WARNING: 모델 가중치 파일이 없습니다."
    echo "Docker 이미지에 포함되어 있거나, 런타임에 다운로드해야 합니다."
fi

# Docker 네트워크 생성 (필요한 경우)
if ! docker network ls | grep -q echoshot-network; then
    echo "Docker 네트워크를 생성합니다..."
    docker network create echoshot-network || true
fi

# 파일 권한 설정
chown -R root:root "$APP_DIR"
chmod 755 "$APP_DIR"

echo "=== AfterInstall 완료 ==="

