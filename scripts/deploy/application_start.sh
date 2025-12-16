#!/bin/bash
# ApplicationStart 스크립트
# Docker Compose로 컨테이너 실행 (빌드 없이)

set -e

echo "=== ApplicationStart: Docker Compose로 컨테이너 시작 ==="

APP_DIR="/opt/echoshot-worker"
ENV_FILE="$APP_DIR/.env.prod"
cd "$APP_DIR"

# 환경 변수 파일 확인
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env.prod 파일이 없습니다."
    exit 1
fi

# docker-compose.yml 파일 확인
if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
    echo "ERROR: docker-compose.yml 파일이 없습니다."
    exit 1
fi

# Docker Compose로 컨테이너 실행 (빌드 없이)
echo "Docker Compose로 컨테이너를 시작합니다..."
docker-compose up -d

# 컨테이너 시작 확인 (헬스체크 기반)
echo "컨테이너 시작 대기 중..."
MAX_WAIT=60
ELAPSED=0
INTERVAL=2

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if docker ps | grep -q echoshot-worker; then
        CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' echoshot-worker 2>/dev/null || echo "notfound")
        if [ "$CONTAINER_STATUS" = "running" ]; then
            echo "컨테이너가 정상적으로 시작되었습니다."
            break
        fi
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "대기 중... (${ELAPSED}/${MAX_WAIT}초)"
done

if ! docker ps | grep -q echoshot-worker; then
    echo "ERROR: 컨테이너가 시작되지 않았습니다."
    echo "컨테이너 로그:"
    docker-compose logs
    echo "컨테이너 상태:"
    docker-compose ps
    exit 1
fi

echo "컨테이너가 성공적으로 시작되었습니다."
docker ps | grep echoshot-worker
docker-compose ps

echo "=== ApplicationStart 완료 ==="

