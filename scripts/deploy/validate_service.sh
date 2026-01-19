#!/bin/bash
# ValidateService 스크립트
# Health check

set -e

echo "=== ValidateService: 서비스 검증 ==="

APP_DIR="/opt/echoshot-worker"
cd "$APP_DIR"

# Docker Compose로 실행 중인 경우
if [ -f "$APP_DIR/docker-compose.yml" ]; then
    # Docker Compose 상태 확인
    if ! docker-compose ps | grep -q "Up"; then
        echo "ERROR: Docker Compose 서비스가 실행 중이 아닙니다."
        docker-compose ps
        exit 1
    fi
    
    # 컨테이너 실행 상태 확인
    if ! docker ps | grep -q echoshot-worker; then
        echo "ERROR: 컨테이너가 실행 중이 아닙니다."
        exit 1
    fi
    
    # 컨테이너 로그 확인 (최근 에러 확인)
    RECENT_ERRORS=$(docker-compose logs --tail 50 2>&1 | grep -i "error\|exception\|failed" | tail -5)
    if [ -n "$RECENT_ERRORS" ]; then
        echo "WARNING: 최근 로그에 에러가 있습니다:"
        echo "$RECENT_ERRORS"
    fi
else
    # 직접 docker run으로 실행 중인 경우 (fallback)
    if ! docker ps | grep -q echoshot-worker; then
        echo "ERROR: 컨테이너가 실행 중이 아닙니다."
        exit 1
    fi
    
    # 컨테이너 상태 확인
    CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' echoshot-worker)
    if [ "$CONTAINER_STATUS" != "running" ]; then
        echo "ERROR: 컨테이너 상태가 'running'이 아닙니다. 현재 상태: $CONTAINER_STATUS"
        exit 1
    fi
    
    # 컨테이너 로그 확인
    RECENT_ERRORS=$(docker logs --tail 50 echoshot-worker 2>&1 | grep -i "error\|exception\|failed" | tail -5)
    if [ -n "$RECENT_ERRORS" ]; then
        echo "WARNING: 최근 로그에 에러가 있습니다:"
        echo "$RECENT_ERRORS"
    fi
fi

# 프로세스 확인 (Python 프로세스가 실행 중인지)
if ! docker exec echoshot-worker ps aux | grep -q "[p]ython.*main"; then
    echo "ERROR: Python 애플리케이션이 실행 중이 아닙니다."
    exit 1
fi

# Redis 연결 테스트 (옵션 - 실패해도 계속 진행)
# REDIS_HOST 환경 변수가 있으면 테스트
if [ -f "$APP_DIR/.env.prod" ]; then
    source "$APP_DIR/.env.prod"
    if [ -n "$REDIS_HOST" ]; then
        if ! docker exec echoshot-worker python -c "import redis; r = redis.Redis(host='$REDIS_HOST', port=${REDIS_PORT:-6379}, socket_connect_timeout=5); r.ping()" 2>/dev/null; then
            echo "WARNING: Redis 연결에 실패했습니다. (선택적이므로 계속 진행)"
        else
            echo "Redis 연결 확인 완료."
        fi
    fi
fi

echo "서비스가 정상적으로 실행 중입니다."
echo "=== ValidateService 완료 ==="

exit 0

