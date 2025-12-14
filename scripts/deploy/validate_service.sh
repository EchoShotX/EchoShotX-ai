#!/bin/bash
# ValidateService 스크립트
# Health check

set -e

echo "=== ValidateService: 서비스 검증 ==="

# 컨테이너 실행 상태 확인
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

# 컨테이너 로그 확인 (최근 에러 확인)
RECENT_ERRORS=$(docker logs --tail 50 echoshot-worker 2>&1 | grep -i "error\|exception\|failed" | tail -5)
if [ -n "$RECENT_ERRORS" ]; then
    echo "WARNING: 최근 로그에 에러가 있습니다:"
    echo "$RECENT_ERRORS"
    # 에러가 있어도 서비스는 실행 중이므로 경고만 출력
fi

# 프로세스 확인 (Python 프로세스가 실행 중인지)
if ! docker exec echoshot-worker ps aux | grep -q "[p]ython.*main"; then
    echo "ERROR: Python 애플리케이션이 실행 중이 아닙니다."
    exit 1
fi

# SQS 연결 테스트 (간접적)
# 실제로는 애플리케이션 내부에서 확인해야 하지만, 여기서는 컨테이너 상태만 확인

# Redis 연결 테스트 (옵션)
# REDIS_HOST 환경 변수가 있으면 테스트
if [ -n "$REDIS_HOST" ]; then
    if ! docker exec echoshot-worker python -c "import redis; r = redis.Redis(host='$REDIS_HOST', port=${REDIS_PORT:-6379}, socket_connect_timeout=5); r.ping()" 2>/dev/null; then
        echo "WARNING: Redis 연결에 실패했습니다. (선택적)"
    fi
fi

echo "서비스가 정상적으로 실행 중입니다."
echo "=== ValidateService 완료 ==="

exit 0

