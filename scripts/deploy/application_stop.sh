#!/bin/bash
# ApplicationStop 스크립트
# Graceful shutdown

set -e

echo "=== ApplicationStop: 컨테이너 중지 ==="

# 실행 중인 컨테이너 확인
if [ "$(docker ps -q -f name=echoshot-worker)" ]; then
    echo "컨테이너를 중지합니다..."
    
    # Graceful shutdown: SIGTERM 전송
    docker stop -t 30 echoshot-worker || true
    
    # 컨테이너가 여전히 실행 중이면 강제 종료
    if [ "$(docker ps -q -f name=echoshot-worker)" ]; then
        echo "컨테이너가 30초 내에 종료되지 않아 강제 종료합니다..."
        docker kill echoshot-worker || true
    fi
    
    # 컨테이너 제거
    docker rm echoshot-worker || true
    echo "컨테이너가 중지되었습니다."
else
    echo "실행 중인 컨테이너가 없습니다."
fi

# 정지된 컨테이너 정리 (선택적)
if [ "$(docker ps -aq -f name=echoshot-worker)" ]; then
    echo "정지된 컨테이너를 제거합니다..."
    docker rm echoshot-worker || true
fi

echo "=== ApplicationStop 완료 ==="

