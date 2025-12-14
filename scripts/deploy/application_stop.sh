#!/bin/bash
# ApplicationStop 스크립트
# Docker Compose로 Graceful shutdown

set -e

echo "=== ApplicationStop: Docker Compose로 컨테이너 중지 ==="

APP_DIR="/opt/echoshot-worker"
cd "$APP_DIR"

# docker-compose.yml 파일이 있으면 Docker Compose로 중지
if [ -f "$APP_DIR/docker-compose.yml" ]; then
    echo "Docker Compose로 컨테이너를 중지합니다..."
    docker-compose down -t 30 || true
    echo "컨테이너가 중지되었습니다."
else
    echo "docker-compose.yml 파일이 없습니다. 직접 컨테이너를 중지합니다..."
    
    # 실행 중인 컨테이너 확인
    if [ "$(docker ps -q -f name=echoshot-worker)" ]; then
        echo "컨테이너를 중지합니다..."
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
fi

echo "=== ApplicationStop 완료 ==="

