#!/bin/bash
# BeforeInstall 스크립트
# 시스템 의존성 확인 및 설치

set -e

echo "=== BeforeInstall: 시스템 의존성 확인 ==="

# 로그 디렉토리 생성
LOG_DIR="/var/log/echoshot-worker"
mkdir -p "$LOG_DIR"

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "Docker가 설치되어 있지 않습니다. 설치를 진행합니다..."
    # Amazon Linux 2
    if [ -f /etc/os-release ] && grep -q "Amazon Linux" /etc/os-release; then
        yum update -y
        yum install -y docker
        systemctl start docker
        systemctl enable docker
    # Ubuntu
    elif [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then
        apt-get update
        apt-get install -y docker.io
        systemctl start docker
        systemctl enable docker
    else
        echo "지원하지 않는 OS입니다."
        exit 1
    fi
fi

# Docker 서비스 상태 확인
if ! systemctl is-active --quiet docker; then
    echo "Docker 서비스를 시작합니다..."
    systemctl start docker
fi

# Docker Compose 설치 확인 (옵션)
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose를 설치합니다..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# FFmpeg 설치 확인
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg가 설치되어 있지 않습니다. 설치를 진행합니다..."
    # Amazon Linux 2
    if [ -f /etc/os-release ] && grep -q "Amazon Linux" /etc/os-release; then
        yum install -y ffmpeg
    # Ubuntu
    elif [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then
        apt-get update
        apt-get install -y ffmpeg
    fi
fi

# AWS CLI 설치 확인 (ECR 로그인용)
if ! command -v aws &> /dev/null; then
    echo "AWS CLI를 설치합니다..."
    # Amazon Linux 2
    if [ -f /etc/os-release ] && grep -q "Amazon Linux" /etc/os-release; then
        yum install -y aws-cli
    # Ubuntu
    elif [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then
        apt-get update
        apt-get install -y awscli
    fi
fi

# 작업 디렉토리 생성
APP_DIR="/opt/echoshot-worker"
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/logs"
mkdir -p "/tmp/video_processing"

# Docker 그룹에 ec2-user 추가 (필요한 경우)
if id "ec2-user" &>/dev/null; then
    usermod -aG docker ec2-user || true
fi

echo "=== BeforeInstall 완료 ==="

