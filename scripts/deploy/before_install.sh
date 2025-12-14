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

# GPU 인스턴스 확인 및 NVIDIA 드라이버/Docker GPU 지원 확인
if lspci | grep -i nvidia &> /dev/null; then
    echo "GPU 인스턴스가 감지되었습니다. NVIDIA 드라이버 및 Docker GPU 지원을 확인합니다..."
    
    # NVIDIA 드라이버 확인
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: NVIDIA 드라이버가 설치되어 있지 않습니다."
        echo "GPU 인스턴스는 일반적으로 NVIDIA 드라이버가 사전 설치되어 있습니다."
        echo "수동으로 설치가 필요한 경우 AWS Deep Learning AMI를 사용하거나 NVIDIA 드라이버를 설치하세요."
    else
        echo "NVIDIA 드라이버 확인 완료:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    fi
    
    # NVIDIA Container Toolkit 확인 (Docker GPU 지원)
    if [ ! -f /usr/bin/nvidia-container-runtime ]; then
        echo "NVIDIA Container Toolkit이 설치되어 있지 않습니다. 설치를 진행합니다..."
        # Ubuntu/Debian
        if [ -f /etc/os-release ] && grep -q "Ubuntu\|Debian" /etc/os-release; then
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
            apt-get update
            apt-get install -y nvidia-container-toolkit
            systemctl restart docker
        # Amazon Linux 2
        elif [ -f /etc/os-release ] && grep -q "Amazon Linux" /etc/os-release; then
            distribution="rhel7"
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo
            yum install -y nvidia-container-toolkit
            systemctl restart docker
        else
            echo "WARNING: NVIDIA Container Toolkit 자동 설치를 지원하지 않는 OS입니다."
            echo "수동으로 설치해주세요: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    else
        echo "NVIDIA Container Toolkit 확인 완료."
    fi
else
    echo "GPU 인스턴스가 아닙니다. CPU 모드로 실행됩니다."
fi

echo "=== BeforeInstall 완료 ==="

