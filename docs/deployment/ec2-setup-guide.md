# EC2 인스턴스 초기 설정 가이드 (Ubuntu)

## 개요

이 문서는 EchoShotX AI Worker 서비스를 배포하기 위해 EC2 인스턴스에 필요한 소프트웨어와 설정을 설치하는 방법을 안내합니다.

**대상 OS**: Ubuntu 20.04 / 22.04 LTS  
**인스턴스 타입**: GPU 인스턴스 권장 (g4dn.xlarge 이상)

## 사전 준비사항

### 1. EC2 인스턴스 생성

- **인스턴스 타입**: g4dn.xlarge (권장) 또는 g4dn.2xlarge
- **AMI**: Ubuntu Server 22.04 LTS (GPU 인스턴스는 Deep Learning AMI 사용 고려)
- **스토리지**: 최소 30GB (Docker 이미지 및 모델 가중치용)
- **보안 그룹**: 아래 "네트워크 설정" 참조

### 2. SSH 접속

```bash
# EC2 인스턴스에 SSH 접속
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

## 필수 소프트웨어 설치

### 1. 시스템 업데이트

```bash
# 패키지 목록 업데이트
sudo apt-get update

# 시스템 업그레이드 (선택사항)
sudo apt-get upgrade -y
```

### 2. Docker 설치

```bash
# Docker 설치를 위한 필수 패키지
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Docker 공식 GPG 키 추가
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker 저장소 추가
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker 서비스 시작 및 자동 시작 설정
sudo systemctl start docker
sudo systemctl enable docker

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 docker 명령어 사용)
sudo usermod -aG docker $USER

# Docker 설치 확인
docker --version
# 출력 예시: Docker version 24.0.7, build afdd53b

# Docker 서비스 상태 확인
sudo systemctl status docker
```

**참고**: docker 그룹 변경사항을 적용하려면 로그아웃 후 다시 로그인하거나 `newgrp docker` 명령어를 실행하세요.

### 3. Docker Compose 설치

```bash
# Docker Compose 설치 (최신 버전)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 실행 권한 부여
sudo chmod +x /usr/local/bin/docker-compose

# Docker Compose 설치 확인
docker-compose --version
# 출력 예시: Docker Compose version v2.27.1
```

**참고**: Docker 20.10 이상 버전에는 `docker compose` (하이픈 없음) 명령어가 포함되어 있습니다. 위 명령어는 별도 설치용입니다.

### 4. FFmpeg 설치

```bash
# FFmpeg 설치
sudo apt-get update
sudo apt-get install -y ffmpeg

# FFmpeg 설치 확인
ffmpeg -version
# 출력 예시: ffmpeg version 4.4.2
```

### 5. AWS CLI 설치

```bash
# AWS CLI v2 설치
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt-get install -y unzip
unzip awscliv2.zip
sudo ./aws/install

# AWS CLI 설치 확인
aws --version
# 출력 예시: aws-cli/2.15.0 Python/3.11.8 Linux/5.15.0

# AWS CLI 설정 (IAM 역할 사용 시 불필요)
# aws configure
```

### 6. CodeDeploy Agent 설치

```bash
# CodeDeploy Agent 설치
sudo apt-get update
sudo apt-get install -y ruby-full wget

# CodeDeploy Agent 다운로드 및 설치
cd /home/ubuntu
wget https://aws-codedeploy-ap-northeast-2.s3.ap-northeast-2.amazonaws.com/latest/install
chmod +x ./install
sudo ./install auto

# CodeDeploy Agent 서비스 상태 확인
sudo service codedeploy-agent status
# 출력 예시: codedeploy-agent (1.3.2-1902) is running.

# CodeDeploy Agent 자동 시작 설정
sudo systemctl enable codedeploy-agent
```

**참고**: CodeDeploy Agent는 EC2 인스턴스가 CodeDeploy 서비스에 등록되어 있어야 정상 작동합니다.

### 7. GPU 인스턴스 설정 (GPU 인스턴스인 경우)

#### 7.1 NVIDIA 드라이버 확인

```bash
# GPU 하드웨어 확인
lspci | grep -i nvidia
# 출력 예시: 00:1e.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)

# NVIDIA 드라이버 확인
nvidia-smi
# 출력 예시:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2  |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
# | N/A   30C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

**참고**: GPU 인스턴스는 일반적으로 NVIDIA 드라이버가 사전 설치되어 있습니다.  
만약 드라이버가 없다면 AWS Deep Learning AMI를 사용하거나 수동으로 설치해야 합니다.

#### 7.2 NVIDIA Container Toolkit 설치

Docker에서 GPU를 사용하려면 NVIDIA Container Toolkit이 필요합니다.

```bash
# 배포 키 추가
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# NVIDIA Container Toolkit 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 데몬 재시작
sudo systemctl restart docker

# GPU 지원 확인
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
# 출력 예시: NVIDIA-SMI 정보가 표시되어야 함
```

**참고**: `--gpus all` 옵션은 Docker Compose에서도 사용 가능합니다 (docker-compose.yml에 설정됨).

## 디렉토리 구조 생성

```bash
# 애플리케이션 디렉토리 생성
sudo mkdir -p /opt/echoshot-worker
sudo mkdir -p /opt/echoshot-worker/logs
sudo mkdir -p /opt/echoshot-worker/data
sudo mkdir -p /tmp/video_processing

# 권한 설정
sudo chmod 755 /opt/echoshot-worker
sudo chmod 755 /tmp/video_processing

# 로그 디렉토리 생성
sudo mkdir -p /var/log/echoshot-worker
```

## Docker 네트워크 생성

```bash
# Docker 네트워크 생성 (배포 스크립트에서도 자동 생성됨)
docker network create echoshot-network || true

# 네트워크 확인
docker network ls | grep echoshot-network
```

## IAM 역할 설정

EC2 인스턴스가 AWS 서비스(S3, SQS, CodeDeploy)에 접근할 수 있도록 IAM 역할을 설정해야 합니다.

### 1. IAM 역할 생성

AWS 콘솔에서 다음 권한을 가진 IAM 역할을 생성:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::codedeploy-echoshot-*",
        "arn:aws:s3:::codedeploy-echoshot-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:ChangeMessageVisibility",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:ap-northeast-2:*:video-processing-queue"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeTags"
      ],
      "Resource": "*"
    }
  ]
}
```

### 2. CodeDeploy 서비스 역할

CodeDeploy가 EC2 인스턴스에 접근할 수 있도록 다음 정책을 추가:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:TerminateInstances"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion"
      ],
      "Resource": "arn:aws:s3:::codedeploy-echoshot-*/*"
    }
  ]
}
```

### 3. EC2 인스턴스에 IAM 역할 연결

```bash
# AWS CLI로 현재 인스턴스의 IAM 역할 확인
aws sts get-caller-identity

# 또는 EC2 콘솔에서:
# 1. EC2 인스턴스 선택
# 2. Actions → Security → Modify IAM role
# 3. 생성한 IAM 역할 선택
```

## 네트워크 설정

### 보안 그룹 설정

EC2 인스턴스의 보안 그룹에 다음 규칙을 추가:

| 타입 | 프로토콜 | 포트 범위 | 소스 | 설명 |
|------|---------|----------|------|------|
| SSH | TCP | 22 | 내 IP | SSH 접속용 |
| HTTPS | TCP | 443 | 0.0.0.0/0 | 외부 API 호출용 |
| Custom TCP | TCP | 6379 | EC2-B Private IP | Redis 접속용 (선택) |

**인바운드 규칙**:
- SSH (22): 관리자 접속용
- HTTPS (443): Spring API 콜백용 (필요한 경우)

**아웃바운드 규칙**:
- 모든 트래픽 허용 (S3, SQS, Redis 접근용)

### VPC 설정

- **서브넷**: 프라이빗 서브넷 권장 (보안 강화)
- **NAT Gateway**: 인터넷 접근 필요 시 (S3, SQS, Docker Hub 등)

## Redis 호스트 설정 (EC2-B 접근)

EC2-B의 Redis에 접근하기 위해 `/etc/hosts`에 매핑을 추가합니다.

```bash
# EC2-B의 Private IP 확인 (AWS CLI 사용)
EC2_B_IP=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=EC2-B" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' \
    --output text)

# /etc/hosts에 추가
echo "$EC2_B_IP redis" | sudo tee -a /etc/hosts

# 확인
cat /etc/hosts | grep redis
```

**참고**: 배포 스크립트(`after_install.sh`)에서 자동으로 설정됩니다.

## 설치 확인

### 전체 설치 확인 스크립트

```bash
#!/bin/bash
echo "=== 설치 확인 ==="

echo "1. Docker:"
docker --version || echo "❌ Docker 미설치"

echo "2. Docker Compose:"
docker-compose --version || echo "❌ Docker Compose 미설치"

echo "3. FFmpeg:"
ffmpeg -version | head -n 1 || echo "❌ FFmpeg 미설치"

echo "4. AWS CLI:"
aws --version || echo "❌ AWS CLI 미설치"

echo "5. CodeDeploy Agent:"
sudo service codedeploy-agent status || echo "❌ CodeDeploy Agent 미설치"

echo "6. GPU (GPU 인스턴스인 경우):"
if lspci | grep -i nvidia &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "❌ NVIDIA 드라이버 미설치"
    docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null && echo "✅ Docker GPU 지원" || echo "❌ Docker GPU 미지원"
else
    echo "⚠️  GPU 인스턴스가 아닙니다."
fi

echo "7. 디렉토리:"
[ -d /opt/echoshot-worker ] && echo "✅ /opt/echoshot-worker 존재" || echo "❌ /opt/echoshot-worker 없음"
[ -d /tmp/video_processing ] && echo "✅ /tmp/video_processing 존재" || echo "❌ /tmp/video_processing 없음"

echo "8. Docker 네트워크:"
docker network ls | grep -q echoshot-network && echo "✅ echoshot-network 존재" || echo "❌ echoshot-network 없음"

echo "9. IAM 역할:"
aws sts get-caller-identity && echo "✅ IAM 역할 설정됨" || echo "❌ IAM 역할 미설정"

echo "=== 확인 완료 ==="
```

## CodeDeploy 설정

### 1. CodeDeploy 애플리케이션 생성

AWS 콘솔에서 CodeDeploy 애플리케이션을 생성하고 배포 그룹을 설정합니다.

### 2. EC2 인스턴스 태깅

CodeDeploy가 인스턴스를 식별할 수 있도록 태그를 추가:

```bash
# AWS CLI로 태그 추가 (또는 EC2 콘솔에서)
aws ec2 create-tags \
    --resources i-xxxxxxxxxxxxxxxxx \
    --tags Key=Name,Value=EC2-A Key=Environment,Value=production
```

### 3. 배포 그룹 설정

CodeDeploy 배포 그룹에서 다음 태그를 사용하여 인스턴스를 선택:
- `Name=EC2-A`
- `Environment=production`

## 다음 단계

1. **환경 변수 설정**: 서브모듈의 `.env.prod` 파일 확인
2. **첫 배포**: GitHub Actions를 통해 자동 배포 또는 수동 배포
3. **모니터링**: CloudWatch 로그 및 메트릭 확인

## 문제 해결

### CodeDeploy Agent가 시작되지 않음

```bash
# CodeDeploy Agent 로그 확인
sudo tail -f /var/log/aws/codedeploy-agent/codedeploy-agent.log

# CodeDeploy Agent 재시작
sudo service codedeploy-agent restart
```

### Docker 권한 오류

```bash
# 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 로그아웃 후 다시 로그인하거나
newgrp docker
```

### GPU가 인식되지 않음

```bash
# NVIDIA 드라이버 재설치 (필요한 경우)
sudo apt-get purge nvidia-*
sudo apt-get install -y nvidia-driver-535  # 버전은 인스턴스에 맞게 조정

# NVIDIA Container Toolkit 재설치
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

## 참고 자료

- [Docker 설치 가이드](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit 설치 가이드](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [CodeDeploy Agent 설치 가이드](https://docs.aws.amazon.com/codedeploy/latest/userguide/codedeploy-agent-operations-install-ubuntu.html)
- [EC2 비용 최적화 가이드](./cost-optimization.md)

