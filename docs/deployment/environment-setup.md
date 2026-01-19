# 환경 변수 및 GitHub Secrets 설정 가이드

베이스 이미지 분리 배포 방식을 사용하기 위해 필요한 모든 환경 변수와 GitHub Secrets 설정 가이드입니다.

## 1. EC2 환경 변수 (.env.prod)

EC2 인스턴스의 `/opt/echoshot-worker/.env.prod` 파일에 설정합니다.

### 필수 환경 변수

```bash
# ===============================
# AWS 설정
# ===============================
AWS_REGION=ap-northeast-2
SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/YOUR_ACCOUNT_ID/video-processing-queue
S3_BUCKET_NAME=echoshot-videos

# ===============================
# Spring API 설정
# ===============================
SPRING_API_BASE_URL=https://api.echoshot.com
SPRING_API_TIMEOUT=30

# ===============================
# Worker 설정
# ===============================
WORKER_COUNT=4
MAX_RETRIES=3
VISIBILITY_TIMEOUT=300

# ===============================
# Redis 설정 (선택사항)
# ===============================
REDIS_HOST=redis.ec2-b.internal
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# ===============================
# 비디오 처리 설정
# ===============================
TEMP_DIR=/tmp/video_processing
MAX_VIDEO_SIZE_MB=500

# ===============================
# 로깅 설정
# ===============================
LOG_LEVEL=INFO
APP_ENV=prod

# ===============================
# Docker 이미지 설정 (중요!)
# ==========================
# 또는 최신 버전 사용 시:
# DOCKER_IMAGE=echoshot/echoshot-worker:latest
```

### 환경 변수 설명

| 변수명 | 필수 | 설명 | 예시 |
|--------|------|------|------|
| `AWS_REGION` | 예 | AWS 리전 | `ap-northeast-2` |
| `SQS_QUEUE_URL` | 예 | SQS 큐 URL | `https://sqs.ap-northeast-2.amazonaws.com/123456789012/video-processing-queue` |
| `S3_BUCKET_NAME` | 예 | S3 버킷 이름 | `echoshot-videos` |
| `SPRING_API_BASE_URL` | 예 | Spring API 서버 URL | `https://api.echoshot.com` |
| `SPRING_API_TIMEOUT` | 선택 | API 타임아웃 (초) | `30` |
| `WORKER_COUNT` | 선택 | 동시 처리 워커 수 | `4` |
| `MAX_RETRIES` | 선택 | 최대 재시도 횟수 | `3` |
| `VISIBILITY_TIMEOUT` | 선택 | SQS 메시지 가시성 타임아웃 (초) | `300` |
| `DOCKER_IMAGE` | 예 | Docker Hub 이미지 태그 | `echoshot/echoshot-worker:gpu-fsrcnn-t4` |
| `LOG_LEVEL` | 선택 | 로그 레벨 | `INFO`, `DEBUG`, `WARNING`, `ERROR` |

### .env.prod 파일 생성 방법

```bash
# EC2 인스턴스에서
cd /opt/echoshot-worker
cp env.prod.example .env.prod
nano .env.prod  # 또는 vi .env.prod
# 실제 값으로 수정 후 저장
```

## 2. GitHub Secrets 설정

GitHub Repository → Settings → Secrets and variables → Actions → New repository secret

### 필수 Secrets

#### Docker Hub 관련

| Secret 이름 | 설명 | 예시 | 필수 |
|------------|------|------|------|
| `DOCKERHUB_USERNAME` | Docker Hub 사용자명 | `echoshot` | 예 |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token | (Docker Hub에서 생성) | 예 |

**Docker Hub Token 생성 방법**:
1. Docker Hub 로그인 (https://hub.docker.com)
2. Account Settings → Security → New Access Token
3. Token 이름: `github-actions-echoshot`
4. Permissions: `Read, Write, Delete`
5. 생성된 Token을 GitHub Secrets에 등록

#### AWS 관련

| Secret 이름 | 설명 | 예시 | 필수 |
|------------|------|------|------|
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 ID | `AKIAIOSFODNN7EXAMPLE` | 예 |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 액세스 키 | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | 예 |
| `AWS_REGION` | AWS 리전 | `ap-northeast-2` | 선택* |

\* 기본값: `ap-northeast-2` (워크플로우에 기본값 설정됨)

**AWS IAM 권한 설정**:
다음 권한이 필요한 IAM 사용자를 생성하고 Access Key를 발급받아야 합니다:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::codedeploy-echoshot-*/*",
        "arn:aws:s3:::codedeploy-echoshot-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "codedeploy:CreateDeployment",
        "codedeploy:GetDeployment"
      ],
      "Resource": "*"
    }
  ]
}
```

#### CodeDeploy 관련

| Secret 이름 | 설명 | 예시 | 필수 |
|------------|------|------|------|
| `CODE_DEPLOY_APPLICATION_NAME` | CodeDeploy 애플리케이션 이름 | `echoshot-worker` | 예 |
| `CODE_DEPLOY_DEPLOYMENT_GROUP` | CodeDeploy 배포 그룹 이름 | `production` | 예 |
| `CODEDEPLOY_S3_BUCKET` | CodeDeploy용 S3 버킷 (선택사항) | `codedeploy-echoshot-apnortheast2` | 선택* |

\* 지정하지 않으면 자동 생성됩니다.

#### GitHub 관련 (선택사항)

| Secret 이름 | 설명 | 예시 | 필수 |
|------------|------|------|------|
| `PAT_TOKEN` | Private Access Token (서브모듈 접근용) | `ghp_xxxxxxxxxxxx` | 선택* |

\* 서브모듈이 private이거나 특별한 권한이 필요한 경우만 설정

### GitHub Secrets 설정 체크리스트

```
필수:
[ ] DOCKERHUB_USERNAME
[ ] DOCKERHUB_TOKEN
[ ] AWS_ACCESS_KEY_ID
[ ] AWS_SECRET_ACCESS_KEY
[ ] CODE_DEPLOY_APPLICATION_NAME
[ ] CODE_DEPLOY_DEPLOYMENT_GROUP

선택:
[ ] AWS_REGION (기본값: ap-northeast-2)
[ ] CODEDEPLOY_S3_BUCKET (자동 생성 가능)
[ ] PAT_TOKEN (서브모듈이 private인 경우만)
```

## 3. 베이스 이미지 빌드 시 환경 변수

EC2에서 베이스 이미지를 빌드할 때 사용하는 환경 변수입니다.

### 환경 변수 (선택사항)

베이스 이미지 빌드 스크립트 실행 시 설정 가능:

```bash
# 기본값 사용 (권장)
./scripts/build-base-image.sh

# 또는 환경 변수로 커스터마이징
export DOCKERHUB_USERNAME=your-dockerhub-username
export OPENCV_VERSION=4.10.0
./scripts/build-base-image.sh
```

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `DOCKERHUB_USERNAME` | `echoshot` | Docker Hub 사용자명 |
| `OPENCV_VERSION` | `4.10.0` | OpenCV 버전 |

**참고**: 베이스 이미지 빌드 전에 Docker Hub에 로그인해야 합니다:

```bash
docker login
# 사용자명과 비밀번호 입력
```

## 4. EC2 IAM 역할 설정

EC2 인스턴스가 AWS 서비스에 접근하기 위한 IAM 역할입니다.

### 필수 권한

EC2 Instance Profile에 다음 권한을 부여해야 합니다:

```json
{
  "Version": "2012-10-17",
  "Statement": [
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
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::echoshot-videos/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::echoshot-videos"
    }
  ]
}
```

## 5. 설정 확인 방법

### EC2 환경 변수 확인

```bash
# EC2 인스턴스에서
cd /opt/echoshot-worker
cat .env.prod
# 또는
docker-compose config  # 환경 변수 확인
```

### GitHub Secrets 확인

GitHub Repository → Settings → Secrets and variables → Actions에서 확인

### 베이스 이미지 확인

```bash
# Docker Hub에서 확인
docker pull echoshot/opencv-cuda-t4:4.10.0

# 또는 로컬에서 확인
docker images | grep opencv-cuda-t4
```

## 6. 문제 해결

### 환경 변수 누락 오류

```
ERROR: 환경 변수가 설정되지 않았습니다
```

→ `.env.prod` 파일이 올바른 위치에 있고 모든 필수 변수가 설정되어 있는지 확인

### Docker Hub 인증 오류

```
ERROR: denied: requested access to the resource is denied
```

→ GitHub Secrets에 `DOCKERHUB_USERNAME`과 `DOCKERHUB_TOKEN`이 올바르게 설정되어 있는지 확인

### AWS 권한 오류

```
ERROR: User is not authorized to perform: codedeploy:CreateDeployment
```

→ IAM 사용자에 필요한 권한이 부여되어 있는지 확인

## 7. 참고 문서

- [환경 변수 상세 명세](./environment-variables.md)
- [배포 계획서](./deployment-plan.md)
- [EC2 초기 설정 가이드](./ec2-setup-guide.md)
- [문제 해결 가이드](./troubleshooting.md)

