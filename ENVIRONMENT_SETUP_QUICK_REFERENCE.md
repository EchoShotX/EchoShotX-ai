# 환경 변수 및 GitHub Secrets 빠른 참조

베이스 이미지 분리 배포 방식을 사용하기 위한 필수 환경 변수와 GitHub Secrets 빠른 참조입니다.

## 빠른 설정 체크리스트

### 1. GitHub Secrets (필수)

```
✅ DOCKERHUB_USERNAME          # Docker Hub 사용자명
✅ DOCKERHUB_TOKEN             # Docker Hub Access Token
✅ AWS_ACCESS_KEY_ID           # AWS 액세스 키
✅ AWS_SECRET_ACCESS_KEY       # AWS 시크릿 키
✅ CODE_DEPLOY_APPLICATION_NAME    # CodeDeploy 애플리케이션 이름
✅ CODE_DEPLOY_DEPLOYMENT_GROUP    # CodeDeploy 배포 그룹 이름
```

### 2. EC2 .env.prod (필수)

```bash
# AWS 설정
AWS_REGION=ap-northeast-2
SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/YOUR_ACCOUNT_ID/video-processing-queue
S3_BUCKET_NAME=echoshot-videos

# Spring API
SPRING_API_BASE_URL=https://api.echoshot.com

# Docker 이미지
DOCKER_IMAGE=echoshot/echoshot-worker:gpu-fsrcnn-t4
```

## 상세 가이드

자세한 설정 방법은 [docs/deployment/environment-setup.md](docs/deployment/environment-setup.md)를 참조하세요.

## 설정 위치

- **GitHub Secrets**: Repository → Settings → Secrets and variables → Actions
- **EC2 .env.prod**: `/opt/echoshot-worker/.env.prod`

