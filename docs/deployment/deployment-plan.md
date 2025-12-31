# 배포 계획서

## 개요

EchoShotX AI Worker 서비스는 AWS EC2, CodeDeploy, Docker를 활용하여 배포되는 Python 기반 비디오 처리 워커 서비스입니다. GitHub Actions를 통한 CI/CD 파이프라인이 구축되어 자동화된 배포가 가능합니다.

## 배포 목표

1. **자동화된 배포**: GitHub Actions를 통한 CI/CD 파이프라인 구축
2. **안정적인 서비스 운영**: Docker 컨테이너 기반 격리된 환경
3. **확장 가능한 아키텍처**: 멀티프로세싱 기반 워커 풀
4. **모니터링 및 로깅**: CloudWatch 연동을 통한 운영 가시성 확보

## 배포 아키텍처

### 베이스 이미지 분리 배포 전략

이 프로젝트는 **베이스 이미지 분리 전략**을 사용하여 배포 시간을 최적화합니다:

1. **베이스 이미지**: OpenCV CUDA 빌드가 포함된 이미지 (한 번만 빌드, Docker Hub에 저장)
2. **애플리케이션 이미지**: 베이스 이미지를 사용하여 애플리케이션 코드만 추가 (매 배포마다 빠르게 빌드)

**장점**:
- 배포 시간 단축: 2시간 → 10-15분 (90% 단축)
- EC2 리소스 절약: GPU 인스턴스가 작업 처리에 집중
- 일관성 보장: 동일한 베이스 이미지로 환경 통일

### 전체 흐름

```
베이스 이미지 빌드 (초기 설정, 한 번만)
    │
    └─> EC2 GPU 인스턴스에서 빌드
        │
        └─> Docker Hub에 푸시
            └─> echoshot/opencv-cuda-t4:4.10.0

정기 배포 흐름
    │
    └─> GitHub Repository
        │
        └─> GitHub Actions (CI/CD)
            │
            ├─> 1. 코드 체크아웃
            ├─> 2. 테스트 실행 (pytest)
            ├─> 3. 베이스 이미지 Pull (Docker Hub)
            ├─> 4. 애플리케이션 이미지 빌드 (5-10분)
            ├─> 5. Docker Hub에 이미지 푸시
            └─> 6. CodeDeploy 배포 트리거
                    │
                    └─> CodeDeploy (배포 오케스트레이션)
                            │
                            └─> EC2-A (Python Worker)
                                    │
                                    ├─> Docker Hub에서 이미지 Pull (1-2분)
                                    ├─> Docker Container 실행
                                    │       ├─> Python Application
                                    │       ├─> SQS Polling
                                    │       ├─> Redis Pub/Sub
                                    │       └─> Spring HTTP Callback
                                    │
                                    └─> Systemd Service (옵션)
```

## 배포 단계

### 1. 사전 준비사항

#### AWS 리소스
- EC2 인스턴스 (Python Worker용, GPU 인스턴스 권장: g4dn.xlarge)
- CodeDeploy 애플리케이션 및 배포 그룹
- SQS 큐 (작업 큐)
- S3 버킷 (영상 파일 저장)
- Redis 인스턴스 (다른 EC2에 위치)

**비용 최적화**: EC2 인스턴스 스펙 및 비용 최적화 전략은 [cost-optimization.md](./cost-optimization.md) 참조

#### GitHub 설정
- GitHub Secrets 설정 (AWS 인증 정보, ECR 정보 등)
- 워크플로우 파일 작성

#### EC2 인스턴스 설정
- Docker 설치
- CodeDeploy Agent 설치
- IAM 역할 설정 (S3, SQS, ECR 접근 권한)

### 2. 초기 설정: 베이스 이미지 빌드

베이스 이미지는 OpenCV CUDA 빌드가 포함된 이미지로, **한 번만 빌드**하고 Docker Hub에 저장합니다.

#### 베이스 이미지 빌드 방법

```bash
# EC2 GPU 인스턴스에서 실행
cd /path/to/echoshotx-ai
./scripts/build-base-image.sh
```

이 스크립트는:
1. OpenCV CUDA를 빌드합니다 (약 1.5-2시간)
2. 베이스 이미지를 생성합니다
3. OpenCV CUDA 빌드 검증을 수행합니다
4. Docker Hub에 푸시합니다

**참고**: 베이스 이미지는 OpenCV 업데이트 시에만 재빌드하면 됩니다 (약 한 달에 1회 이하).

### 3. CI/CD 파이프라인

#### GitHub Actions (CI)
1. **코드 체크아웃**: 소스 코드 다운로드
2. **테스트 실행**: pytest를 통한 단위 테스트 및 통합 테스트
3. **베이스 이미지 Pull**: Docker Hub에서 베이스 이미지 다운로드 (약 3분)
4. **애플리케이션 이미지 빌드**: 베이스 이미지에 애플리케이션 코드 추가 (약 5-10분)
5. **Docker Hub 푸시**: 빌드된 이미지를 Docker Hub에 업로드
6. **CodeDeploy 트리거**: 배포 시작

#### CodeDeploy (CD)
1. **Before Install**: 시스템 의존성 확인 (Docker, Python, FFmpeg)
2. **Install**: Docker Hub에서 이미지 Pull만 수행 (약 1-2분, 빌드 제거)
3. **After Install**: 환경 변수 설정, 임시 디렉토리 생성, 모델 가중치 확인
4. **Application Start**: Docker 컨테이너 실행, Health check
5. **Application Stop**: Graceful shutdown (진행 중인 작업 완료 대기)

### 4. 배포 프로세스

#### 자동 배포
- `main`/`master` 브랜치에 push 시 자동 배포
- Pull Request는 테스트만 실행
- **배포 시간**: 약 10-15분 (GitHub Actions 빌드 10-12분 + EC2 Pull 1-2분)

#### 수동 배포
- GitHub Actions 워크플로우에서 `workflow_dispatch` 사용

#### 롤백
- CodeDeploy 콘솔에서 이전 배포로 롤백
- GitHub Actions에서 배포 실패 시 자동 롤백 트리거
- Docker Hub의 이전 태그 사용 가능

### 5. 배포 시간 개선 내역

**이전 방식**:
- EC2에서 전체 빌드: 약 2시간
- OpenCV CUDA 빌드: 약 1.5시간 (매 배포마다)

**개선된 방식**:
- 베이스 이미지: 한 번만 빌드 (초기 설정)
- 애플리케이션 이미지: GitHub Actions에서 자동 빌드 (약 10-12분)
- EC2 Pull: 약 1-2분
- **총 배포 시간**: 약 10-15분 (90% 단축)

## 환경 변수 관리

### 필수 환경 변수
- AWS 설정: `AWS_REGION`, `SQS_QUEUE_URL`, `S3_BUCKET_NAME`
- Spring API: `SPRING_API_BASE_URL`, `SPRING_API_TIMEOUT`
- Worker 설정: `WORKER_COUNT`, `MAX_RETRIES`, `VISIBILITY_TIMEOUT`
- Redis 설정: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` (옵션)

자세한 내용은 [environment-variables.md](./environment-variables.md) 참조

## 모니터링 및 로깅

### CloudWatch Logs
- Docker 컨테이너 로그를 CloudWatch로 전송
- 로그 그룹: `/aws/ec2/echoshot-worker`

### 메트릭 수집
- SQS 메시지 처리량
- 작업 완료율
- 워커 프로세스 상태

### Health Check
- Docker 컨테이너 상태 확인
- SQS 연결 상태 확인
- Redis 연결 상태 확인

## 보안 고려사항

### IAM 역할
- EC2 Instance Profile 사용
- 최소 권한 원칙 적용
- S3, SQS, ECR, Redis 접근 권한만 부여

### Docker 보안
- Non-root 사용자로 실행
- 필요한 포트만 노출
- 보안 스캔 수행

### 네트워크 보안
- 보안 그룹 설정
- VPC 내부 통신만 허용

## 롤백 전략

### 자동 롤백
- Health check 실패 시 자동 롤백
- 배포 실패 시 이전 버전으로 복구

### 수동 롤백
- CodeDeploy 콘솔에서 롤백 실행
- 이전 Docker 이미지 태그 사용

### 데이터 보호
- SQS 메시지 가시성 타임아웃 관리
- 진행 중인 작업 완료 대기

## 문제 해결

자세한 문제 해결 가이드는 [troubleshooting.md](./troubleshooting.md) 참조

## 배포 체크리스트

### 배포 전
- [ ] 베이스 이미지 빌드 완료 (초기 설정 시만)
- [ ] 환경 변수 설정 확인
- [ ] AWS 리소스 준비 완료
- [ ] GitHub Secrets 설정 완료 (DOCKERHUB_USERNAME, DOCKERHUB_TOKEN)
- [ ] 테스트 통과 확인

### 배포 중
- [ ] GitHub Actions 워크플로우 실행 확인
- [ ] CodeDeploy 배포 상태 모니터링
- [ ] 로그 확인

### 배포 후
- [ ] Health check 통과 확인
- [ ] SQS 메시지 수신 확인
- [ ] 작업 처리 확인
- [ ] 모니터링 대시보드 확인

## 참고 문서

- [환경 변수 및 GitHub Secrets 설정 가이드](./environment-setup.md) ⭐ 새로 추가
- [배포 아키텍처 상세](./deployment-architecture.md)
- [환경 변수 명세](./environment-variables.md)
- [문제 해결 가이드](./troubleshooting.md)
- [CI/CD 가이드](./cicd-guide.md)
- [비용 최적화 가이드](./cost-optimization.md)
- [EC2 초기 설정 가이드](./ec2-setup-guide.md)

