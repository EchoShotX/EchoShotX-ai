# 환경 변수 명세

## 개요

EchoShotX AI Worker 서비스는 Pydantic Settings를 사용하여 환경 변수를 관리합니다. 환경 변수는 `.env.{환경}` 파일 또는 시스템 환경 변수로 설정할 수 있습니다.

## 환경 변수 목록

### AWS 설정

#### AWS_REGION
- **설명**: AWS 리전
- **타입**: `string`
- **기본값**: `ap-northeast-2`
- **필수**: 아니오
- **예시**: `ap-northeast-2`

#### SQS_QUEUE_URL
- **설명**: SQS 큐 URL
- **타입**: `string`
- **기본값**: 없음
- **필수**: 예
- **예시**: `https://sqs.ap-northeast-2.amazonaws.com/123456789012/video-processing-queue`

#### S3_BUCKET_NAME
- **설명**: S3 버킷 이름
- **타입**: `string`
- **기본값**: 없음
- **필수**: 예
- **예시**: `echoshot-videos`

### Spring API 설정

#### SPRING_API_BASE_URL
- **설명**: Spring 서버의 기본 URL
- **타입**: `string`
- **기본값**: 없음
- **필수**: 예
- **예시**: `https://api.echoshot.com`

#### SPRING_API_TIMEOUT
- **설명**: API 요청 타임아웃 (초 단위)
- **타입**: `integer`
- **기본값**: `30`
- **필수**: 아니오
- **예시**: `30`

### Worker 설정

#### WORKER_COUNT
- **설명**: 동시에 처리할 워커 수
- **타입**: `integer`
- **기본값**: `4`
- **필수**: 아니오
- **예시**: `4`

#### MAX_RETRIES
- **설명**: 작업 재시도 횟수
- **타입**: `integer`
- **기본값**: `3`
- **필수**: 아니오
- **예시**: `3`

#### VISIBILITY_TIMEOUT
- **설명**: SQS 메시지 가시성 타임아웃 (초 단위)
- **타입**: `integer`
- **기본값**: `300`
- **필수**: 아니오
- **예시**: `300`

### 비디오 처리 설정

#### TEMP_DIR
- **설명**: 임시 비디오 저장 디렉토리
- **타입**: `string` (Path)
- **기본값**: `/tmp/video_processing`
- **필수**: 아니오
- **예시**: `/tmp/video_processing`

#### MAX_VIDEO_SIZE_MB
- **설명**: 업로드 가능한 최대 비디오 크기 (MB)
- **타입**: `integer`
- **기본값**: `500`
- **필수**: 아니오
- **예시**: `500`

### Redis 설정 (구현 필요)

#### REDIS_HOST
- **설명**: Redis 호스트 주소
- **타입**: `string`
- **기본값**: 없음
- **필수**: 예 (Redis Pub/Sub 사용 시)
- **예시**: `redis.ec2-b.internal` 또는 `192.168.1.100`

#### REDIS_PORT
- **설명**: Redis 포트
- **타입**: `integer`
- **기본값**: `6379`
- **필수**: 아니오
- **예시**: `6379`

#### REDIS_PASSWORD
- **설명**: Redis 비밀번호
- **타입**: `string`
- **기본값**: 없음
- **필수**: 아니오 (Redis 인증 사용 시)
- **예시**: `your-redis-password`

#### REDIS_DB
- **설명**: Redis 데이터베이스 번호
- **타입**: `integer`
- **기본값**: `0`
- **필수**: 아니오
- **예시**: `0`

### 로깅 설정

#### LOG_LEVEL
- **설명**: 로그 레벨
- **타입**: `string`
- **기본값**: `INFO`
- **필수**: 아니오
- **가능한 값**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **예시**: `INFO`

#### APP_ENV
- **설명**: 애플리케이션 환경 (프로덕션만 사용)
- **타입**: `string`
- **기본값**: `prod`
- **필수**: 아니오
- **값**: `prod` (프로덕션 환경만 사용)
- **예시**: `prod`
- **참고**: 항상 `.env.prod` 파일을 사용합니다.

## 환경 변수 설정 방법

### 1. 환경 변수 파일 사용

프로젝트 루트에 `.env.prod` 파일을 생성:

```bash
# AWS 설정
AWS_REGION=ap-northeast-2
SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/123456789012/video-processing-queue
S3_BUCKET_NAME=echoshot-videos

# Spring API 설정
SPRING_API_BASE_URL=https://api.echoshot.com
SPRING_API_TIMEOUT=30

# Worker 설정
WORKER_COUNT=4
MAX_RETRIES=3
VISIBILITY_TIMEOUT=300

# Redis 설정
REDIS_HOST=redis.ec2-b.internal
REDIS_PORT=6379

# 로깅 설정
LOG_LEVEL=INFO
APP_ENV=prod
```

### 2. 시스템 환경 변수 사용

Docker 컨테이너 실행 시 환경 변수 주입:

```bash
docker run -e SQS_QUEUE_URL=... -e S3_BUCKET_NAME=... ...
```

### 3. CodeDeploy를 통한 설정

`after_install.sh` 스크립트에서 환경 변수 파일 생성:

```bash
cat > /opt/echoshot-worker/.env.prod <<EOF
AWS_REGION=${AWS_REGION}
SQS_QUEUE_URL=${SQS_QUEUE_URL}
...
EOF
```

### 4. AWS Systems Manager Parameter Store (옵션)

민감한 정보는 Parameter Store에 저장하고 런타임에 조회:

```python
import boto3

ssm = boto3.client('ssm')
parameter = ssm.get_parameter(Name='/echoshot/worker/redis-password', WithDecryption=True)
redis_password = parameter['Parameter']['Value']
```

## 프로덕션 환경 설정

프로젝트는 프로덕션 환경만 사용합니다.

### 프로덕션 환경 (prod)
- `.env.prod` 파일 사용
- 로그 레벨: `INFO` 또는 `WARNING`
- 워커 수: 인스턴스 사양에 맞게 조정 (기본값: 4)
- 최적화된 타임아웃 설정

## 보안 고려사항

### 민감한 정보 관리
- 비밀번호, API 키는 환경 변수 파일에 직접 저장하지 않음
- AWS Secrets Manager 또는 Parameter Store 사용 권장
- 환경 변수 파일은 `.gitignore`에 추가

### 권한 관리
- 환경 변수 파일 권한: `600` (소유자만 읽기/쓰기)
- Docker 컨테이너 내부에서만 접근 가능하도록 설정

## 검증

환경 변수 설정 후 다음 명령으로 검증:

```bash
python -c "from echoshot_ai_server.config.settings import get_settings; s = get_settings(); print(s.dict())"
```

## 문제 해결

### 환경 변수를 찾을 수 없음
- `.env.prod` 파일이 존재하는지 확인
- 파일 경로가 올바른지 확인
- 환경 변수 이름이 대문자인지 확인

### 타입 오류
- 숫자 타입은 문자열이 아닌 정수로 설정
- 불리언 타입은 `true`/`false` 또는 `1`/`0` 사용

### 기본값이 적용되지 않음
- 환경 변수가 빈 문자열로 설정되어 있지 않은지 확인
- `None` 값이 필요한 경우 환경 변수를 설정하지 않음

