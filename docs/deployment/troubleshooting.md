# 문제 해결 가이드

## 일반적인 문제

### 1. Docker 컨테이너가 시작되지 않음

#### 증상
- 컨테이너가 즉시 종료됨
- `docker ps`에 컨테이너가 표시되지 않음

#### 원인 및 해결
1. **환경 변수 누락**
   ```bash
   # 컨테이너 로그 확인
   docker logs <container-id>
   
   # 필수 환경 변수 확인
   docker exec <container-id> env | grep -E "SQS_QUEUE_URL|S3_BUCKET_NAME|SPRING_API_BASE_URL"
   ```

2. **권한 문제**
   ```bash
   # Docker 소켓 권한 확인
   ls -la /var/run/docker.sock
   
   # 사용자 권한 확인
   id
   ```

3. **포트 충돌**
   ```bash
   # 포트 사용 확인
   netstat -tulpn | grep <port>
   ```

### 2. SQS 메시지를 수신하지 못함

#### 증상
- 로그에 "No messages received" 반복
- 작업이 처리되지 않음

#### 원인 및 해결
1. **SQS 큐 URL 확인**
   ```bash
   # 환경 변수 확인
   echo $SQS_QUEUE_URL
   
   # AWS CLI로 큐 확인
   aws sqs get-queue-attributes --queue-url $SQS_QUEUE_URL
   ```

2. **IAM 권한 확인**
   - EC2 Instance Profile에 `sqs:ReceiveMessage`, `sqs:DeleteMessage` 권한이 있는지 확인
   - IAM 정책 예시:
     ```json
     {
       "Effect": "Allow",
       "Action": [
         "sqs:ReceiveMessage",
         "sqs:DeleteMessage",
         "sqs:ChangeMessageVisibility"
       ],
       "Resource": "arn:aws:sqs:ap-northeast-2:123456789012:video-processing-queue"
     }
     ```

3. **가시성 타임아웃 확인**
   - 작업 처리 시간이 가시성 타임아웃보다 길면 메시지가 다시 큐에 나타남
   - `VISIBILITY_TIMEOUT` 환경 변수 증가

### 3. S3 파일 다운로드/업로드 실패

#### 증상
- "Failed to download" 또는 "Failed to upload" 에러
- `ClientError` 예외 발생

#### 원인 및 해결
1. **S3 버킷 이름 확인**
   ```bash
   # 환경 변수 확인
   echo $S3_BUCKET_NAME
   
   # 버킷 존재 확인
   aws s3 ls s3://$S3_BUCKET_NAME
   ```

2. **IAM 권한 확인**
   - `s3:GetObject`, `s3:PutObject` 권한 필요
   - 버킷 정책 확인

3. **네트워크 연결 확인**
   ```bash
   # S3 엔드포인트 연결 테스트
   curl -I https://s3.ap-northeast-2.amazonaws.com
   ```

### 4. Redis 연결 실패

#### 증상
- "Connection refused" 또는 "Timeout" 에러
- 진행률이 전송되지 않음

#### 원인 및 해결
1. **Redis 호스트 및 포트 확인**
   ```bash
   # 환경 변수 확인
   echo $REDIS_HOST
   echo $REDIS_PORT
   
   # 연결 테스트
   redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
   ```

2. **네트워크 연결 확인**
   ```bash
   # 포트 열림 확인
   telnet $REDIS_HOST $REDIS_PORT
   
   # 보안 그룹 확인 (EC2)
   # Redis 포트(6379)가 열려있는지 확인
   ```

3. **Redis 인증 확인**
   - 비밀번호가 필요한 경우 `REDIS_PASSWORD` 환경 변수 설정
   - Redis 설정 파일에서 `requirepass` 확인

### 5. Spring API 콜백 실패

#### 증상
- "Connection timeout" 또는 "HTTP 500" 에러
- 작업은 완료되었지만 콜백이 전송되지 않음

#### 원인 및 해결
1. **Spring API URL 확인**
   ```bash
   # 환경 변수 확인
   echo $SPRING_API_BASE_URL
   
   # 연결 테스트
   curl -I $SPRING_API_BASE_URL/health
   ```

2. **네트워크 연결 확인**
   - EC2-B (Spring 서버)와의 네트워크 연결 확인
   - 보안 그룹에서 HTTP/HTTPS 포트 허용 확인

3. **타임아웃 설정 확인**
   - `SPRING_API_TIMEOUT` 환경 변수 증가
   - 네트워크 지연 고려

### 6. Docker 이미지 빌드 실패

#### 증상
- GitHub Actions에서 빌드 실패
- "Failed to build" 에러

#### 원인 및 해결
1. **의존성 설치 실패**
   - `requirements.txt` 확인
   - PyTorch CUDA 버전 호환성 확인
   - Dockerfile의 Python 버전 확인

2. **메모리 부족**
   - GitHub Actions 러너 메모리 확인
   - Docker 빌드 시 메모리 제한 설정

3. **네트워크 문제**
   - PyPI 또는 PyTorch 다운로드 서버 연결 확인
   - 프록시 설정 필요 시 설정

### 7. CodeDeploy 배포 실패

#### 증상
- CodeDeploy 배포가 실패 상태
- "Installation failed" 에러

#### 원인 및 해결
1. **스크립트 실행 권한**
   ```bash
   # 스크립트 권한 확인
   ls -la scripts/deploy/
   
   # 실행 권한 부여
   chmod +x scripts/deploy/*.sh
   ```

2. **환경 변수 누락**
   - `appspec.yml`에서 환경 변수 설정 확인
   - CodeDeploy 환경 변수 설정 확인

3. **Docker 이미지 Pull 실패**
   - ECR 접근 권한 확인
   - 이미지 태그 확인
   - 네트워크 연결 확인

## 로그 확인 방법

### Docker 컨테이너 로그
```bash
# 실시간 로그 확인
docker logs -f <container-id>

# 최근 100줄 확인
docker logs --tail 100 <container-id>

# 특정 시간 이후 로그
docker logs --since 2024-01-01T00:00:00 <container-id>
```

### CloudWatch Logs
```bash
# 로그 그룹 확인
aws logs describe-log-groups --log-group-name-prefix /aws/ec2/echoshot-worker

# 로그 스트림 확인
aws logs describe-log-streams --log-group-name /aws/ec2/echoshot-worker

# 로그 조회
aws logs tail /aws/ec2/echoshot-worker --follow
```

### CodeDeploy 로그
```bash
# CodeDeploy Agent 로그
tail -f /var/log/aws/codedeploy-agent/codedeploy-agent.log

# 배포 로그
tail -f /opt/codedeploy-agent/deployment-root/deployment-logs/codedeploy-agent-deployments.log
```

## 성능 문제

### 1. 작업 처리 속도가 느림

#### 원인 및 해결
1. **워커 수 증가**
   - `WORKER_COUNT` 환경 변수 증가
   - 인스턴스 사양 확인 (CPU, 메모리)

2. **GPU 사용**
   - GPU 인스턴스 사용 (g4dn.xlarge 이상)
   - Docker에서 GPU 접근 설정
   - PyTorch CUDA 버전 확인

3. **모델 최적화**
   - 빠른 모델 사용 (FSRCNN, EDSR)
   - RealESRGAN은 품질 우선 시에만 사용

### 2. 메모리 부족

#### 증상
- "Out of memory" 에러
- 컨테이너가 종료됨

#### 원인 및 해결
1. **워커 수 감소**
   - `WORKER_COUNT` 환경 변수 감소
   - 동시 처리 작업 수 제한

2. **인스턴스 사양 증가**
   - 더 큰 인스턴스 타입 사용
   - 메모리 최적화 인스턴스 사용

3. **임시 파일 정리**
   - 작업 완료 후 임시 파일 즉시 삭제
   - 디스크 공간 모니터링

## 모니터링 및 알림

### CloudWatch 알람 설정
```bash
# SQS 큐 깊이 알람
aws cloudwatch put-metric-alarm \
  --alarm-name sqs-queue-depth-high \
  --metric-name ApproximateNumberOfMessagesVisible \
  --namespace AWS/SQS \
  --statistic Average \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold
```

### Health Check 스크립트
```bash
# Health check 실행
./scripts/deploy/validate_service.sh

# 결과 확인
echo $?
# 0: 정상, 1: 비정상
```

## 지원 및 문의

문제가 지속되면 다음 정보를 수집하여 문의:

1. 에러 메시지 전체
2. 로그 파일 (최근 100줄)
3. 환경 변수 설정 (민감 정보 제외)
4. 시스템 정보 (인스턴스 타입, OS 버전)
5. 재현 단계

