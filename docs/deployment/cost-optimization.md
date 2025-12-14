# EC2 비용 최적화 가이드

## 개요

이 프로젝트는 GPU를 사용하여 영상 처리를 수행하므로 EC2 인스턴스 비용이 주요 비용 요소입니다. 비용을 최소화하면서 성능을 유지하는 방법을 안내합니다.

## GPU 인스턴스 타입 비교

### 권장 인스턴스 타입 (ap-northeast-2 기준, 2024년)

| 인스턴스 타입 | GPU | vCPU | 메모리 | 시간당 비용 (USD) | 특징 |
|-------------|-----|------|--------|------------------|------|
| **g4dn.xlarge** | 1x T4 (16GB) | 4 | 16GB | ~$0.526 | **가장 경제적**, 권장 |
| g4dn.2xlarge | 1x T4 (16GB) | 8 | 32GB | ~$0.752 | 더 많은 CPU/메모리 |
| g5.xlarge | 1x A10G (24GB) | 4 | 16GB | ~$1.006 | 더 강력한 GPU |
| g5.2xlarge | 1x A10G (24GB) | 8 | 32GB | ~$1.212 | 고성능 필요 시 |

### 비용 비교 (월간, 24/7 운영 기준)

- **g4dn.xlarge**: 약 $380/월
- **g4dn.2xlarge**: 약 $540/월
- **g5.xlarge**: 약 $725/월
- **g5.2xlarge**: 약 $870/월

## 비용 최적화 전략

### 1. 적절한 인스턴스 타입 선택

#### 권장: g4dn.xlarge
- **이유**: 
  - T4 GPU (16GB VRAM)로 Real-ESRGAN 충분히 실행 가능
  - FSRCNN, EDSR 모델은 더 적은 VRAM으로도 동작
  - 가격 대비 성능 우수
- **사용 모델**:
  - `fast` (FSRCNN): VRAM Low → g4dn.xlarge 적합
  - `balanced` (EDSR): VRAM Medium → g4dn.xlarge 적합
  - `quality` (RealESRGAN): VRAM High → g4dn.xlarge 적합 (타일 크기 조정)

#### 업그레이드 고려 사항
- 동시 작업이 많고 처리 시간이 중요한 경우: g4dn.2xlarge
- 최고 품질이 필수이고 시간 여유가 있는 경우: g5.xlarge

### 2. Spot 인스턴스 사용 (최대 90% 절감)

#### Spot 인스턴스란?
- AWS의 미사용 EC2 용량을 할인된 가격에 제공
- 중단될 수 있지만 비용이 매우 저렴

#### Spot 인스턴스 사용 방법

```bash
# EC2 Launch Template 또는 Auto Scaling Group에서 설정
Instance Type: g4dn.xlarge
Capacity Type: Spot
Max Price: On-Demand 가격의 100% (또는 더 낮게 설정)
```

#### Spot 인스턴스 비용 (예시)
- g4dn.xlarge On-Demand: $0.526/시간
- g4dn.xlarge Spot: $0.158/시간 (약 70% 할인)
- **월간 비용**: 약 $114/월 (24/7 운영 시)

#### Spot 인스턴스 주의사항
- **중단 가능성**: AWS가 용량이 필요하면 2분 전에 알림 후 종료
- **대응 방법**:
  - CodeDeploy로 자동 재배포 설정
  - 작업 진행률을 Redis에 저장하여 중단 시 재시작 가능
  - SQS 메시지 가시성 타임아웃으로 재처리 보장

### 3. Auto Scaling으로 필요할 때만 실행

#### 전략: 작업이 있을 때만 인스턴스 실행

```yaml
# Auto Scaling Group 설정
Min Size: 0
Desired Size: 0
Max Size: 2

# CloudWatch 알람 기반 스케일링
- SQS 큐 깊이 > 10 → 인스턴스 시작
- SQS 큐 깊이 = 0 (5분 지속) → 인스턴스 종료
```

#### 예상 비용 절감
- 작업이 없는 시간 (밤, 주말 등): $0
- 작업이 있는 시간만 과금
- **예시**: 하루 8시간만 사용 시 → 월간 약 $127 (g4dn.xlarge On-Demand)

### 4. 모델 프로필 선택으로 성능/비용 균형

#### 모델별 특징

| 모델 프로필 | 모델 | VRAM | 처리 속도 | 비용 영향 |
|------------|------|------|----------|----------|
| `fast` | FSRCNN | Low | 매우 빠름 | 인스턴스 작게 가능 |
| `balanced` | EDSR | Medium | 빠름 | **권장** (균형) |
| `quality` | RealESRGAN | High | 느림 | 더 큰 인스턴스 필요 |

#### 권장 설정

```bash
# .env.prod
# 빠른 처리 우선 (비용 절감)
MODEL_PROFILE=fast  # 또는 balanced

# 최고 품질 필요 시
MODEL_PROFILE=quality  # 더 큰 인스턴스 필요
```

### 5. 워커 수 조정

#### 워커 수와 인스턴스 사양 관계

```bash
# g4dn.xlarge (4 vCPU, 16GB RAM)
WORKER_COUNT=2  # 안정적
WORKER_COUNT=4  # 최대 (CPU 집약적)

# g4dn.2xlarge (8 vCPU, 32GB RAM)
WORKER_COUNT=4  # 안정적
WORKER_COUNT=8  # 최대
```

#### 비용 최적화
- 워커 수를 줄이면 → 더 작은 인스턴스 사용 가능
- 워커 수를 늘리면 → 더 큰 인스턴스 필요하지만 처리량 증가

### 6. 예약 인스턴스 (RI) 사용

#### 1년 예약 인스턴스
- **할인율**: 약 40-50%
- **조건**: 1년간 사용 약속
- **예상 비용**: g4dn.xlarge → 약 $190-230/월

#### 3년 예약 인스턴스
- **할인율**: 약 60-70%
- **조건**: 3년간 사용 약속
- **예상 비용**: g4dn.xlarge → 약 $114-150/월

#### 언제 사용?
- ✅ 작업량이 일정하고 장기 운영 예정
- ✅ 비용 예측 가능성 필요
- ❌ 작업량이 불규칙하거나 단기 프로젝트

### 7. 하이브리드 전략 (권장)

#### CPU + GPU 조합

```
작업 큐 분리:
- 빠른 작업 (FSRCNN) → CPU 인스턴스 (c5.2xlarge)
- 품질 작업 (RealESRGAN) → GPU 인스턴스 (g4dn.xlarge Spot)
```

#### 비용 비교
- GPU만 사용: g4dn.xlarge 24/7 = $380/월
- 하이브리드: 
  - CPU (c5.2xlarge) 24/7 = $340/월
  - GPU (g4dn.xlarge Spot) 필요시만 = $50/월
  - **총합**: 약 $390/월 (유연성 증가)

## 실제 비용 시나리오

### 시나리오 1: 최소 비용 (권장)

```
인스턴스: g4dn.xlarge Spot
Auto Scaling: 작업 있을 때만 실행
모델: fast 또는 balanced
워커: 2-4개

예상 비용:
- 작업 시간: 하루 4시간
- Spot 할인: 70%
- 월간: 약 $50-80
```

### 시나리오 2: 균형 (권장)

```
인스턴스: g4dn.xlarge On-Demand
Auto Scaling: 작업 있을 때만 실행
모델: balanced
워커: 4개

예상 비용:
- 작업 시간: 하루 8시간
- 월간: 약 $127
```

### 시나리오 3: 고가용성

```
인스턴스: g4dn.xlarge On-Demand
Auto Scaling: 최소 1개 항상 실행
모델: balanced
워커: 4개

예상 비용:
- 24/7 운영
- 월간: 약 $380
```

### 시나리오 4: 최고 품질

```
인스턴스: g5.xlarge On-Demand
Auto Scaling: 작업 있을 때만 실행
모델: quality (RealESRGAN)
워커: 2-4개

예상 비용:
- 작업 시간: 하루 8시간
- 월간: 약 $240
```

## 모니터링 및 비용 추적

### CloudWatch 비용 알람 설정

```bash
# 월간 예산 알람
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

### 비용 최적화 체크리스트

- [ ] 적절한 인스턴스 타입 선택 (g4dn.xlarge 권장)
- [ ] Spot 인스턴스 사용 검토
- [ ] Auto Scaling 설정 (필요할 때만 실행)
- [ ] 모델 프로필 최적화 (fast/balanced 권장)
- [ ] 워커 수 조정
- [ ] 예약 인스턴스 고려 (장기 운영 시)
- [ ] CloudWatch 비용 모니터링 설정

## 추가 최적화 팁

### 1. EBS 볼륨 최적화
- **gp3** 사용 (gp2 대비 20% 저렴)
- 필요한 용량만 할당
- 불필요한 스냅샷 정리

### 2. 데이터 전송 비용 절감
- 같은 리전 내 통신 (S3, SQS, Redis)
- CloudFront 사용 (대용량 다운로드 시)

### 3. 모니터링 비용
- CloudWatch Logs 보관 기간 조정 (7일 권장)
- 불필요한 메트릭 제거

## 결론

**최소 비용 구성 (권장)**:
- 인스턴스: **g4dn.xlarge Spot**
- Auto Scaling: 작업 있을 때만 실행
- 모델: **balanced** (FSRCNN 또는 EDSR)
- 예상 비용: **월 $50-150** (사용량에 따라)

**균형 구성**:
- 인스턴스: **g4dn.xlarge On-Demand**
- Auto Scaling: 작업 있을 때만 실행
- 모델: **balanced**
- 예상 비용: **월 $100-200**

## 참고 자료

- [EC2 가격](https://aws.amazon.com/ec2/pricing/)
- [Spot 인스턴스 가격](https://aws.amazon.com/ec2/spot/pricing/)
- [Auto Scaling 가이드](https://docs.aws.amazon.com/autoscaling/)

