# 비디오 업스케일링 시스템 가이드

## 목차
1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [주요 기능](#주요-기능)
4. [처리 파이프라인](#처리-파이프라인)
5. [GPU/CPU 선택 가이드](#gpucpu-선택-가이드)
6. [메모리 최적화](#메모리-최적화)
7. [구독 서비스 연동](#구독-서비스-연동)
8. [로컬 테스트](#로컬-테스트)
9. [배포 가이드](#배포-가이드)
10. [트러블슈팅](#트러블슈팅)

---

## 개요

EchoShotX AI 서버의 비디오 업스케일링 시스템은 **Real-ESRGAN** 딥러닝 모델을 사용하여 저해상도 비디오를 고해상도로 변환합니다.

### 핵심 특징
- ✅ **GPU/CPU 선택적 사용** - 구독 서비스에 따라 처리 방식 선택
- ✅ **메모리 최적화** - 타일링 기법으로 대용량 비디오 처리
- ✅ **오디오 자동 처리** - 오디오 트랙 자동 감지 및 병합
- ✅ **진행률 추적** - 실시간 처리 상태 로깅
- ✅ **안정성** - 예외 처리 및 리소스 정리

---

## 아키텍처

```
┌─────────────┐
│   Spring    │
│   Backend   │
└──────┬──────┘
       │ Job 생성 (SQS)
       ▼
┌─────────────┐
│   AWS SQS   │ ◄─── Job Queue
└──────┬──────┘
       │ 폴링
       ▼
┌─────────────────────────────────┐
│   Python Worker (EC2)           │
│                                 │
│  ┌──────────────────────────┐  │
│  │  JobProcessor            │  │
│  │  - SQS 폴링              │  │
│  │  - Job 파싱              │  │
│  └────────┬─────────────────┘  │
│           │                     │
│           ▼                     │
│  ┌──────────────────────────┐  │
│  │  UpscaleTask             │  │
│  │  1. S3에서 비디오 다운로드 │  │
│  │  2. 업스케일 처리         │  │
│  │  3. S3로 결과 업로드      │  │
│  │  4. Spring에 완료 통보    │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   AWS S3    │ ◄─── 결과 저장
└─────────────┘
```

---

## 주요 기능

### 1. Real-ESRGAN 업스케일링
- **모델**: RealESRGAN_x4plus (4배 업스케일 기본 모델)
- **지원 배율**: 2x, 4x
- **네트워크**: RRDB (Residual in Residual Dense Block)
- **특징**: 자연스러운 고화질 복원, 노이즈 제거, 디테일 강화

### 2. 디바이스 선택 (GPU/CPU)

#### GPU 모드 (프리미엄)
```python
{
  "device": "gpu",
  "scale_factor": 4
}
```
- **장점**: 빠른 처리 속도 (CPU 대비 10~20배)
- **요구사항**: CUDA 지원 GPU (최소 4GB VRAM 권장)
- **타일 크기**: 256x256 (큰 타일로 효율적 처리)
- **정밀도**: FP16 (Half Precision) 사용으로 메모리 절약

#### CPU 모드 (베이직)
```python
{
  "device": "cpu",
  "scale_factor": 2
}
```
- **장점**: 추가 하드웨어 불필요, 비용 절감
- **단점**: 느린 처리 속도
- **타일 크기**: 128x128 (작은 타일로 메모리 절약)
- **정밀도**: FP32 (Full Precision)

### 3. 메모리 최적화

#### 타일링 (Tiling) 기법
큰 이미지를 작은 타일로 나누어 처리하여 메모리 사용량 감소:

```
원본 이미지 (1920x1080)
┌──────────────────────┐
│  ┌────┬────┬────┐    │
│  │T1  │T2  │T3  │    │  각 타일을
│  ├────┼────┼────┤    │  순차적으로
│  │T4  │T5  │T6  │    │  업스케일
│  ├────┼────┼────┤    │
│  │T7  │T8  │T9  │    │
│  └────┴────┴────┘    │
└──────────────────────┘
```

**타일 크기 설정**:
- GPU: 256x256 (처리 속도와 메모리 균형)
- CPU: 128x128 (메모리 절약 우선)

#### 타일 패딩 (Tile Padding)
타일 경계의 아티팩트 방지를 위해 인접 영역을 포함하여 처리:
- GPU: 10px 패딩
- CPU: 5px 패딩

---

## 처리 파이프라인

### 전체 흐름

```
1. Job 수신 (SQS)
   ↓
2. S3에서 비디오 다운로드
   ↓
3. 비디오 정보 추출
   │ - 해상도 (width, height)
   │ - FPS (frames per second)
   │ - 총 프레임 수
   ↓
4. 업스케일러 초기화
   │ - 디바이스 설정 (GPU/CPU)
   │ - 모델 로드 (RealESRGAN_x4plus.pth)
   │ - 타일 크기 설정
   ↓
5. 오디오 추출 (ffmpeg)
   │ - 오디오 트랙 존재 시 별도 파일로 추출
   │ - 없으면 스킵
   ↓
6. 프레임별 업스케일
   │ - 각 프레임을 순차적으로 처리
   │ - BGR → RGB 변환
   │ - Real-ESRGAN 업스케일
   │ - RGB → BGR 변환
   │ - 임시 비디오 파일에 저장
   │ - 진행률 로깅 (5% 단위)
   ↓
7. 비디오 인코딩 및 병합 (ffmpeg)
   │ - 업스케일된 비디오를 H.264로 인코딩
   │ - 오디오 트랙 병합 (있는 경우)
   ↓
8. S3에 결과 업로드
   ↓
9. Spring 서버에 완료 통보
   ↓
10. 임시 파일 정리
```

### 상세 단계

#### 3단계: 비디오 정보 추출
```python
cap = cv2.VideoCapture(str(input_path))
fps = cap.get(cv2.CAP_PROP_FPS)          # 예: 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 예: 1920
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 예: 1080
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 예: 900
```

#### 4단계: 업스케일러 초기화
```python
# RRDBNet 모델 생성 (Real-ESRGAN의 기본 네트워크)
model = RRDBNet(
    num_in_ch=3,      # RGB 입력
    num_out_ch=3,     # RGB 출력
    num_feat=64,      # Feature 채널 수
    num_block=23,     # RRDB 블록 수
    num_grow_ch=32,   # Dense connection 채널 증가량
    scale=4           # 4배 업스케일
)

# RealESRGANer 래퍼
upscaler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=256,         # GPU: 256, CPU: 128
    tile_pad=10,      # GPU: 10, CPU: 5
    pre_pad=0,
    half=True,        # GPU에서만 FP16 사용
    device='cuda'     # 'cuda' 또는 'cpu'
)
```

#### 5단계: 오디오 추출
```bash
ffmpeg -y \
  -i input.mp4 \
  -vn \              # 비디오 제외
  -acodec copy \     # 오디오 원본 유지
  temp_audio.aac
```

#### 6단계: 프레임별 업스케일
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR(OpenCV 기본) → RGB(모델 입력)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Real-ESRGAN 업스케일
    # - 타일링 자동 처리
    # - outscale로 최종 배율 조정 (2 또는 4)
    upscaled_rgb, _ = upscaler.enhance(img_rgb, outscale=scale)
    
    # RGB → BGR(OpenCV 저장)
    upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
    
    # 임시 비디오에 저장
    out.write(upscaled_bgr)
```

#### 7단계: 비디오 인코딩 및 병합
```bash
# 오디오가 있는 경우
ffmpeg -y \
  -i temp_upscaled_no_audio.mp4 \
  -i temp_audio.aac \
  -c:v libx264 \      # H.264 비디오 코덱
  -preset medium \    # 인코딩 속도/품질 균형
  -crf 23 \           # 품질 설정 (18-28 권장, 낮을수록 고품질)
  -c:a aac \          # AAC 오디오 코덱
  -b:a 128k \         # 오디오 비트레이트
  -shortest \         # 짧은 쪽에 길이 맞춤
  output.mp4

# 오디오가 없는 경우
ffmpeg -y \
  -i temp_upscaled_no_audio.mp4 \
  -c:v libx264 \
  -preset medium \
  -crf 23 \
  output.mp4
```

---

## GPU/CPU 선택 가이드

### 구독 티어별 권장 설정

| 구독 티어 | 디바이스 | 스케일 | 처리 시간 (1분 영상) | 비용 |
|---------|---------|--------|-------------------|------|
| **베이직** | CPU | 2x | ~20-30분 | 저렴 |
| **스탠다드** | CPU | 4x | ~40-60분 | 저렴 |
| **프리미엄** | GPU | 2x | ~2-3분 | 고가 |
| **프로** | GPU | 4x | ~4-6분 | 고가 |

### 처리 시간 비교 (예시)

**테스트 환경**:
- 비디오: 1920x1080, 30fps, 1분
- CPU: Intel i7-12700K
- GPU: NVIDIA RTX 3080 (10GB)

| 설정 | 처리 시간 | VRAM/RAM 사용 |
|-----|---------|-------------|
| CPU 2x | 25분 | 4GB RAM |
| CPU 4x | 50분 | 6GB RAM |
| GPU 2x | 2.5분 | 3GB VRAM |
| GPU 4x | 5분 | 5GB VRAM |

### Spring에서 Job 생성 예시

```java
// 베이직 구독 - CPU 처리
JobRequest basicJob = JobRequest.builder()
    .userId(userId)
    .taskType(TaskType.UPSCALE)
    .sourceS3Key("uploads/user123/video.mp4")
    .parameters(Map.of(
        "device", "cpu",
        "scale_factor", 2
    ))
    .build();

// 프리미엄 구독 - GPU 처리
JobRequest premiumJob = JobRequest.builder()
    .userId(userId)
    .taskType(TaskType.UPSCALE)
    .sourceS3Key("uploads/user123/video.mp4")
    .parameters(Map.of(
        "device", "gpu",
        "scale_factor", 4
    ))
    .build();
```

---

## 메모리 최적화

### 동시 처리 최적화

여러 사용자가 동시에 작업을 요청할 때 서버 비용 절감 방법:

#### 1. Worker Pool 설정
```python
# echoshot_ai_server/services/worker_pool.py

# CPU 전용 인스턴스 (t3.xlarge)
CPU_WORKER_COUNT = 2  # 2개 작업 동시 처리

# GPU 인스턴스 (g4dn.xlarge) 
GPU_WORKER_COUNT = 1  # GPU는 메모리 제약으로 1개씩
```

#### 2. 인스턴스 타입 선택

**CPU 전용 인스턴스**:
- **t3.xlarge** (4 vCPU, 16GB RAM)
  - 동시 처리: 2개 작업
  - 시간당 비용: ~$0.17
  - 용도: 베이직/스탠다드 구독

**GPU 인스턴스**:
- **g4dn.xlarge** (4 vCPU, 16GB RAM, T4 GPU 16GB)
  - 동시 처리: 1-2개 작업
  - 시간당 비용: ~$0.53
  - 용도: 프리미엄/프로 구독

#### 3. Auto Scaling 전략

```yaml
# CPU Worker Auto Scaling
Min: 1 (항상 대기)
Max: 10 (피크 시간)
Scale Up: 큐에 10개 이상 작업 대기
Scale Down: 유휴 10분 후

# GPU Worker Auto Scaling
Min: 0 (비용 절감)
Max: 5 (피크 시간)
Scale Up: 프리미엄 작업 대기 시 즉시
Scale Down: 유휴 5분 후 (빠른 종료로 비용 절감)
```

#### 4. 타일 크기 최적화

메모리 사용량을 줄이기 위해 타일 크기 조정:

```python
# 메모리 부족 시 타일 크기 자동 감소
DEVICE_CONFIGS = {
    "gpu": {
        "tile": 256,      # 표준
        "tile_fallback": 128  # 메모리 부족 시
    },
    "cpu": {
        "tile": 128,      # 표준
        "tile_fallback": 64   # 메모리 부족 시
    }
}
```

#### 5. 배치 처리 우선순위

```python
# 우선순위 큐 사용
Priority 1: 프리미엄 구독 (GPU)
Priority 2: 스탠다드 구독 (CPU)
Priority 3: 베이직 구독 (CPU)

# SQS 큐 분리
upscale-gpu-queue.fifo    # GPU 작업 전용
upscale-cpu-queue.fifo    # CPU 작업 전용
```

---

## 구독 서비스 연동

### Spring 백엔드 연동 예시

```java
@Service
@RequiredArgsConstructor
public class VideoUpscaleService {
    
    private final SubscriptionService subscriptionService;
    private final SqsClient sqsClient;
    
    public JobResponse requestUpscale(Long userId, UpscaleRequest request) {
        // 1. 사용자 구독 정보 확인
        Subscription subscription = subscriptionService.getUserSubscription(userId);
        
        // 2. 구독 티어에 따라 디바이스 결정
        String device = determineDevice(subscription.getTier());
        int maxScale = determineMaxScale(subscription.getTier());
        
        // 3. 스케일 검증
        int scale = Math.min(request.getScale(), maxScale);
        
        // 4. Job 생성
        Job job = Job.builder()
            .userId(userId)
            .taskType(TaskType.UPSCALE)
            .sourceS3Key(request.getS3Key())
            .parameters(Map.of(
                "device", device,
                "scale_factor", scale
            ))
            .status(JobStatus.QUEUED)
            .build();
        
        jobRepository.save(job);
        
        // 5. SQS 큐에 전송 (디바이스에 따라 다른 큐)
        String queueUrl = device.equals("gpu") 
            ? sqsConfig.getGpuQueueUrl() 
            : sqsConfig.getCpuQueueUrl();
        
        sqsClient.sendMessage(queueUrl, job.toJson());
        
        return JobResponse.from(job);
    }
    
    private String determineDevice(SubscriptionTier tier) {
        return switch (tier) {
            case BASIC, STANDARD -> "cpu";
            case PREMIUM, PRO -> "gpu";
        };
    }
    
    private int determineMaxScale(SubscriptionTier tier) {
        return switch (tier) {
            case BASIC -> 2;
            case STANDARD, PREMIUM, PRO -> 4;
        };
    }
}
```

### 구독 티어 정의

```java
public enum SubscriptionTier {
    BASIC("베이직", "cpu", 2, 100),      // 디바이스, 최대 스케일, 월 한도(분)
    STANDARD("스탠다드", "cpu", 4, 300),
    PREMIUM("프리미엄", "gpu", 2, 200),
    PRO("프로", "gpu", 4, 500);
    
    private final String displayName;
    private final String device;
    private final int maxScale;
    private final int monthlyMinutes;
}
```

---

## 로컬 테스트

### 환경 설정

#### 1. Python 환경
```bash
# 가상환경 생성
python -m venv venv

# 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

#### 2. FFmpeg 설치

**Windows**:
```bash
# Chocolatey 사용
choco install ffmpeg

# 또는 수동 다운로드
# https://ffmpeg.org/download.html
```

**macOS**:
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### 3. Real-ESRGAN 모델 다운로드

```bash
# weights 폴더 생성
mkdir weights

# 모델 다운로드 (약 64MB)
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -o weights/RealESRGAN_x4plus.pth
```

### 테스트 실행

#### 기본 사용법
```bash
# CPU로 2배 업스케일 (안전하고 느림)
python test_upscale_local.py --input sample.mp4 --scale 2 --device cpu

# GPU로 4배 업스케일 (빠르지만 GPU 필요)
python test_upscale_local.py --input sample.mp4 --scale 4 --device gpu

# 출력 디렉토리 지정
python test_upscale_local.py --input sample.mp4 --output results/ --device cpu
```

#### 샘플 비디오 준비
테스트용 짧은 비디오 생성 (FFmpeg 사용):
```bash
# 5초 테스트 비디오 생성
ffmpeg -f lavfi -i testsrc=duration=5:size=640x480:rate=30 \
  -pix_fmt yuv420p sample.mp4
```

#### 환경 검증
```bash
# 환경 검증 포함 실행
python test_upscale_local.py --input sample.mp4 --device cpu

# 환경 검증 건너뛰기
python test_upscale_local.py --input sample.mp4 --device cpu --no-validation
```

### 예상 출력

```
======================================================================
환경 검증 중...
======================================================================
PyTorch 버전: 2.0.1
CUDA 사용 가능: True
CUDA 버전: 11.8
GPU: NVIDIA GeForce RTX 3080
GPU 메모리: 10.00 GB
ffmpeg: ffmpeg version 5.1.2
✅ 모델 가중치 확인: weights\RealESRGAN_x4plus.pth
======================================================================

======================================================================
업스케일 테스트 시작
======================================================================
입력 파일: sample.mp4
출력 디렉토리: output_upscaled
스케일: 2x
디바이스: GPU
======================================================================

2024-01-15 10:30:00 [INFO] echoshot_ai_server.tasks.upscale_task: 업스케일 작업 시작 - Scale: 2x, Device: GPU
2024-01-15 10:30:01 [INFO] echoshot_ai_server.tasks.upscale_task: 비디오 정보 - 해상도: 640x480, FPS: 30.0, 프레임: 150
2024-01-15 10:30:02 [INFO] echoshot_ai_server.tasks.upscale_task: 업스케일러 초기화 완료 - Device: GPU, Tile: 256, Half Precision: True
2024-01-15 10:30:02 [INFO] echoshot_ai_server.tasks.upscale_task: 오디오 트랙이 없습니다
2024-01-15 10:30:15 [INFO] echoshot_ai_server.tasks.upscale_task: 업스케일 진행률: 20.0% (30/150)
2024-01-15 10:30:28 [INFO] echoshot_ai_server.tasks.upscale_task: 업스케일 진행률: 40.0% (60/150)
2024-01-15 10:30:41 [INFO] echoshot_ai_server.tasks.upscale_task: 업스케일 진행률: 60.0% (90/150)
2024-01-15 10:30:54 [INFO] echoshot_ai_server.tasks.upscale_task: 업스케일 진행률: 80.0% (120/150)
2024-01-15 10:31:07 [INFO] echoshot_ai_server.tasks.upscale_task: 프레임 업스케일 완료: 150/150 프레임 처리
2024-01-15 10:31:10 [INFO] echoshot_ai_server.tasks.upscale_task: 비디오 인코딩 중 (오디오 없음)...
2024-01-15 10:31:15 [INFO] echoshot_ai_server.tasks.upscale_task: 최종 비디오 생성 완료
2024-01-15 10:31:15 [INFO] __main__: ✅ 업스케일 완료! 결과 파일: output_upscaled\test_20240115_103000_input_upscaled.mp4

======================================================================
✅ 업스케일 성공!
Job ID: test_20240115_103000
출력 키: processed/upscaled/test_20240115_103000/sample_upscaled.mp4
메타데이터:
  - width: 1280
  - height: 960
  - fps: 30.0
  - frame_count: 150
======================================================================
```

---

## 배포 가이드

### EC2 인스턴스 설정

#### CPU 전용 인스턴스 (t3.xlarge)

```bash
# 1. Python 환경 설치
sudo apt update
sudo apt install -y python3.10 python3-pip python3-venv

# 2. FFmpeg 설치
sudo apt install -y ffmpeg

# 3. 프로젝트 클론
git clone https://github.com/your-repo/echoshotx-ai.git
cd echoshotx-ai

# 4. 가상환경 설정
python3 -m venv venv
source venv/bin/activate

# 5. 의존성 설치
pip install -r echoshot_ai_server/requirements.txt

# 6. 모델 다운로드
mkdir -p weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O weights/RealESRGAN_x4plus.pth

# 7. 환경 변수 설정
export AWS_REGION=ap-northeast-2
export SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/xxx/upscale-cpu-queue
export S3_BUCKET_NAME=echoshotx-videos

# 8. Worker 시작
python echoshot_ai_server/main.py
```

#### GPU 인스턴스 (g4dn.xlarge)

```bash
# 1-3: CPU 인스턴스와 동일

# 4. NVIDIA 드라이버 설치
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# 5. CUDA 설치
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 6. PyTorch (CUDA 지원) 설치
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 7-9: CPU 인스턴스와 동일 (큐 URL만 GPU 큐로 변경)
export SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/xxx/upscale-gpu-queue
```

### Systemd 서비스 등록

```bash
# /etc/systemd/system/echoshotx-worker.service
[Unit]
Description=EchoShotX AI Worker
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/echoshotx-ai
Environment="PATH=/home/ubuntu/echoshotx-ai/venv/bin"
Environment="AWS_REGION=ap-northeast-2"
Environment="SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/xxx/upscale-queue"
Environment="S3_BUCKET_NAME=echoshotx-videos"
ExecStart=/home/ubuntu/echoshotx-ai/venv/bin/python echoshot_ai_server/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable echoshotx-worker
sudo systemctl start echoshotx-worker

# 상태 확인
sudo systemctl status echoshotx-worker

# 로그 확인
sudo journalctl -u echoshotx-worker -f
```

### Docker 배포 (선택사항)

#### Dockerfile (CPU)
```dockerfile
FROM python:3.10-slim

# FFmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치
COPY echoshot_ai_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY echoshot_ai_server/ ./echoshot_ai_server/
COPY weights/ ./weights/

# Worker 실행
CMD ["python", "echoshot_ai_server/main.py"]
```

#### Dockerfile (GPU)
```dockerfile
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Python 및 FFmpeg 설치
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch (CUDA) 설치
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
COPY echoshot_ai_server/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 소스 복사
COPY echoshot_ai_server/ ./echoshot_ai_server/
COPY weights/ ./weights/

# Worker 실행
CMD ["python3", "echoshot_ai_server/main.py"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  worker-cpu:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    environment:
      - AWS_REGION=ap-northeast-2
      - SQS_QUEUE_URL=${SQS_CPU_QUEUE_URL}
      - S3_BUCKET_NAME=${S3_BUCKET_NAME}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    deploy:
      replicas: 2
    restart: always

  worker-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    environment:
      - AWS_REGION=ap-northeast-2
      - SQS_QUEUE_URL=${SQS_GPU_QUEUE_URL}
      - S3_BUCKET_NAME=${S3_BUCKET_NAME}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: always
```

---

## 트러블슈팅

### 1. CUDA Out of Memory

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결 방법**:
```python
# 타일 크기 줄이기
DEVICE_CONFIGS = {
    "gpu": {
        "tile": 128,  # 256 → 128로 감소
        "tile_pad": 5
    }
}

# 또는 배치 크기 줄이기 (동시 처리 작업 수)
GPU_WORKER_COUNT = 1
```

### 2. FFmpeg 오디오 추출 실패

**증상**:
```
[ERROR] 오디오 추출 실패
```

**원인**:
- 비디오에 오디오 트랙이 없음
- 손상된 오디오 스트림

**해결 방법**:
코드에서 이미 처리됨 - 오디오 없이도 정상 동작

### 3. 느린 처리 속도 (CPU)

**증상**:
- CPU에서 처리 시간이 너무 오래 걸림

**해결 방법**:
```python
# 1. 낮은 스케일 사용
scale_factor = 2  # 4 대신 2 사용

# 2. 낮은 해상도로 전처리
# Spring에서 미리 720p로 리사이즈 후 처리

# 3. GPU 인스턴스 사용 권장
```

### 4. 모델 로드 실패

**증상**:
```
FileNotFoundError: weights/RealESRGAN_x4plus.pth
```

**해결 방법**:
```bash
# 모델 다운로드
mkdir -p weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O weights/RealESRGAN_x4plus.pth
```

### 5. 메모리 부족 (CPU)

**증상**:
```
MemoryError: Unable to allocate array
```

**해결 방법**:
```python
# 타일 크기 더 줄이기
DEVICE_CONFIGS = {
    "cpu": {
        "tile": 64,  # 128 → 64로 감소
        "tile_pad": 3
    }
}

# 동시 작업 수 줄이기
CPU_WORKER_COUNT = 1
```

### 6. GPU 인식 안됨

**증상**:
```
GPU가 요청되었지만 사용 불가능합니다. CPU로 대체합니다.
```

**확인 사항**:
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"
```

**해결 방법**:
```bash
# PyTorch CUDA 버전 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 7. S3 업로드 실패

**증상**:
```
ClientError: An error occurred (AccessDenied) when calling the PutObject operation
```

**해결 방법**:
```bash
# IAM 권한 확인
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::echoshotx-videos/*"
    }
  ]
}
```

---

## 성능 벤치마크

### 테스트 환경
- **비디오**: 1920x1080, 30fps, 다양한 길이
- **CPU**: Intel i7-12700K
- **GPU**: NVIDIA RTX 3080 (10GB)

### 처리 시간 (초 단위)

| 비디오 길이 | CPU 2x | CPU 4x | GPU 2x | GPU 4x |
|----------|--------|--------|--------|--------|
| 10초 | 250 | 500 | 25 | 50 |
| 30초 | 750 | 1500 | 75 | 150 |
| 1분 | 1500 | 3000 | 150 | 300 |
| 5분 | 7500 | 15000 | 750 | 1500 |

### 메모리 사용량

| 설정 | 피크 RAM/VRAM | 평균 RAM/VRAM |
|-----|--------------|--------------|
| CPU 2x (tile 128) | 4GB | 3GB |
| CPU 4x (tile 128) | 6GB | 4.5GB |
| GPU 2x (tile 256) | 3GB | 2GB |
| GPU 4x (tile 256) | 5GB | 4GB |

---

## 참고 자료

### Real-ESRGAN
- **논문**: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- **GitHub**: https://github.com/xinntao/Real-ESRGAN
- **모델**: RealESRGAN_x4plus (64MB)

### FFmpeg
- **공식 사이트**: https://ffmpeg.org/
- **문서**: https://ffmpeg.org/documentation.html

### PyTorch
- **공식 사이트**: https://pytorch.org/
- **CUDA 버전 호환성**: https://pytorch.org/get-started/locally/

---

## 버전 히스토리

### v2.0.0 (2024-01-15)
- ✅ GPU/CPU 선택 기능 추가
- ✅ 메모리 최적화 (타일링)
- ✅ 오디오 자동 처리
- ✅ 진행률 추적
- ✅ 로컬 테스트 스크립트
- ✅ 상세한 한글 문서화

### v1.0.0 (2023-12-01)
- 초기 버전 (GPU 전용)

---

## 문의 및 지원

문제가 발생하거나 질문이 있으시면:
1. GitHub Issues에 등록
2. 팀 Slack 채널에 문의
3. 이 문서의 트러블슈팅 섹션 확인

**작성자**: EchoShotX AI 팀  
**최종 수정**: 2024-01-15

