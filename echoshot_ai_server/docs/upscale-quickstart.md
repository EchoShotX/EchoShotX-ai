# 비디오 업스케일링 빠른 시작 가이드

## 5분 안에 시작하기

### 1. 환경 설정 (2분)

```bash
# 1. 프로젝트 디렉토리 이동
cd echoshotx-ai

# 2. 가상환경 활성화 (이미 있다고 가정)
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. 모델 다운로드 (첫 실행 시 한 번만)
mkdir -p weights
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o weights/RealESRGAN_x4plus.pth
```

### 2. 테스트 비디오 준비 (1분)

```bash
# FFmpeg로 5초 테스트 비디오 생성
ffmpeg -f lavfi -i testsrc=duration=5:size=640x480:rate=30 -pix_fmt yuv420p test_video.mp4
```

### 3. 로컬 테스트 실행 (2분)

```bash
# CPU로 테스트 (느리지만 안전)
python test_upscale_local.py --input test_video.mp4 --scale 2 --device cpu

# GPU가 있다면 (빠름)
python test_upscale_local.py --input test_video.mp4 --scale 2 --device gpu
```

### 4. 결과 확인

```bash
# output_upscaled/ 폴더에 결과 파일 생성됨
ls output_upscaled/
```

---

## Spring에서 Job 요청하기

### 베이직 구독 - CPU 2배

```java
JobRequest request = JobRequest.builder()
    .userId(userId)
    .taskType(TaskType.UPSCALE)
    .sourceS3Key("uploads/user123/video.mp4")
    .parameters(Map.of(
        "device", "cpu",
        "scale_factor", 2
    ))
    .build();

Job job = jobService.createJob(request);
sqsService.sendMessage(cpuQueueUrl, job);
```

### 프리미엄 구독 - GPU 4배

```java
JobRequest request = JobRequest.builder()
    .userId(userId)
    .taskType(TaskType.UPSCALE)
    .sourceS3Key("uploads/user123/video.mp4")
    .parameters(Map.of(
        "device", "gpu",
        "scale_factor", 4
    ))
    .build();

Job job = jobService.createJob(request);
sqsService.sendMessage(gpuQueueUrl, job);
```

---

## 구독 티어별 설정

| 구독 | 디바이스 | 최대 스케일 | 예상 시간 (1분 영상) |
|-----|---------|-----------|------------------|
| 베이직 | CPU | 2x | ~25분 |
| 스탠다드 | CPU | 4x | ~50분 |
| 프리미엄 | GPU | 2x | ~2.5분 |
| 프로 | GPU | 4x | ~5분 |

---

## 문제 해결

### 모델을 찾을 수 없습니다
```bash
mkdir -p weights
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o weights/RealESRGAN_x4plus.pth
```

### FFmpeg를 찾을 수 없습니다
- Windows: `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`

### GPU를 사용할 수 없습니다
```bash
# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# False가 나오면 CUDA 버전 PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 더 알아보기

- 📖 [상세 가이드](upscale-guide.md) - 전체 문서
- 🔧 [트러블슈팅](upscale-guide.md#트러블슈팅) - 문제 해결
- 🚀 [배포 가이드](upscale-guide.md#배포-가이드) - EC2 배포

---

**다음 단계**: [upscale-guide.md](upscale-guide.md)에서 전체 문서를 확인하세요.

