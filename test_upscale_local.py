"""
비디오 업스케일링 로컬 테스트 스크립트

이 스크립트는 실제 EC2 배포 전에 로컬 환경에서 업스케일 기능을 테스트합니다.
S3/SQS 없이 로컬 파일로 직접 테스트할 수 있습니다.

사용법:
    python test_upscale_local.py --input sample.mp4 --scale 2 --device cpu
    python test_upscale_local.py --input sample.mp4 --scale 4 --device gpu
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

from echoshot_ai_server.tasks.upscale_task import UpscaleTask
from echoshot_ai_server.domain.job import Job, JobStatus, TaskType

# from tasks.upscale_task import UpscaleTask
# from domain.job import Job, JobStatus, TaskType

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MockS3Client:
    """S3 클라이언트 목(Mock) - 로컬 파일 시스템 사용"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, s3_key: str, local_path: Path):
        """S3 다운로드 대신 로컬 파일 복사"""
        logger.info(f"[MOCK S3] 파일 복사: {self.input_path} -> {local_path}")
        shutil.copy2(self.input_path, local_path)
    
    def upload_file(self, local_path: Path, s3_key: str):
        """S3 업로드 대신 output 폴더에 저장"""
        output_file = self.output_dir / local_path.name
        logger.info(f"[MOCK S3] 결과 저장: {local_path} -> {output_file}")
        shutil.copy2(local_path, output_file)
        logger.info(f"✅ 업스케일 완료! 결과 파일: {output_file}")


def create_test_job(job_id: str, input_file: str, scale: int, device: str) -> Job:
    """테스트용 Job 객체 생성"""
    return Job(
        job_id=job_id,
        user_id="test_user",
        task_type=TaskType.UPSCALE,
        source_s3_key=f"input/{Path(input_file).name}",  # 실제로는 사용 안함
        parameters={
            "scale_factor": scale,
            "device": device
        },
        status=JobStatus.QUEUED,
        # created_at=datetime.now(),
        callback_url="http://localhost/callback",
        receipt_handle="test_receipt_handle"

    )


def validate_environment():
    """실행 환경 검증"""
    import torch
    
    logger.info("=" * 70)
    logger.info("환경 검증 중...")
    logger.info("=" * 70)
    
    # PyTorch 확인
    logger.info(f"PyTorch 버전: {torch.__version__}")
    
    # CUDA 확인
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA 사용 가능: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA 버전: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # ffmpeg 확인
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        ffmpeg_version = result.stdout.split('\n')[0]
        logger.info(f"ffmpeg: {ffmpeg_version}")
    except FileNotFoundError:
        logger.error("❌ ffmpeg를 찾을 수 없습니다. ffmpeg를 설치해주세요.")
        logger.error("   다운로드: https://ffmpeg.org/download.html")
        return False
    except subprocess.CalledProcessError:
        logger.error("❌ ffmpeg 실행 중 오류 발생")
        return False
    
    # Real-ESRGAN 모델 가중치 확인
    model_path = Path("weights/RealESRGAN_x4plus.pth")
    if not model_path.exists():
        logger.error(f"❌ 모델 가중치를 찾을 수 없습니다: {model_path}")
        logger.error("   다운로드 방법:")
        logger.error("   1. weights/ 폴더 생성")
        logger.error("   2. https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        logger.error("      위 링크에서 모델 다운로드 후 weights/ 폴더에 저장")
        return False
    
    logger.info(f"✅ 모델 가중치 확인: {model_path}")
    logger.info("=" * 70)
    logger.info("")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="비디오 업스케일링 로컬 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # CPU로 2배 업스케일 (느리지만 안전)
  python test_upscale_local.py --input sample.mp4 --scale 2 --device cpu

  # GPU로 4배 업스케일 (빠르지만 GPU 필요)
  python test_upscale_local.py --input sample.mp4 --scale 4 --device gpu

  # 출력 디렉토리 지정
  python test_upscale_local.py --input sample.mp4 --output results/ --device cpu
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 비디오 파일 경로 (예: sample.mp4)"
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=2,
        choices=[2, 4],
        help="업스케일 배율 (2 또는 4, 기본값: 2)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "gpu"],
        help="처리 장치 (cpu 또는 gpu, 기본값: cpu)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output_upscaled",
        help="출력 디렉토리 (기본값: output_upscaled/)"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="환경 검증 건너뛰기"
    )
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        return 1
    
    # 환경 검증
    if not args.no_validation:
        if not validate_environment():
            logger.error("환경 검증 실패. 위 오류를 해결한 후 다시 시도하세요.")
            return 1
    
    # GPU 요청 시 경고
    if args.device == "gpu":
        import torch
        if not torch.cuda.is_available():
            logger.warning("⚠️  GPU가 요청되었지만 사용할 수 없습니다. CPU로 대체됩니다.")
            logger.warning("    처리 시간이 상당히 오래 걸릴 수 있습니다.")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 테스트 실행
    logger.info("=" * 70)
    logger.info("업스케일 테스트 시작")
    logger.info("=" * 70)
    logger.info(f"입력 파일: {input_path}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"스케일: {args.scale}x")
    logger.info(f"디바이스: {args.device.upper()}")
    logger.info("=" * 70)
    logger.info("")
    
    # Job 생성
    job_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job = create_test_job(job_id, str(input_path), args.scale, args.device)
    
    # Mock S3 클라이언트
    mock_s3 = MockS3Client(input_path, output_dir)
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory(prefix="upscale_test_") as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"임시 디렉토리: {temp_path}")
        logger.info("")
        
        # UpscaleTask 실행
        task = UpscaleTask(job, mock_s3, temp_path)
        
        try:
            result = task.execute()
            
            logger.info("")
            logger.info("=" * 70)
            if result.status == JobStatus.COMPLETED:
                logger.info("✅ 업스케일 성공!")
                logger.info(f"Job ID: {result.job_id}")
                logger.info(f"출력 키: {result.output_s3_key}")
                
                if result.metadata:
                    logger.info("메타데이터:")
                    for key, value in result.metadata.items():
                        logger.info(f"  - {key}: {value}")
                
                logger.info("=" * 70)
                return 0
            else:
                logger.error("❌ 업스케일 실패")
                logger.error(f"상태: {result.status}")
                logger.error(f"에러: {result.error_message}")
                logger.info("=" * 70)
                return 1
                
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌ 예외 발생: {e}", exc_info=True)
            logger.error("=" * 70)
            return 1


if __name__ == "__main__":
    sys.exit(main())

