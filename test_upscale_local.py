#!/usr/bin/env python3
"""
비디오 업스케일 로컬 테스트 스크립트

사용법:
    python test_upscale.py

디렉토리 구조:
    test_videos/       # 여기에 테스트할 비디오 파일 넣기
    output/            # 결과물 저장됨 (자동 생성)
    weights/           # 모델 파일 (자동 다운로드)
"""

import sys
from pathlib import Path
import time
import logging
from typing import Optional

# 로컬 임포트
try:
    from upscale_task import OptimizedUpscaleTask, MODEL_PROFILES
except ImportError:
    print("❌ upscale_task.py를 같은 폴더에 넣어주세요!")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class VideoUpscaleTester:
    """로컬 테스트용 래퍼"""

    def __init__(self):
        self.input_dir = Path("test_videos")
        self.output_dir = Path("output")
        self.temp_dir = Path("temp")
        self.weights_dir = Path("weights")

        # 디렉토리 생성
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)

    def setup_models(self):
        """모델 파일 체크 및 다운로드 안내"""
        logger.info("=== 모델 파일 체크 ===")

        model_files = {
            "FSRCNN_x2.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
            "EDSR_x2.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
            "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        }

        missing = []
        for filename, url in model_files.items():
            model_path = self.weights_dir / filename
            if model_path.exists():
                logger.info(f"✅ {filename} 존재")
            else:
                logger.warning(f"⚠️  {filename} 없음")
                missing.append((filename, url))

        if missing:
            print("\n📥 모델 다운로드가 필요합니다:")
            print("=" * 60)
            for filename, url in missing:
                print(f"\n파일명: {filename}")
                print(f"다운로드: {url}")
                print(f"저장 위치: weights/{filename}")
            print("=" * 60)
            print("\n또는 자동 다운로드:")
            print("pip install gdown")
            print("python download_models.py  # 별도 스크립트 제공\n")

    def list_videos(self):
        """테스트 가능한 비디오 목록 출력"""
        videos = list(self.input_dir.glob("*.mp4")) + \
                 list(self.input_dir.glob("*.avi")) + \
                 list(self.input_dir.glob("*.mov"))

        if not videos:
            return []

        print("\n📹 사용 가능한 비디오:")
        print("=" * 60)
        for i, video in enumerate(videos, 1):
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"{i}. {video.name} ({size_mb:.1f} MB)")
        print("=" * 60)

        return videos

    def select_model_profile(self) -> str:
        """모델 프로필 선택"""
        print("\n🎯 모델 프로필 선택:")
        print("=" * 60)
        for i, (key, profile) in enumerate(MODEL_PROFILES.items(), 1):
            print(f"{i}. {key.upper()}")
            print(f"   - 모델: {profile.name}")
            print(f"   - 속도: {'⚡' * profile.speed_score}/10")
            print(f"   - 품질: {'⭐' * profile.quality_score}/10")
            print(f"   - VRAM: {profile.vram_usage}")
            print(f"   - 추천: {profile.best_for}\n")
        print("=" * 60)

        while True:
            choice = input("선택 (1-3) [기본: 2]: ").strip() or "2"
            if choice in ["1", "2", "3"]:
                return list(MODEL_PROFILES.keys())[int(choice) - 1]
            print("❌ 1, 2, 3 중 하나를 입력하세요")

    def select_device(self) -> str:
        """디바이스 선택"""
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n🎮 GPU 감지: {gpu_name}")
            choice = input("GPU 사용? (y/n) [기본: y]: ").strip().lower() or "y"
            return "gpu" if choice == "y" else "cpu"
        else:
            print("\n💻 GPU 없음, CPU 사용")
            return "cpu"

    def run(self):
        """메인 실행"""
        print("\n" + "=" * 60)
        print("🎬 비디오 업스케일 테스트 도구")
        print("=" * 60)

        # 1. 모델 체크
        self.setup_models()

        # 2. 비디오 목록
        videos = self.list_videos()
        if not videos:
            print("\n❌ test_videos/ 폴더에 비디오 파일이 없습니다!")
            print("   .mp4, .avi, .mov 파일을 넣어주세요.")
            return

        # 3. 비디오 선택
        while True:
            choice = input("\n처리할 비디오 번호 입력: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(videos):
                input_video = videos[int(choice) - 1]
                break
            print("❌ 올바른 번호를 입력하세요")

        # 4. 설정 선택
        model_profile = self.select_model_profile()
        device = self.select_device()

        scale = input("\n배율 선택 (2 또는 4) [기본: 2]: ").strip() or "2"
        scale = int(scale) if scale in ["2", "4"] else 2

        # 5. 출력 파일명
        output_name = f"{input_video.stem}_x{scale}_{model_profile}.mp4"
        output_path = self.output_dir / output_name

        # 6. 처리 시작
        print("\n" + "=" * 60)
        print("⚙️  처리 시작...")
        print(f"   입력: {input_video.name}")
        print(f"   출력: {output_name}")
        print(f"   배율: x{scale}")
        print(f"   모델: {MODEL_PROFILES[model_profile].name}")
        print(f"   디바이스: {device.upper()}")
        print("=" * 60 + "\n")

        start_time = time.time()

        try:
            task = OptimizedUpscaleTask(self.temp_dir)
            task.process(
                input_path=input_video,
                output_path=output_path,
                scale=scale,
                device=device,
                model_profile=model_profile
            )

            elapsed = time.time() - start_time

            # 결과 출력
            print("\n" + "=" * 60)
            print("✅ 처리 완료!")
            print("=" * 60)
            print(f"⏱️  처리 시간: {elapsed:.1f}초")
            print(f"📁 출력 파일: {output_path}")
            print(f"📦 파일 크기: {output_path.stat().st_size / (1024 * 1024):.1f} MB")
            print("=" * 60)

        except Exception as e:
            logger.error(f"❌ 처리 실패: {e}", exc_info=True)
            print("\n처리 중 오류 발생. 로그를 확인하세요.")

        finally:
            # 임시 파일 정리
            for temp_file in self.temp_dir.glob("*"):
                temp_file.unlink()


def main():
    """엔트리 포인트"""
    tester = VideoUpscaleTester()

    try:
        tester.run()
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()