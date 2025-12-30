#!/usr/bin/env python3
"""
FSCRNN GPU 빠른 테스트 스크립트
자동으로 FSRCNN을 GPU 모드로 테스트
"""

import sys
from pathlib import Path
import logging
import cv2
import numpy as np

# 로컬 임포트
from upscale_task import OptimizedUpscaleTask, VideoUpscaler, MODEL_PROFILES

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_gpu_support():
    """GPU 지원 확인"""
    print("\n" + "=" * 60)
    print("🔍 GPU 지원 확인")
    print("=" * 60)
    
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            print(f"✅ CUDA 사용 가능: {cuda_count}개 GPU 감지")
            for i in range(cuda_count):
                try:
                    device_info = cv2.cuda.getDevice(i)
                    print(f"   GPU {i}: {device_info.name()}")
                except:
                    print(f"   GPU {i}: 정보 조회 실패")
            return True
        else:
            print("❌ CUDA 사용 불가 (GPU 감지 안됨)")
            return False
    except Exception as e:
        print(f"❌ CUDA 체크 실패: {e}")
        print("   OpenCV가 CUDA로 빌드되지 않았을 수 있습니다.")
        return False


def test_fsrcnn_gpu_direct():
    """FSRCNN GPU 직접 테스트 (비디오 없이)"""
    print("\n" + "=" * 60)
    print("🧪 FSRCNN GPU 직접 테스트")
    print("=" * 60)
    
    # 모델 파일 확인
    model_path = Path("weights/FSRCNN_x2.pb")
    if not model_path.exists():
        print(f"❌ 모델 파일 없음: {model_path}")
        return False
    
    print(f"✅ 모델 파일 확인: {model_path}")
    
    # GPU 모드로 FSRCNN 초기화
    print("\n--- GPU 모드로 FSRCNN 초기화 ---")
    try:
        upscaler = VideoUpscaler(model_profile="fast", device="gpu")
        
        if upscaler.use_gpu:
            print("✅ GPU 모드로 초기화 성공!")
            print(f"   모델: {upscaler.profile.name}")
            print(f"   디바이스: GPU (CUDA)")
        else:
            print("⚠️  GPU 모드로 설정했지만 CPU로 전환됨")
            print("   OpenCV CUDA 지원이 없거나 모델이 GPU를 지원하지 않음")
            return False
        
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        print(f"\n📸 테스트 이미지 생성: {test_image.shape}")
        
        # 업스케일 테스트
        print("🔄 업스케일 테스트 실행...")
        result = upscaler.upscale_frame(test_image, scale=2)
        print(f"✅ 업스케일 성공!")
        print(f"   입력 크기: {test_image.shape}")
        print(f"   출력 크기: {result.shape}")
        print("\n🎉 FSRCNN이 GPU로 정상 실행됩니다!")
        return True
        
    except Exception as e:
        print(f"❌ GPU 모드 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fsrcnn_with_video():
    """비디오로 FSRCNN GPU 테스트"""
    print("\n" + "=" * 60)
    print("🎬 비디오로 FSRCNN GPU 테스트")
    print("=" * 60)
    
    # 비디오 파일 찾기
    test_videos_dir = Path("test_videos")
    videos = []
    if test_videos_dir.exists():
        videos = list(test_videos_dir.glob("*.mp4")) + \
                 list(test_videos_dir.glob("*.avi")) + \
                 list(test_videos_dir.glob("*.mov"))
    
    if not videos:
        print("⚠️  test_videos/ 폴더에 비디오 파일이 없습니다.")
        print("   비디오 테스트를 건너뜁니다.")
        return None
    
    # 첫 번째 비디오 사용
    input_video = videos[0]
    print(f"📹 테스트 비디오: {input_video.name}")
    
    # 출력 경로
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{input_video.stem}_fsrcnn_gpu_test.mp4"
    
    # 임시 디렉토리
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"📁 출력 파일: {output_path}")
    print(f"⚙️  모델: FSRCNN (fast)")
    print(f"🎮 디바이스: GPU")
    print("\n처리 시작...")
    
    try:
        task = OptimizedUpscaleTask(temp_dir)
        task.process(
            input_path=input_video,
            output_path=output_path,
            scale=2,
            device="gpu",
            model_profile="fast"  # FSRCNN
        )
        
        print(f"\n✅ 비디오 처리 완료: {output_path}")
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"📦 파일 크기: {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ 비디오 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 실행"""
    print("\n" + "=" * 60)
    print("🚀 FSRCNN GPU 테스트 도구")
    print("=" * 60)
    
    # 1. GPU 지원 확인
    gpu_available = check_gpu_support()
    
    if not gpu_available:
        print("\n⚠️  GPU를 사용할 수 없습니다.")
        print("   CPU 모드로만 테스트할 수 있습니다.")
        response = input("\n계속하시겠습니까? (y/n): ").strip().lower()
        if response != "y":
            return
    
    # 2. FSRCNN GPU 직접 테스트
    print("\n" + "=" * 60)
    direct_test = test_fsrcnn_gpu_direct()
    
    # 3. 비디오 테스트 (선택사항)
    print("\n" + "=" * 60)
    response = input("비디오로도 테스트하시겠습니까? (y/n) [기본: n]: ").strip().lower() or "n"
    if response == "y":
        video_test = test_fsrcnn_with_video()
    else:
        video_test = None
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    print(f"GPU 지원: {'✅' if gpu_available else '❌'}")
    print(f"FSRCNN GPU 직접 테스트: {'✅ 성공' if direct_test else '❌ 실패'}")
    if video_test is not None:
        print(f"비디오 테스트: {'✅ 성공' if video_test else '❌ 실패'}")
    
    if direct_test:
        print("\n🎉 FSRCNN이 GPU로 정상 실행됩니다!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}", exc_info=True)

