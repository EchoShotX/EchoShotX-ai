#!/usr/bin/env python3
"""
FSCRNN GPU 테스트 스크립트
GPU로 실행되는지 확인하는 간단한 테스트
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# 로컬 임포트
try:
    from upscale_task import VideoUpscaler
except ImportError:
    # echoshot_ai_server/tasks에서 임포트 시도
    sys.path.insert(0, str(Path(__file__).parent / "echoshot_ai_server" / "tasks"))
    try:
        from upscale_task import VideoUpscaler
    except ImportError:
        print("❌ upscale_task.py를 찾을 수 없습니다!")
        sys.exit(1)


def check_cuda_support():
    """OpenCV CUDA 지원 확인"""
    print("\n" + "=" * 60)
    print("🔍 CUDA 지원 확인")
    print("=" * 60)
    
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            print(f"✅ CUDA 사용 가능: {cuda_count}개 GPU 감지")
            for i in range(cuda_count):
                device_info = cv2.cuda.getDevice(i)
                print(f"   GPU {i}: {device_info.name()}")
            return True
        else:
            print("❌ CUDA 사용 불가 (GPU 감지 안됨)")
            return False
    except Exception as e:
        print(f"❌ CUDA 체크 실패: {e}")
        return False


def test_fsrcnn_gpu():
    """FSCRNN GPU 테스트"""
    print("\n" + "=" * 60)
    print("🧪 FSCRNN GPU 테스트")
    print("=" * 60)
    
    # 모델 파일 확인
    model_path = Path("weights/FSRCNN_x2.pb")
    if not model_path.exists():
        print(f"❌ 모델 파일 없음: {model_path}")
        print("   다운로드 필요: https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb")
        return False
    
    print(f"✅ 모델 파일 확인: {model_path}")
    
    # 테스트 이미지 생성 (작은 이미지로 빠른 테스트)
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"📸 테스트 이미지: {test_image.shape}")
    
    # GPU 모드로 FSCRNN 초기화
    print("\n--- GPU 모드 테스트 ---")
    try:
        upscaler_gpu = VideoUpscaler(model_profile="fast", device="gpu")
        
        if upscaler_gpu.use_gpu:
            print("✅ GPU 모드로 초기화 성공")
            
            # 모델이 실제로 GPU 백엔드를 사용하는지 확인
            sr = upscaler_gpu.model
            if sr is not None:
                # 간단한 업스케일 테스트
                print("🔄 업스케일 테스트 실행...")
                result = sr.upsample(test_image)
                print(f"✅ 업스케일 성공: {test_image.shape} -> {result.shape}")
                print("✅ FSCRNN이 GPU로 정상 실행 중입니다!")
                return True
            else:
                print("❌ 모델 로드 실패")
                return False
        else:
            print("⚠️  GPU 모드로 설정했지만 CPU로 전환됨")
            print("   (OpenCV CUDA 지원이 없거나 모델이 GPU를 지원하지 않음)")
            return False
            
    except Exception as e:
        print(f"❌ GPU 모드 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fsrcnn_cpu():
    """FSCRNN CPU 테스트 (비교용)"""
    print("\n--- CPU 모드 테스트 (비교용) ---")
    try:
        upscaler_cpu = VideoUpscaler(model_profile="fast", device="cpu")
        print("✅ CPU 모드로 초기화 성공")
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        sr = upscaler_cpu.model
        if sr is not None:
            result = sr.upsample(test_image)
            print(f"✅ CPU 업스케일 성공: {test_image.shape} -> {result.shape}")
            return True
        return False
    except Exception as e:
        print(f"❌ CPU 모드 테스트 실패: {e}")
        return False


def main():
    """메인 실행"""
    print("\n" + "=" * 60)
    print("🚀 FSCRNN GPU 테스트 도구")
    print("=" * 60)
    
    # 1. CUDA 지원 확인
    cuda_available = check_cuda_support()
    
    # 2. GPU 모드 테스트
    gpu_success = test_fsrcnn_gpu()
    
    # 3. CPU 모드 테스트 (비교용)
    cpu_success = test_fsrcnn_cpu()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    print(f"CUDA 지원: {'✅' if cuda_available else '❌'}")
    print(f"GPU 모드: {'✅ 성공' if gpu_success else '❌ 실패'}")
    print(f"CPU 모드: {'✅ 성공' if cpu_success else '❌ 실패'}")
    
    if gpu_success:
        print("\n🎉 FSCRNN이 GPU로 정상 실행됩니다!")
    elif cuda_available:
        print("\n⚠️  CUDA는 사용 가능하지만 GPU 모드가 작동하지 않습니다.")
        print("   OpenCV가 CUDA로 빌드되었는지 확인하세요.")
    else:
        print("\n⚠️  GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()

