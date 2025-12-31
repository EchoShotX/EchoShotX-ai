#!/usr/bin/env python3
"""
FSCRNN GPU 독립 테스트 스크립트
전체 서버 인프라(SQS, S3)에 의존하지 않고 GPU 연산만 테스트
프로젝트 루트에서 독립 실행 가능
"""

import cv2
import numpy as np
import time
from pathlib import Path


def check_gpu_support():
    """GPU 지원 확인"""
    print("\n" + "=" * 60)
    print("🔍 CUDA 지원 확인")
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


def test_fsrcnn_gpu_speed():
    """FSRCNN GPU 속도 테스트 (독립 실행)"""
    print("\n" + "=" * 60)
    print("🧪 FSRCNN GPU 속도 테스트")
    print("=" * 60)
    
    # 모델 파일 경로 확인 (프로젝트 루트 기준)
    possible_paths = [
        Path("echoshot_ai_server/tasks/weights/FSRCNN_x2.pb"),  # 루트에서 실행 시
        Path("weights/FSRCNN_x2.pb"),  # tasks 폴더에서 실행 시
        Path(__file__).parent / "echoshot_ai_server" / "tasks" / "weights" / "FSRCNN_x2.pb",
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        print("   다음 경로에서 찾았습니다:")
        for path in possible_paths:
            abs_path = path.resolve()
            print(f"   - {abs_path} {'✅' if path.exists() else '❌'}")
        return False
    
    print(f"✅ 모델 파일 확인: {model_path.resolve()}")
    
    # 1. 모델 로드
    print("\n--- FSRCNN 모델 로드 중 ---")
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel("fsrcnn", 2)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. GPU 설정
    print("\n--- GPU 백엔드 설정 중 ---")
    try:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("✅ GPU 백엔드 설정 완료 (CUDA)")
    except Exception as e:
        print(f"⚠️  GPU 백엔드 설정 실패: {e}")
        print("   CPU 모드로 실행됩니다.")
    
    # 3. 다양한 해상도로 속도 테스트
    test_cases = [
        (480, 640, "480p (SD)"),
        (720, 1280, "720p (HD)"),
        (1080, 1920, "1080p (Full HD)"),
        (1440, 2560, "1440p (2K)"),
    ]
    
    print("\n" + "=" * 60)
    print("⚡ 속도 측정 시작")
    print("=" * 60)
    
    results = []
    
    for height, width, name in test_cases:
        # 테스트 이미지 생성
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 워밍업 (첫 실행은 보통 느림)
        _ = sr.upsample(img)
        
        # 실제 속도 측정 (3회 평균)
        times = []
        for i in range(3):
            start = time.time()
            result = sr.upsample(img)
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        output_h, output_w = result.shape[:2]
        
        print(f"\n{name}:")
        print(f"  입력: {width}x{height}")
        print(f"  출력: {output_w}x{output_h}")
        print(f"  평균 처리 시간: {avg_time*1000:.2f}ms")
        print(f"  예상 FPS: {fps:.2f}")
        
        results.append({
            "resolution": name,
            "input": f"{width}x{height}",
            "output": f"{output_w}x{output_h}",
            "time_ms": avg_time * 1000,
            "fps": fps
        })
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    print(f"{'해상도':<20} {'처리시간':<15} {'FPS':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['resolution']:<20} {r['time_ms']:>6.2f}ms      {r['fps']:>6.2f}")
    print("=" * 60)
    
    # 1080p 기준 성능 평가
    hd_result = next((r for r in results if "1080p" in r["resolution"]), None)
    if hd_result:
        print(f"\n🎯 1080p → 4K 업스케일 성능:")
        print(f"   처리 시간: {hd_result['time_ms']:.2f}ms")
        print(f"   예상 FPS: {hd_result['fps']:.2f}")
        if hd_result['fps'] >= 30:
            print("   ✅ 실시간 처리 가능 (30fps 이상)")
        elif hd_result['fps'] >= 15:
            print("   ⚠️  준실시간 처리 가능 (15-30fps)")
        else:
            print("   ⚠️  실시간 처리 어려움 (15fps 미만)")
    
    print("\n🎉 FSRCNN GPU 테스트 완료!")
    return True


def main():
    """메인 실행"""
    print("\n" + "=" * 60)
    print("🚀 FSRCNN GPU 독립 테스트 도구")
    print("=" * 60)
    print("전체 서버 인프라 없이 GPU 연산만 테스트합니다.")
    print("프로젝트 루트에서 독립 실행 중...")
    
    # 1. GPU 지원 확인
    gpu_available = check_gpu_support()
    
    if not gpu_available:
        print("\n⚠️  GPU를 사용할 수 없습니다.")
        print("   CPU 모드로 테스트를 계속합니다.")
    
    # 2. FSRCNN GPU 속도 테스트
    success = test_fsrcnn_gpu_speed()
    
    if success:
        print("\n✅ 모든 테스트 완료!")
    else:
        print("\n❌ 테스트 실패")
    
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

