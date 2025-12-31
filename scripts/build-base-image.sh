#!/bin/bash
# 베이스 이미지 빌드 스크립트
# EC2 GPU 인스턴스에서 실행하여 OpenCV CUDA가 포함된 베이스 이미지를 빌드하고 Docker Hub에 푸시합니다.
# 이 스크립트는 OpenCV 업데이트 시에만 실행하면 됩니다 (약 한 달에 1회 이하).

set -e

echo "=========================================="
echo "베이스 이미지 빌드 스크립트"
echo "=========================================="

# 환경 변수 확인
DOCKERHUB_USERNAME=${DOCKERHUB_USERNAME:-echoshot}
OPENCV_VERSION=${OPENCV_VERSION:-4.10.0}
BASE_IMAGE_NAME="${DOCKERHUB_USERNAME}/opencv-cuda-t4"
BASE_IMAGE_TAG="${BASE_IMAGE_NAME}:${OPENCV_VERSION}"
BASE_IMAGE_LATEST="${BASE_IMAGE_NAME}:latest"

echo "빌드 설정:"
echo "  Docker Hub 사용자명: ${DOCKERHUB_USERNAME}"
echo "  OpenCV 버전: ${OPENCV_VERSION}"
echo "  베이스 이미지: ${BASE_IMAGE_TAG}"
echo ""

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker가 설치되어 있지 않습니다."
    exit 1
fi

# Docker Hub 로그인 확인
echo "Docker Hub 로그인 상태 확인..."
if ! docker info | grep -q "Username"; then
    echo "Docker Hub에 로그인이 필요합니다."
    echo "다음 명령어로 로그인하세요:"
    echo "  docker login"
    read -p "로그인을 완료했습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "로그인이 취소되었습니다."
        exit 1
    fi
fi

# GPU 확인 (선택사항이지만 권장)
if lspci | grep -i nvidia &> /dev/null; then
    echo "GPU 인스턴스가 감지되었습니다."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    else
        echo "WARNING: NVIDIA 드라이버가 설치되어 있지 않습니다."
    fi
else
    echo "WARNING: GPU 인스턴스가 아닙니다. 베이스 이미지는 빌드되지만 CUDA는 사용할 수 없습니다."
fi

# Dockerfile.base 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_BASE="${PROJECT_ROOT}/Dockerfile.base"

if [ ! -f "${DOCKERFILE_BASE}" ]; then
    echo "ERROR: Dockerfile.base를 찾을 수 없습니다: ${DOCKERFILE_BASE}"
    exit 1
fi

echo ""
echo "베이스 이미지 빌드를 시작합니다..."
echo "이 작업은 약 1.5-2시간이 소요될 수 있습니다."
echo ""

# BuildKit 활성화
export DOCKER_BUILDKIT=1

# 베이스 이미지 빌드
cd "${PROJECT_ROOT}"
docker build \
    --build-arg OPENCV_VERSION=${OPENCV_VERSION} \
    --build-arg CUDA_VERSION=11.8.0 \
    --build-arg CUDA_ARCH=7.5 \
    -f Dockerfile.base \
    -t "${BASE_IMAGE_TAG}" \
    -t "${BASE_IMAGE_LATEST}" \
    .

echo ""
echo "베이스 이미지 빌드 완료!"
echo ""

# OpenCV CUDA 빌드 검증
echo "OpenCV CUDA 빌드 검증 중..."
docker run --rm "${BASE_IMAGE_TAG}" python3 -c "
import cv2
print(f'OpenCV 버전: {cv2.__version__}')
print(f'CUDA 사용 가능: {cv2.cuda.getCudaEnabledDeviceCount() > 0}')
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print(f'CUDA 디바이스 수: {cv2.cuda.getCudaEnabledDeviceCount()}')
    print('OpenCV CUDA 빌드가 정상적으로 완료되었습니다.')
else:
    print('WARNING: CUDA가 활성화되지 않았습니다.')
"

if [ $? -ne 0 ]; then
    echo "ERROR: OpenCV 검증에 실패했습니다."
    exit 1
fi

echo ""
echo "Docker Hub에 푸시합니다..."

# Docker Hub에 푸시
docker push "${BASE_IMAGE_TAG}"
docker push "${BASE_IMAGE_LATEST}"

echo ""
echo "=========================================="
echo "베이스 이미지 빌드 및 푸시 완료!"
echo "=========================================="
echo ""
echo "베이스 이미지 태그:"
echo "  ${BASE_IMAGE_TAG}"
echo "  ${BASE_IMAGE_LATEST}"
echo ""
echo "이제 애플리케이션 이미지를 빌드할 수 있습니다."
echo "애플리케이션 Dockerfile에서 이 베이스 이미지를 사용합니다."

