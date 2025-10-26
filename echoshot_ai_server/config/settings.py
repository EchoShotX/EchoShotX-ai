from functools import lru_cache
from pathlib import Path
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    환경 설정 관리 클래스.
    - Pydantic의 BaseSettings를 상속하여 .env 및 환경 변수를 자동 로드함
    - 타입 변환(int, bool 등)을 자동 처리
    - 누락 시 ValidationError 발생 → 조기 오류 탐지 가능
    """

    # ===============================
    # AWS 설정
    # ===============================
    AWS_REGION: str = "ap-northeast-2"  # AWS 리전 (기본값 설정)
    SQS_QUEUE_URL: str  # SQS 큐 URL
    S3_BUCKET_NAME: str  # S3 버킷 이름

    # ===============================
    # Spring API 설정
    # ===============================
    SPRING_API_BASE_URL: str  # Spring 서버의 기본 URL
    SPRING_API_TIMEOUT: int = 30  # API 요청 타임아웃 (초 단위)

    # ===============================
    # Worker 설정
    # ===============================
    WORKER_COUNT: int = 4  # 동시에 처리할 워커 수
    MAX_RETRIES: int = 3  # 재시도 횟수
    VISIBILITY_TIMEOUT: int = 300  # SQS 메시지 가시성 타임아웃 (초 단위)

    # ===============================
    # 비디오 처리 설정
    # ===============================
    TEMP_DIR: Path = Path("/tmp/video_processing")  # 임시 비디오 저장 디렉토리
    MAX_VIDEO_SIZE_MB: int = 500  # 업로드 가능한 최대 비디오 크기(MB)

    # ===============================
    # 로깅 설정
    # ===============================
    LOG_LEVEL: str = "INFO"  # 로그 레벨 (DEBUG, INFO, WARNING, ERROR 등)

    class Config:
        """
        Pydantic Settings Config
        - env_file: .env 파일 경로 지정
        - case_sensitive: 환경 변수 대소문자 구분 여부
        - APP_ENV 값(dev, prod 등)에 따라 다른 .env 파일 로드 가능
        """
        env_file = f".env.{os.getenv('APP_ENV', 'dev')}"  # 기본은 .env.dev
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Settings 인스턴스를 싱글톤으로 캐싱하여
    불필요한 .env 파일 재로딩을 방지합니다.
    (성능 향상 및 일관성 유지)
    """
    return Settings()


# 전역에서 settings를 바로 사용 가능
settings = get_settings()
