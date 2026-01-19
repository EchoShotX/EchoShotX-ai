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
    SQS_QUEUE_URL: str = ""  # SQS 큐 URL (개발 환경에서는 선택적)
    S3_BUCKET_NAME: str = ""  # S3 버킷 이름 (개발 환경에서는 선택적)

    # ===============================
    # Spring API 설정
    # ===============================
    SPRING_API_BASE_URL: str = "http://localhost:8080"  # Spring 서버의 기본 URL (개발 환경 기본값)
    SPRING_API_TIMEOUT: int = 30  # API 요청 타임아웃 (초 단위)
    CALLBACK_URL: str = ""  # 작업 완료 후 콜백을 받을 고정 URL (.env에서 설정)

    # ===============================
    # Worker 설정
    # ===============================
    WORKER_COUNT: int = 4  # 동시에 처리할 워커 수
    MAX_RETRIES: int = 3  # 재시도 횟수
    VISIBILITY_TIMEOUT: int = 300  # SQS 메시지 가시성 타임아웃 (초 단위)

    # ===============================
    # Redis 설정 (진행률 Pub/Sub용)
    # ===============================
    REDIS_HOST: str = "localhost"  # Redis 서버 호스트 (EC2-B의 Redis)
    REDIS_PORT: int = 6379  # Redis 서버 포트
    REDIS_PASSWORD: str = ""  # Redis 비밀번호 (선택적)
    REDIS_DB: int = 0  # Redis 데이터베이스 번호
    REDIS_SOCKET_TIMEOUT: float = 5.0  # 소켓 타임아웃 (초)
    REDIS_RETRY_ON_TIMEOUT: bool = True  # 타임아웃 시 재시도 여부

    # ===============================
    # 비디오 처리 설정
    # ===============================
    TEMP_DIR: Path = Path(os.getenv("TEMP", "/tmp")) / "video_processing"  # 임시 비디오 저장 디렉토리 (Windows/Linux 호환)
    MAX_VIDEO_SIZE_MB: int = 500  # 업로드 가능한 최대 비디오 크기(MB)

    # ===============================
    # 로깅 설정
    # ===============================
    LOG_LEVEL: str = "INFO"  # 로그 레벨 (DEBUG, INFO, WARNING, ERROR 등)
    
    # ===============================
    # 환경 설정
    # ===============================
    APP_ENV: str = "dev"  # 환경 구분 (dev, prod)

    class Config:
        """
        Pydantic Settings Config
        - env_file: .env 파일 경로 지정 (여러 파일 시도)
        - case_sensitive: 환경 변수 대소문자 구분 여부
        - 개발 환경: .env 파일 우선 사용
        - 프로덕션 환경: .env.prod 파일 사용
        """
        env_file = [".env", ".env.prod"]  # .env 파일 우선, 없으면 .env.prod 시도
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
