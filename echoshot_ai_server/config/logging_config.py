import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""

    # 로그 포맷
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )

    # 로그 디렉토리 생성 (없으면 자동 생성)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "video_ai_server.log"

    # 루트 로거 설정
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_file,
                encoding='utf-8'
            )
        ]
    )

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)