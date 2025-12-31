import sys
import logging
from pathlib import Path
from echoshot_ai_server.config.settings import get_settings
from echoshot_ai_server.config.logging_config import setup_logging
from echoshot_ai_server.core.sqs_client import SQSClient
from echoshot_ai_server.core.s3_client import S3Client
from echoshot_ai_server.core.api_client import SpringAPIClient
from echoshot_ai_server.services.job_processor import JobProcessor
from echoshot_ai_server.services.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


def main():
    """애플리케이션 진입점"""
    # 설정 로드
    settings = get_settings()

    # 로깅 설정
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting Video AI Server")
    logger.info(f"Configuration: {settings.model_dump()}")

    # 임시 디렉토리 생성
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 클라이언트 초기화
    sqs_client = SQSClient()
    s3_client = S3Client()
    api_client = SpringAPIClient()

    # Job Processor 초기화
    job_processor = JobProcessor(
        s3_client=s3_client,
        api_client=api_client,
        temp_dir=temp_dir,
        max_retries=settings.MAX_RETRIES
    )

    # Worker Pool 초기화 및 시작
    worker_pool = WorkerPool(
        worker_count=settings.WORKER_COUNT,
        sqs_client=sqs_client,
        job_processor=job_processor
    )

    try:
        logger.info("Starting worker pool...")
        worker_pool.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()