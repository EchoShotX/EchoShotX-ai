import sys
from pathlib import Path
from config.settings import get_settings
from config.logging_config import setup_logging
from core.sqs_client import SQSClient
from core.s3_client import S3Client
from core.api_client import SpringAPIClient
from services.job_processor import JobProcessor
from services.worker_pool import WorkerPool


def main():
    """애플리케이션 진입점"""
    # 설정 로드
    settings = get_settings()

    # 로깅 설정
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting Video AI Server")
    logger.info(f"Configuration: {settings.dict()}")