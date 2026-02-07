"""
Job Processor 모듈

Job 처리 서비스 - 단일 책임 원칙 적용
Redis Pub/Sub을 통해 실시간 진행률을 보고합니다.
"""

from pathlib import Path
import logging
from typing import Optional
from ..core.api_client import SpringAPIClient
from ..core.s3_client import S3Client
from ..core.redis_client import RedisClient, get_redis_client
from ..domain.job import Job, TaskResult, JobStatus
from ..tasks.task_factory import TaskFactory

logger = logging.getLogger(__name__)


class JobProcessor:
    """Job 처리 서비스 - 단일 책임 원칙"""

    def __init__(
        self, 
        s3_client: S3Client, 
        api_client: SpringAPIClient,
        temp_dir: Path, 
        max_retries: int = 3,
        redis_client: Optional[RedisClient] = None
    ):
        """
        JobProcessor 초기화
        
        Args:
            s3_client: S3 클라이언트
            api_client: Spring API 클라이언트
            temp_dir: 임시 디렉토리
            max_retries: 최대 재시도 횟수
            redis_client: Redis 클라이언트 (진행률 보고용, None이면 자동 생성)
        """
        self.s3_client = s3_client
        self.api_client = api_client
        self.temp_dir = temp_dir
        self.max_retries = max_retries
        
        # Redis 클라이언트 초기화 (선택적)
        self._redis_client = redis_client
    
    @property
    def redis_client(self) -> Optional[RedisClient]:
        """Redis 클라이언트 (lazy initialization)"""
        if self._redis_client is None:
            try:
                self._redis_client = get_redis_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Redis client: {e}")
                return None
        return self._redis_client

    def process_job(self, job: Job) -> TaskResult:
        """Job 처리 메인 로직"""
        logger.info(f"Processing job {job.job_id}")

        result = None
        try:
            # 1. Task 생성 (Redis 클라이언트 전달)
            task = TaskFactory.create_task(
                job, 
                self.s3_client, 
                self.temp_dir,
                redis_client=self.redis_client
            )

            # 2. Task 실행
            result = task.execute()

        except Exception as e:
            logger.error(f"Job {job.job_id} processing failed: {e}", exc_info=True)

            # 실패 결과 생성
            result = TaskResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e)
            )

        # 3. 콜백 전송 (성공/실패 관계없이 한 번만 시도)
        try:
            self._send_callback_with_retry(result, job)
        except Exception as callback_error:
            # 콜백 실패는 로깅만 하고 재시도하지 않음 (무한 재시도 방지)
            logger.error(f"Failed to send callback for job {job.job_id} after {self.max_retries} retries: {callback_error}")

        return result

    def _send_callback_with_retry(self, result: TaskResult, job: Job) -> None:
        """재시도 로직이 포함된 콜백 전송"""
        for attempt in range(self.max_retries):
            try:
                self.api_client.send_callback(result, job)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Callback retry {attempt + 1}/{self.max_retries}: {e}")