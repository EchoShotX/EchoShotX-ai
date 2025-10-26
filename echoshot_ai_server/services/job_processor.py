from pathlib import Path
import logging
from echoshot_ai_server.core.api_client import SpringAPIClient
from echoshot_ai_server.core.s3_client import S3Client
from echoshot_ai_server.domain.job import Job, TaskResult, JobStatus
from echoshot_ai_server.tasks.task_factory import TaskFactory

logger = logging.getLogger(__name__)

class JobProcessor:
    """Job 처리 서비스 - 단일 책임 원칙"""

    def __init__(self, s3_client: S3Client, api_client: SpringAPIClient,
                 temp_dir: Path, max_retries: int = 3):
        self.s3_client = s3_client
        self.api_client = api_client
        self.temp_dir = temp_dir
        self.max_retries = max_retries

    def process_job(self, job: Job) -> TaskResult:
        """Job 처리 메인 로직"""
        logger.info(f"Processing job {job.job_id}")

        try:
            # 1. Task 생성
            task = TaskFactory.create_task(job, self.s3_client, self.temp_dir)

            # 2. Task 실행
            result = task.execute()

            # 3. 콜백 전송
            self._send_callback_with_retry(result)

            return result

        except Exception as e:
            logger.error(f"Job {job.job_id} processing failed: {e}", exc_info=True)

            # 실패 결과 생성
            result = TaskResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e)
            )

            # 실패 콜백 전송 시도
            try:
                self._send_callback_with_retry(result)
            except Exception as callback_error:
                logger.error(f"Failed to send failure callback: {callback_error}")

            return result

    def _send_callback_with_retry(self, result: TaskResult) -> None:
        """재시도 로직이 포함된 콜백 전송"""
        for attempt in range(self.max_retries):
            try:
                self.api_client.send_callback(result)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Callback retry {attempt + 1}/{self.max_retries}")