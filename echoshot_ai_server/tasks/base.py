from abc import ABC, abstractmethod
from ..domain.job import *
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """추상 Task 클래스 - Strategy Pattern"""

    def __init__(self, job: Job, s3_client, temp_dir: Path):
        self.job = job
        self.s3_client = s3_client
        self.temp_dir = temp_dir
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None

    def execute(self) -> TaskResult:
        """Template Method Pattern으로 실행 흐름 정의"""
        try:
            logger.info(f"Starting task {self.job.job_id} - {self.job.task_type}")

            # 1. 입력 파일 다운로드
            self.input_path = self._download_input()

            # 2. 작업 실행 (하위 클래스 구현)
            self.output_path = self._process()

            # 3. 결과 업로드
            output_key = self._upload_output()

            # 4. 메타데이터 생성
            metadata = self._generate_metadata()

            logger.info(f"Task {self.job.job_id} completed successfully")

            return TaskResult(
                job_id=self.job.job_id,
                status=JobStatus.COMPLETED,
                output_s3_key=output_key,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Task {self.job.job_id} failed: {str(e)}", exc_info=True)
            return TaskResult(
                job_id=self.job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e)
            )
        finally:
            self._cleanup()

    def _download_input(self) -> Path:
        """S3에서 입력 파일 다운로드"""
        input_file = self.temp_dir / f"{self.job.job_id}_input.mp4"
        self.s3_client.download_file(self.job.source_s3_key, input_file)
        logger.info(f"Downloaded input file: {input_file}")
        return input_file

    def _upload_output(self) -> str:
        """S3로 결과 파일 업로드"""
        output_key = self._generate_output_key()
        self.s3_client.upload_file(self.output_path, output_key)
        logger.info(f"Uploaded output file: {output_key}")
        return output_key

    def _cleanup(self):
        """임시 파일 정리"""
        for path in [self.input_path, self.output_path]:
            if path and path.exists():
                path.unlink()
                logger.debug(f"Cleaned up: {path}")