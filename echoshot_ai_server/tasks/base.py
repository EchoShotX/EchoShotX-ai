"""
Base Task 모듈

모든 작업(Task)의 추상 베이스 클래스입니다.
Template Method Pattern을 사용하여 작업 흐름을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
from ..domain.job import *
from ..core.progress_reporter import ProgressReporter
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    추상 Task 클래스 - Strategy Pattern
    
    모든 Task는 이 클래스를 상속받아 _process() 메서드를 구현해야 합니다.
    ProgressReporter를 통해 실시간 진행률 보고가 가능합니다.
    """

    def __init__(self, job: Job, s3_client, temp_dir: Path, redis_client=None):
        """
        BaseTask 초기화
        
        Args:
            job: 작업 정보
            s3_client: S3 클라이언트
            temp_dir: 임시 디렉토리 경로
            redis_client: Redis 클라이언트 (선택적, 진행률 보고용)
        """
        self.job = job
        self.s3_client = s3_client
        self.temp_dir = temp_dir
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        
        # Job 메타데이터에서 video_id 추출
        video_id = None
        if job.metadata and "video_id" in job.metadata:
            video_id = str(job.metadata["video_id"])
        
        # 진행률 보고 초기화 (user_id, video_id 포함)
        self.progress = ProgressReporter(
            job_id=str(job.job_id),
            member_id=str(job.user_id) if job.user_id else None,
            video_id=video_id,
            redis_client=redis_client
        )

    def execute(self) -> TaskResult:
        """Template Method Pattern으로 실행 흐름 정의"""
        try:
            logger.info(f"Starting task {self.job.job_id} - {self.job.task_type}")
            
            # 작업 시작 알림
            self.progress.start(f"작업 시작: {self.job.task_type.value}")

            # 1. 입력 파일 다운로드
            self.progress.downloading()
            self.input_path = self._download_input()

            # 2. 작업 실행 (하위 클래스 구현)
            self.output_path = self._process()

            # 3. 결과 업로드
            self.progress.uploading()
            output_key = self._upload_output()

            # 4. 메타데이터 생성
            metadata = self._generate_metadata()

            # 5. 처리된 파일 크기 계산
            processed_file_size_bytes = None
            if self.output_path and self.output_path.exists():
                processed_file_size_bytes = self.output_path.stat().st_size

            # 완료 알림
            self.progress.complete(
                message="작업이 완료되었습니다",
                metadata=metadata
            )
            
            logger.info(f"Task {self.job.job_id} completed successfully")

            return TaskResult(
                job_id=self.job.job_id,
                status=JobStatus.COMPLETED,
                output_s3_key=output_key,
                metadata=metadata,
                processed_file_size_bytes=processed_file_size_bytes
            )

        except Exception as e:
            # 실패 알림
            self.progress.fail(error_message=str(e))
            
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

    def report_progress(self, percentage: float, message: Optional[str] = None) -> bool:
        """
        진행률 보고 (하위 Task에서 호출)
        
        Args:
            percentage: 진행률 (0-100, 실제 처리 작업 기준)
            message: 추가 메시지
            
        Returns:
            보고 성공 여부
        """
        return self.progress.update(percentage, message)

    @abstractmethod
    def _process(self) -> Path:
        """
        실제 처리 로직 (하위 클래스에서 구현)
        
        처리 중 self.report_progress()를 호출하여 진행률을 보고할 수 있습니다.
        
        Returns:
            처리된 출력 파일 경로
        """
        pass

    @abstractmethod
    def _generate_output_key(self) -> str:
        """출력 S3 키 생성 (하위 클래스에서 구현)"""
        pass

    @abstractmethod
    def _generate_metadata(self) -> dict:
        """메타데이터 생성 (하위 클래스에서 구현)"""
        pass