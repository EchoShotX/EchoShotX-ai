from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class TaskType(str, Enum):
    """작업 타입 열거형"""
    UPSCALE = "upscale"
    SUBTITLE = "subtitle"
    AUDIO_EXTRACT = "audio_extract"


class JobStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """작업 도메인 모델"""
    job_id: str
    task_type: TaskType
    source_s3_key: str
    parameters: Dict[str, Any]
    callback_url: str
    receipt_handle: str
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "job_id": self.job_id,
            "task_type": self.task_type.value,
            "source_s3_key": self.source_s3_key,
            "parameters": self.parameters,
            "callback_url": self.callback_url,
            "retry_count": self.retry_count
        }


@dataclass
class TaskResult:
    """작업 결과 모델"""
    job_id: str
    status: JobStatus
    output_s3_key: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "output_s3_key": self.output_s3_key,
            "error_message": self.error_message,
            "metadata": self.metadata or {}
        }