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
    QUEUED = "queued"


@dataclass
class Job:
    """작업 도메인 모델"""
    job_id: str
    user_id: str
    task_type: TaskType
    source_s3_key: str
    parameters: Dict[str, Any]
    callback_url: str
    receipt_handle: str
    status: JobStatus.QUEUED
    retry_count: int = 0
    priority: int = 0
    submitted_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "version": self.version,
            "job_id": self.job_id,
            "member_id": self.member_id,
            "task_type": self.task_type.value,
            "source_s3_key": self.source_s3_key,
            "parameters": self.parameters,
            "callback_url": self.callback_url,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "submitted_at": self.submitted_at.isoformat(),
            "metadata": self.metadata or {},
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