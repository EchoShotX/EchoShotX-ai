"""
Task Factory 모듈

작업 타입에 맞는 Task 인스턴스를 생성하는 Factory Pattern 구현
"""

from typing import Type, Optional
from pathlib import Path
from ..domain.job import Job, TaskType, Dict
from .base import BaseTask
from .upscale_task import UpscaleTask
from ..core.redis_client import RedisClient


class TaskFactory:
    """Task Factory Pattern"""

    """ 등록된 Task 타입 """
    _task_map: Dict[TaskType, Type[BaseTask]] = {
        TaskType.UPSCALE: UpscaleTask,
        # TaskType.SUBTITLE: SubtitleTask,
        # TaskType.AUDIO_EXTRACT: AudioExtractTask,
    }

    @classmethod
    def create_task(
        cls, 
        job: Job, 
        s3_client, 
        temp_dir: Path,
        redis_client: Optional[RedisClient] = None
    ) -> BaseTask:
        """
        작업 타입에 맞는 Task 인스턴스 생성
        
        Args:
            job: 작업 정보
            s3_client: S3 클라이언트
            temp_dir: 임시 디렉토리
            redis_client: Redis 클라이언트 (진행률 보고용, 선택적)
            
        Returns:
            BaseTask 인스턴스
        """
        task_class = cls._task_map.get(job.task_type)

        if not task_class:
            raise ValueError(f"Unknown task type: {job.task_type}")

        return task_class(job, s3_client, temp_dir, redis_client)

    @classmethod
    def register_task(cls, task_type: TaskType, task_class: Type[BaseTask]):
        """새로운 Task 타입 등록 (확장성)"""
        cls._task_map[task_type] = task_class