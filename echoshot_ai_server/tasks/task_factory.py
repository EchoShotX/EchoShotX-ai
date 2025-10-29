from typing import Type
from ..domain.job import *
from base import *

class TaskFactory:
    """Task Factory Pattern"""

    """ 추가 가능 """
    _task_map: Dict[TaskType, Type[BaseTask]] = {
        # TaskType.UPSCALE: UpscaleTask,
        # TaskType.SUBTITLE: SubtitleTask,
        # TaskType.AUDIO_EXTRACT: AudioExtractTask,
    }

    @classmethod
    def create_task(cls, job: Job, s3_client, temp_dir: Path) -> BaseTask:
        """작업 타입에 맞는 Task 인스턴스 생성"""
        task_class = cls._task_map.get(job.task_type)

        if not task_class:
            raise ValueError(f"Unknown task type: {job.task_type}")

        return task_class(job, s3_client, temp_dir)

    @classmethod
    def register_task(cls, task_type: TaskType, task_class: Type[BaseTask]):
        """새로운 Task 타입 등록 (확장성)"""
        cls._task_map[task_type] = task_class