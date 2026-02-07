"""
진행률 보고 모듈 (ProgressReporter)

모든 작업(Task)에서 공통으로 사용할 수 있는 진행률 보고 클래스입니다.
Redis Pub/Sub을 통해 실시간으로 진행률을 Spring 서버에 전달합니다.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

from .redis_client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


class ProgressStatus(str, Enum):
    """작업 진행 상태"""
    PENDING = "PENDING"          # 대기 중
    DOWNLOADING = "DOWNLOADING"  # 입력 파일 다운로드 중
    PROCESSING = "PROCESSING"    # 처리 중
    UPLOADING = "UPLOADING"      # 결과 업로드 중
    COMPLETED = "COMPLETED"      # 완료
    FAILED = "FAILED"            # 실패


class ProgressReporter:
    """
    진행률 보고 클래스
    
    모든 Task에서 공통으로 사용할 수 있는 진행률 보고 인터페이스입니다.
    Redis Pub/Sub을 통해 진행률을 발행하며, Redis 연결 실패 시에도
    작업은 계속 진행됩니다 (fail-safe).
    
    사용 예시:
        reporter = ProgressReporter(job_id="123")
        reporter.start()
        reporter.update(25.0, "프레임 처리 중...")
        reporter.update(50.0)
        reporter.complete()
    """
    
    def __init__(
        self,
        job_id: str,
        member_id: Optional[str] = None,
        video_id: Optional[str] = None,
        redis_client: Optional[RedisClient] = None,
        report_interval: float = 5.0
    ):
        """
        ProgressReporter 초기화
        
        Args:
            job_id: 작업 ID
            member_id: 사용자 ID (Spring에서 사용자별 전송용)
            video_id: 비디오 ID (클라이언트에서 식별용)
            redis_client: Redis 클라이언트 (None이면 싱글톤 사용)
            report_interval: 최소 보고 간격 (초) - 너무 빈번한 업데이트 방지
        """
        self.job_id = job_id
        self.member_id = member_id
        self.video_id = video_id
        self._redis_client = redis_client
        self.report_interval = report_interval
        
        self._last_progress: float = 0.0
        self._last_status: ProgressStatus = ProgressStatus.PENDING
        self._enabled: bool = True
        
        logger.debug(f"ProgressReporter initialized for job {job_id} (member={member_id}, video={video_id})")
    
    @property
    def redis_client(self) -> Optional[RedisClient]:
        """Redis 클라이언트 (lazy initialization)"""
        if self._redis_client is None:
            try:
                self._redis_client = get_redis_client()
            except Exception as e:
                logger.warning(f"Failed to get Redis client: {e}")
                self._enabled = False
                return None
        return self._redis_client
    
    def _publish(
        self,
        progress: float,
        status: ProgressStatus,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        진행률을 Redis에 발행
        
        Args:
            progress: 진행률 (0-100)
            status: 작업 상태
            message: 추가 메시지
            metadata: 추가 메타데이터
            
        Returns:
            발행 성공 여부
        """
        if not self._enabled:
            return False
        
        client = self.redis_client
        if not client:
            return False
        
        try:
            success = client.publish_progress(
                job_id=self.job_id,
                progress=progress,
                status=status.value,
                message=message,
                metadata=metadata,
                member_id=self.member_id,
                video_id=self.video_id
            )
            
            if success:
                self._last_progress = progress
                self._last_status = status
                logger.debug(
                    f"Progress published: job={self.job_id}, "
                    f"progress={progress:.1f}%, status={status.value}"
                )
            
            return success
            
        except Exception as e:
            logger.warning(f"Failed to publish progress for job {self.job_id}: {e}")
            return False
    
    def start(self, message: str = "작업을 시작합니다") -> bool:
        """
        작업 시작 알림
        
        Args:
            message: 시작 메시지
            
        Returns:
            발행 성공 여부
        """
        return self._publish(
            progress=0.0,
            status=ProgressStatus.PENDING,
            message=message
        )
    
    def downloading(self, message: str = "입력 파일 다운로드 중") -> bool:
        """
        다운로드 시작 알림
        
        Args:
            message: 다운로드 메시지
            
        Returns:
            발행 성공 여부
        """
        return self._publish(
            progress=5.0,
            status=ProgressStatus.DOWNLOADING,
            message=message
        )
    
    def update(
        self,
        progress: float,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """
        진행률 업데이트
        
        Args:
            progress: 진행률 (0-100)
            message: 추가 메시지
            metadata: 추가 메타데이터
            force: 강제 발행 여부 (간격 무시)
            
        Returns:
            발행 성공 여부
        """
        # 진행률 범위 보정 (10-90 사이로 매핑, 시작/완료는 별도)
        adjusted_progress = 10.0 + (progress * 0.8)  # 10% ~ 90%
        
        return self._publish(
            progress=adjusted_progress,
            status=ProgressStatus.PROCESSING,
            message=message,
            metadata=metadata
        )
    
    def uploading(self, message: str = "결과 업로드 중") -> bool:
        """
        업로드 시작 알림
        
        Args:
            message: 업로드 메시지
            
        Returns:
            발행 성공 여부
        """
        return self._publish(
            progress=95.0,
            status=ProgressStatus.UPLOADING,
            message=message
        )
    
    def complete(
        self,
        message: str = "작업이 완료되었습니다",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        작업 완료 알림
        
        Args:
            message: 완료 메시지
            metadata: 결과 메타데이터
            
        Returns:
            발행 성공 여부
        """
        return self._publish(
            progress=100.0,
            status=ProgressStatus.COMPLETED,
            message=message,
            metadata=metadata
        )
    
    def fail(
        self,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        작업 실패 알림
        
        Args:
            error_message: 오류 메시지
            metadata: 오류 관련 메타데이터
            
        Returns:
            발행 성공 여부
        """
        return self._publish(
            progress=self._last_progress,
            status=ProgressStatus.FAILED,
            message=error_message,
            metadata=metadata
        )
    
    def disable(self) -> None:
        """진행률 보고 비활성화"""
        self._enabled = False
        logger.info(f"Progress reporting disabled for job {self.job_id}")
    
    def enable(self) -> None:
        """진행률 보고 활성화"""
        self._enabled = True
        logger.info(f"Progress reporting enabled for job {self.job_id}")
    
    @property
    def is_enabled(self) -> bool:
        """진행률 보고 활성화 여부"""
        return self._enabled
    
    @property
    def last_progress(self) -> float:
        """마지막으로 보고된 진행률"""
        return self._last_progress
    
    @property
    def last_status(self) -> ProgressStatus:
        """마지막으로 보고된 상태"""
        return self._last_status
