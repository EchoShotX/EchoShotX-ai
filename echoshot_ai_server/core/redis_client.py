"""
Redis 클라이언트 모듈

외부 EC2 인스턴스의 Redis 서버에 연결하여
작업 진행률을 Pub/Sub으로 전송하는 기능을 제공합니다.
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError, TimeoutError
except ImportError:
    redis = None
    RedisError = Exception
    ConnectionError = Exception
    TimeoutError = Exception

from ..config.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis 클라이언트 래퍼
    
    주요 기능:
    - 진행률 Pub/Sub 발행 (publish)
    - 연결 상태 관리 및 재연결
    - 에러 핸들링 (연결 실패 시에도 작업 중단 방지)
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        password: str = None,
        db: int = None,
        socket_timeout: float = None,
        retry_on_timeout: bool = None
    ):
        """
        Redis 클라이언트 초기화
        
        Args:
            host: Redis 서버 호스트 (기본값: settings.REDIS_HOST)
            port: Redis 서버 포트 (기본값: settings.REDIS_PORT)
            password: Redis 비밀번호 (기본값: settings.REDIS_PASSWORD)
            db: 데이터베이스 번호 (기본값: settings.REDIS_DB)
            socket_timeout: 소켓 타임아웃 (기본값: settings.REDIS_SOCKET_TIMEOUT)
            retry_on_timeout: 타임아웃 시 재시도 여부 (기본값: settings.REDIS_RETRY_ON_TIMEOUT)
        """
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.password = password or settings.REDIS_PASSWORD or None  # 빈 문자열 -> None
        self.db = db if db is not None else settings.REDIS_DB
        # 타임아웃을 2초로 줄여서 빠른 fallback 가능
        self.socket_timeout = socket_timeout or min(settings.REDIS_SOCKET_TIMEOUT, 2.0)
        self.retry_on_timeout = False  # 타임아웃 시 재시도하지 않음 (빠른 fallback)
        
        self._client: Optional[redis.Redis] = None
        self._connected: bool = False
        self._connection_failed: bool = False  # 연결 실패 시 더 이상 시도하지 않음
        
        logger.info(f"Redis client initialized for {self.host}:{self.port} (db={self.db})")
    
    @property
    def client(self) -> Optional[redis.Redis]:
        """Redis 클라이언트 인스턴스 (lazy initialization)"""
        # 이전에 연결 실패한 경우 더 이상 시도하지 않음
        if self._connection_failed:
            return None
        if self._client is None:
            self._connect()
        return self._client
    
    def _connect(self) -> bool:
        """
        Redis 서버에 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            if redis is not None:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=self.db,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_timeout,
                    retry_on_timeout=self.retry_on_timeout,
                    decode_responses=True,  # 문자열로 디코딩
                    health_check_interval=30  # 30초마다 헬스체크
                )
                
                # 연결 테스트 (PING)
                self._client.ping()
                self._connected = True
                logger.info(f"Successfully connected to Redis at {self.host}:{self.port}")
                return True
            else:
                logger.warning("Redis not available, using fallback mode")
                self._client = None
                self._connected = False
                return False
            
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to connect to Redis at {self.host}:{self.port}: {e}")
            self._connected = False
            self._connection_failed = True  # 다시 시도하지 않음
            return False
        except Exception as e:
            logger.error(f"Redis error during connection: {e}")
            self._connected = False
            self._connection_failed = True  # 다시 시도하지 않음
            return False
    
    def is_connected(self) -> bool:
        """
        Redis 연결 상태 확인
        
        Returns:
            연결 상태 (True/False)
        """
        if self._client is None:
            return False
        
        try:
            self._client.ping()
            self._connected = True
            return True
        except (RedisError, ConnectionError, TimeoutError):
            self._connected = False
            return False
    
    def ensure_connection(self) -> bool:
        """
        Redis 연결 보장 (재연결 시도)
        
        Returns:
            연결 성공 여부
        """
        if self.is_connected():
            return True
        
        logger.info("Redis connection lost, attempting to reconnect...")
        return self._connect()
    
    def publish_with_retry(
        self,
        channel: str,
        message: Dict[str, Any],
        max_retries: int = 2
    ) -> bool:
        """
        재시도 로직이 포함된 메시지 발행
        
        Args:
            channel: 발행할 채널
            message: 발행할 메시지
            max_retries: 최대 재시도 횟수 (기본값 2로 줄임)
            
        Returns:
            발행 성공 여부
        """
        for attempt in range(max_retries):
            try:
                # 연결 상태 확인 (reconnect 시도)
                if self.ensure_connection() and self.client is not None:
                    success = self.publish(channel, message)
                    if success:
                        return True
                    logger.debug(f"Publish successful on attempt {attempt + 1}")
                else:
                    logger.debug(f"Redis connection unavailable on attempt {attempt + 1}")
                    # 첫 시도 실패 시 바로 실패 반환 (빠른 fallback)
                    break
            except Exception as e:
                logger.debug(f"Publish attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 0.5초 대기 후 재시도
                else:
                    logger.debug(f"Redis publish failed after {max_retries} attempts: {e}")
        
        # Redis 연결이 안되어도 실패로 처리하지 않고 계속 진행
        return False
    
    def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """
        메시지를 Redis 채널에 발행
        
        Args:
            channel: 발행할 채널 이름
            message: 발행할 메시지 (dict -> JSON 직렬화)
            
        Returns:
            발행 성공 여부
        """
        if not self.client:
            logger.debug(f"Redis not connected, skipping publish to {channel}")
            return False
        
        try:
            json_message = json.dumps(message, ensure_ascii=False, default=str)
            subscriber_count = self._client.publish(channel, json_message)
            
            logger.debug(
                f"Published to channel '{channel}' "
                f"(subscribers: {subscriber_count}): {message}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to publish to channel '{channel}': {e}")
            if isinstance(e, (RedisError, ConnectionError, TimeoutError)):
                self._connected = False
            return False
    
    def publish_progress(
        self,
        job_id: str,
        progress: float,
        status: str = "PROCESSING",
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        member_id: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> bool:
        """
        작업 진행률을 Redis Pub/Sub으로 발행
        
        채널 형식: job:{job_id}:progress
        
        Args:
            job_id: 작업 ID
            progress: 진행률 (0-100)
            status: 작업 상태 (PROCESSING, COMPLETED, FAILED 등)
            message: 추가 메시지 (선택적)
            metadata: 추가 메타데이터 (선택적)
            member_id: 사용자 ID (선택적, Spring에서 사용자별 전송용)
            video_id: 비디오 ID (선택적, 클라이언트에서 식별용)
            
        Returns:
            발행 성공 여부
        """
        channel = f"job:{job_id}:progress"
        
        payload = {
            "jobId": job_id,  # Spring과 호환성 유지 (camelCase)
            "progress": min(100.0, max(0.0, progress)),
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # 사용자 ID 추가
        if member_id:
            payload["memberId"] = member_id
            
        # 비디오 ID 추가
        if video_id:
            payload["videoId"] = video_id
        
        if message:
            payload["message"] = message
            
        if metadata:
            payload["metadata"] = metadata
        
        # Redis fallback 로직
        if self.client is not None:
            return self.publish(channel, payload)
        else:
            logger.warning(f"Redis not available, skipping progress publish for job {job_id}")
            return False
    
    def close(self) -> None:
        """Redis 연결 종료"""
        if self._client:
            try:
                self._client.close()
                logger.info("Redis connection closed")
            except RedisError as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
                self._connected = False
    
    def __enter__(self) -> "RedisClient":
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager 종료"""
        self.close()


# 싱글톤 패턴: 전역 Redis 클라이언트 인스턴스
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """
    Redis 클라이언트 싱글톤 인스턴스 반환
    
    Returns:
        RedisClient 인스턴스
    """
    global _redis_client
    
    if _redis_client is None:
        _redis_client = RedisClient()
    
    return _redis_client


def reset_redis_client() -> None:
    """
    Redis 클라이언트 인스턴스 리셋 (테스트용)
    """
    global _redis_client
    
    if _redis_client:
        _redis_client.close()
        _redis_client = None
