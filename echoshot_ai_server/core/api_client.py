# services/spring_api_client.py
import requests
import logging
from typing import Dict, Any, Optional
from ..config.settings import settings
from ..domain.job import TaskResult, Job, JobStatus

logger = logging.getLogger(__name__)

class SpringAPIClient:
    """Spring API 클라이언트"""

    def __init__(self):
        self.base_url = settings.SPRING_API_BASE_URL.rstrip("/")
        self.timeout = settings.SPRING_API_TIMEOUT
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """HTTP 세션 생성"""
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'EchoShot-AI-Worker/1.0'
        })
        return session

    def _build_url(self, endpoint_path: str) -> str:
        """
        기본 URL과 엔드포인트 경로를 결합하여 전체 URL 생성
        
        Args:
            endpoint_path: 엔드포인트 경로 (예: /videos/webhook/processing-completed)
            
        Returns:
            전체 URL (예: http://api.example.com/videos/webhook/processing-completed)
        """
        return f"{self.base_url}{endpoint_path}"

    def send_callback(self, result: TaskResult, job: Job) -> None:
        """
        작업 결과를 Spring API로 콜백 전송
        
        Args:
            result: 작업 결과
            job: 원본 작업 정보 (videoId 등을 가져오기 위해 필요)
        """
        if result.status == JobStatus.COMPLETED:
            self._send_success_callback(result, job)
        elif result.status == JobStatus.FAILED:
            self._send_failure_callback(result, job)
        else:
            logger.warning(f"Unknown status {result.status} for job {result.job_id}, skipping callback")

    def _send_success_callback(self, result: TaskResult, job: Job) -> None:
        """성공 콜백 전송: POST /videos/webhook/processing-completed"""
        url = self._build_url("/videos/webhook/processing-completed")
        
        # videoId 가져오기 (Job의 metadata에서)
        video_id = None
        if job.metadata and "video_id" in job.metadata:
            try:
                video_id = int(job.metadata["video_id"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid video_id in job metadata: {job.metadata.get('video_id')}")
        
        if video_id is None:
            logger.error(f"Cannot send success callback: videoId is missing for job {result.job_id}")
            raise ValueError("videoId is required for success callback")
        
        # 처리된 영상 메타데이터 가져오기
        processed_metadata = result.metadata or {}
        
        # payload 구성 (WebhookProcessingCompletedRequest 형식)
        payload = {
            "videoId": video_id,
            "aiJobId": result.job_id,
            "processedS3Key": result.output_s3_key or "",
            "processedFileSizeBytes": result.processed_file_size_bytes,
        }
        
        # 처리된 영상 메타데이터 추가 (있는 경우만)
        if processed_metadata.get("duration_seconds") is not None:
            payload["processedDurationSeconds"] = processed_metadata.get("duration_seconds")
        if processed_metadata.get("width") is not None:
            payload["processedWidth"] = processed_metadata.get("width")
        if processed_metadata.get("height") is not None:
            payload["processedHeight"] = processed_metadata.get("height")
        if processed_metadata.get("codec"):
            payload["processedCodec"] = processed_metadata.get("codec")
        if processed_metadata.get("bitrate") is not None:
            payload["processedBitrate"] = processed_metadata.get("bitrate")
        if processed_metadata.get("frame_rate") is not None:
            payload["processedFrameRate"] = processed_metadata.get("frame_rate")
        # fps를 frame_rate로 매핑 (기존 코드 호환성)
        elif processed_metadata.get("fps") is not None:
            payload["processedFrameRate"] = processed_metadata.get("fps")
        
        # 썸네일 (있는 경우만)
        if processed_metadata.get("thumbnail_s3_key"):
            payload["thumbnailS3Key"] = processed_metadata.get("thumbnail_s3_key")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Success callback sent for job {result.job_id} (videoId={video_id}) to {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send success callback for job {result.job_id} to {url}: {e}")
            raise

    def _send_failure_callback(self, result: TaskResult, job: Job) -> None:
        """실패 콜백 전송: POST /videos/webhook/processing-failed"""
        url = self._build_url("/videos/webhook/processing-failed")
        
        # videoId 가져오기 (Job의 metadata에서)
        video_id = None
        if job.metadata and "video_id" in job.metadata:
            try:
                video_id = int(job.metadata["video_id"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid video_id in job metadata: {job.metadata.get('video_id')}")
        
        if video_id is None:
            logger.error(f"Cannot send failure callback: videoId is missing for job {result.job_id}")
            raise ValueError("videoId is required for failure callback")
        
        if not result.error_message:
            logger.error(f"Cannot send failure callback: errorMessage is missing for job {result.job_id}")
            raise ValueError("errorMessage is required for failure callback")
        
        # payload 구성 (WebhookProcessingFailedRequest 형식)
        payload = {
            "videoId": video_id,
            "aiJobId": result.job_id,
            "errorMessage": result.error_message,
        }
        
        # errorCode는 metadata에 있는 경우만 추가
        if result.metadata and result.metadata.get("error_code"):
            payload["errorCode"] = result.metadata.get("error_code")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Failure callback sent for job {result.job_id} (videoId={video_id}) to {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send failure callback for job {result.job_id} to {url}: {e}")
            raise
