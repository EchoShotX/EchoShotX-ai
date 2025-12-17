# services/spring_api_client.py
import requests
import logging
from typing import Dict, Any
from ..config.settings import settings
from ..config.api_endpoints import SpringAPIEndpoints
from ..domain.job import TaskResult

logger = logging.getLogger(__name__)

class SpringAPIClient:
    """Spring API 클라이언트"""

    def __init__(self):
        self.base_url = settings.SPRING_API_BASE_URL.rstrip("/")
        self.timeout = settings.SPRING_API_TIMEOUT
        self.session = self._create_session()
        self.endpoints = SpringAPIEndpoints()

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
            endpoint_path: 엔드포인트 경로 (예: /api/jobs/{job_id}/callback)
            
        Returns:
            전체 URL (예: http://api.example.com/api/jobs/{job_id}/callback)
        """
        return f"{self.base_url}{endpoint_path}"

    def send_callback(self, result: TaskResult) -> None:
        """작업 결과를 Spring API로 콜백 전송"""
        try:
            # TaskResult를 딕셔너리로 변환
            payload = result.to_dict()
            
            # 엔드포인트 경로 가져오기
            endpoint_path = self.endpoints.job_callback(result.job_id)
            url = self._build_url(endpoint_path)
            
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Callback sent successfully for job {result.job_id}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send callback for job {result.job_id}: {e}")
            raise
