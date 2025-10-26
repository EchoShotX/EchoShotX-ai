# services/spring_api_client.py
import requests
import logging
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SpringAPIClient:
    """Spring API 클라이언트"""

    def __init__(self):
        # Settings에서 값 가져오기
        self.base_url = settings.SPRING_API_BASE_URL.rstrip("/")
        self.timeout = settings.SPRING_API_TIMEOUT
        self.session = self._create_session()
