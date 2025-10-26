# services/spring_api_client.py
import requests
import logging
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SpringAPIClient:
    """Spring API 클라이언트"""

    def __init__(self):
        self.base_url = settings.SPRING_API_BASE_URL.rstrip("/")
        self.timeout = settings.SPRING_API_TIMEOUT
        self.session = self._create_session()
