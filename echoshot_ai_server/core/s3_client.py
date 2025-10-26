import boto3
from pathlib import Path
import logging
from botocore.exceptions import ClientError
from ..config.settings import settings

logger = logging.getLogger(__name__)


class S3Client:
    """S3 클라이언트 래퍼"""

    def __init__(self, region: str):
        self.bucket_name = settings.S3_BUCKET_NAME
        self.s3_client = boto3.client('s3', region_name=region)