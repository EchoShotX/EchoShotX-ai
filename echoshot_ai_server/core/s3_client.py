import boto3
from pathlib import Path
import logging
from botocore.exceptions import ClientError
from ..config.settings import settings

logger = logging.getLogger(__name__)


class S3Client:
    """S3 클라이언트 래퍼"""

    def __init__(self):
        self.bucket_name = settings.S3_BUCKET_NAME
        self.s3_client = boto3.client("s3", region_name=settings.AWS_REGION)

    def download_file(self, s3_key: str, local_path: Path) -> None:
        """S3에서 파일 다운로드"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

    def upload_file(self, local_path: Path, s3_key: str) -> None:
        """S3로 파일 업로드"""
        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': self._get_content_type(local_path)}
            )
            logger.info(f"Uploaded to s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to upload {s3_key}: {e}")
            raise

    def _get_content_type(self, path: Path) -> str:
        """파일 확장자로 Content-Type 결정"""
        ext_map = {
            '.mp4': 'video/mp4',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.aac': 'audio/aac'
        }
        return ext_map.get(path.suffix.lower(), 'application/octet-stream')