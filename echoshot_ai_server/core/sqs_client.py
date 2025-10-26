import boto3
import json
from ..config.settings import settings
from typing import Optional, List
from dataclasses import asdict


class SQSClient:
    """SQS 클라이언트 래퍼"""

    def __init__(self):
        self.queue_url = settings.SQS_QUEUE_URL
        self.sqs_client = boto3.client('sqs', region_name=settings.AWS_REGION)