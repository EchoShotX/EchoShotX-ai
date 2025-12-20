import boto3
import json
import logging
from botocore.exceptions import ClientError

from ..config.settings import settings
from typing import Optional, List
from dataclasses import asdict

from ..domain.job import Job, TaskType

logger = logging.getLogger(__name__)

class SQSClient:
    """SQS 클라이언트 래퍼"""

    def __init__(self):
        self.queue_url = settings.SQS_QUEUE_URL
        self.sqs_client = boto3.client('sqs', region_name=settings.AWS_REGION)

    def receive_messages(self, max_messages: int = 1,
                         visibility_timeout: int = 300) -> List[Job]:
        """SQS 메시지 수신 및 Job 객체로 변환"""
        try:
            logger.debug(f"Polling SQS queue: {self.queue_url} (max_messages={max_messages})")
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                VisibilityTimeout=visibility_timeout,
                WaitTimeSeconds=20  # Long polling
            )

            messages = response.get('Messages', [])
            message_count = len(messages)
            
            if message_count > 0:
                logger.info(f"Received {message_count} message(s) from SQS queue")
            else:
                logger.debug("No messages received from SQS queue (long polling timeout)")
            
            jobs = []

            for msg in messages:
                try:
                    body = json.loads(msg['Body'])
                    job = Job(
                        job_id=body['job_id'],
                        task_type=TaskType(body['task_type']),
                        source_s3_key=body['source_s3_key'],
                        parameters=body.get('parameters', {}),
                        callback_url=body['callback_url'],
                        receipt_handle=msg['ReceiptHandle']
                    )
                    jobs.append(job)
                    logger.debug(f"Successfully converted message to Job: {job.job_id} (task_type={job.task_type})")
                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    logger.error(f"Invalid message format: {e}, message_id={msg.get('MessageId', 'unknown')}")
                    # 잘못된 메시지 삭제
                    self.delete_message(msg['ReceiptHandle'])

            if jobs:
                logger.info(f"Successfully converted {len(jobs)} message(s) to Job objects")

            return jobs

        except ClientError as e:
            logger.error(f"Failed to receive messages from SQS: {e}")
            return []

    def delete_message(self, receipt_handle: str) -> None:
        """메시지 삭제 (처리 완료)"""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.debug("Message deleted from queue")
        except ClientError as e:
            logger.error(f"Failed to delete message: {e}")

    def change_visibility(self, receipt_handle: str, timeout: int) -> None:
        """메시지 가시성 타임아웃 변경"""
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=timeout
            )
        except ClientError as e:
            logger.error(f"Failed to change visibility: {e}")
