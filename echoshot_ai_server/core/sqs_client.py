import boto3
import json
import logging
from botocore.exceptions import ClientError

from ..config.settings import settings
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from ..domain.job import Job, TaskType

logger = logging.getLogger(__name__)


def map_processing_type(processing_type: str) -> TaskType:
    """
    Spring의 processingType을 Python의 TaskType enum으로 매핑
    
    Args:
        processing_type: Spring에서 보내는 처리 타입 (예: "AI_UPSCALING", "SUBTITLE")
        
    Returns:
        TaskType enum 값
    """
    mapping = {
        "AI_UPSCALING": TaskType.UPSCALE,
        "UPSCALE": TaskType.UPSCALE,
        "SUBTITLE": TaskType.SUBTITLE,
        "AUDIO_EXTRACT": TaskType.AUDIO_EXTRACT,
        "AUDIOEXTRACT": TaskType.AUDIO_EXTRACT,
    }
    # 대소문자 구분 없이 매핑
    normalized = processing_type.upper().replace("_", "")
    for key, value in mapping.items():
        if normalized == key.upper().replace("_", ""):
            return value
    # 기본값: UPSCALE
    logger.warning(f"Unknown processing_type: {processing_type}, defaulting to UPSCALE")
    return TaskType.UPSCALE


def convert_video_metadata(video_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Spring의 VideoMetadata 객체를 Python dict로 변환
    
    Args:
        video_metadata: Spring VideoMetadata 객체 (dict 형태)
        
    Returns:
        변환된 metadata dict
    """
    if not video_metadata:
        return None
    
    return {
        "duration_seconds": video_metadata.get("durationSeconds"),
        "width": video_metadata.get("width"),
        "height": video_metadata.get("height"),
        "codec": video_metadata.get("codec"),
        "bitrate": video_metadata.get("bitrate"),
        "frame_rate": video_metadata.get("frameRate"),
    }


def parse_sqs_message_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Spring JobMessage 형식(camelCase) 또는 기존 형식(snake_case)을 
    Python Job 모델에 맞는 형식으로 변환
    
    Args:
        body: SQS 메시지 본문 (JSON 파싱된 dict)
        
    Returns:
        Job 생성에 필요한 필드들을 포함한 dict
        
    Raises:
        ValueError: 필수 필드가 누락되거나 형식이 잘못된 경우
        KeyError: 필드명이 잘못된 경우
    """
    # Spring 형식 (camelCase) 지원
    if "jobId" in body:
        # 필수 필드 검증
        required_fields = ["jobId", "s3Key", "memberId", "processingType"]
        missing_fields = [field for field in required_fields if field not in body]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        # Spring JobMessage 형식
        try:
            job_id = str(body["jobId"])
            source_s3_key = body["s3Key"]
            user_id = str(body["memberId"])
            processing_type = body["processingType"]
            task_type = map_processing_type(processing_type)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Error parsing basic fields: {e}")
        
        # metadata 구성
        metadata = {}
        try:
            if "videoId" in body and body["videoId"] is not None:
                metadata["video_id"] = str(body["videoId"])
            if "videoMetadata" in body and body["videoMetadata"] is not None:
                video_metadata = convert_video_metadata(body["videoMetadata"])
                if video_metadata:
                    metadata["video_metadata"] = video_metadata
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error parsing metadata (will continue without metadata): {e}")
        
        return {
            "job_id": job_id,
            "user_id": user_id,
            "task_type": task_type,
            "source_s3_key": source_s3_key,
            "parameters": body.get("parameters", {}),
            "metadata": metadata if metadata else None,
        }
    else:
        # 기존 형식 (snake_case) - 하위 호환성
        return {
            "job_id": body["job_id"],
            "user_id": body.get("user_id") or body.get("member_id", ""),
            "task_type": TaskType(body["task_type"]),
            "source_s3_key": body["source_s3_key"],
            "parameters": body.get("parameters", {}),
            "metadata": body.get("metadata"),
        }

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
                WaitTimeSeconds=20,  # Long polling
                AttributeNames=['ApproximateReceiveCount']  # 재시도 횟수 확인용
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
                    # 재시도 횟수 확인 (3번 이상이면 스킵)
                    receive_count = int(msg.get('Attributes', {}).get('ApproximateReceiveCount', 1))
                    if receive_count > 3:
                        logger.warning(f"Message received {receive_count} times (>3), deleting without processing: message_id={msg.get('MessageId', 'unknown')}")
                        self.delete_message(msg['ReceiptHandle'])
                        continue
                    
                    body = json.loads(msg['Body'])
                    
                    # 메시지 형식 미리 확인
                    logger.debug(f"Raw SQS message body keys: {list(body.keys())} (receive_count={receive_count})")
                    if "jobId" in body:
                        logger.debug(f"Detected Spring format message with fields: jobId={body.get('jobId')}, memberId={body.get('memberId')}, processingType={body.get('processingType')}")
                    
                    # Spring 형식 또는 기존 형식으로 파싱
                    parsed = parse_sqs_message_body(body)
                    
                    # Job 객체 생성
                    job = Job(
                        job_id=parsed['job_id'],
                        user_id=parsed['user_id'],
                        task_type=parsed['task_type'],
                        source_s3_key=parsed['source_s3_key'],
                        parameters=parsed['parameters'],
                        receipt_handle=msg['ReceiptHandle'],
                        metadata=parsed.get('metadata')
                    )
                    jobs.append(job)
                    logger.debug(f"Successfully converted message to Job: {job.job_id} (task_type={job.task_type}, user_id={job.user_id}, s3_key={parsed['source_s3_key']})")
                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    raw_body = msg.get('Body', '')[:500]
                    logger.error(f"Invalid message format: {e}, message_id={msg.get('MessageId', 'unknown')}, body={raw_body}...")
                    # 잘못된 메시지 삭제
                    self.delete_message(msg['ReceiptHandle'])

            if jobs:
                logger.info(f"Successfully converted {len(jobs)} message(s) to Job objects")

            return jobs

        except ClientError as e:
            logger.error(f"Failed to receive messages from SQS: {e}")
            return []

    def delete_message(self, receipt_handle: str) -> bool:
        """메시지 삭제 (처리 완료)"""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info("Message deleted from SQS queue successfully")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete message from SQS: {e}")
            return False

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
