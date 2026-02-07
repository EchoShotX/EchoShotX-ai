import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Empty
from typing import List
import signal
import time
import logging
from ..core.sqs_client import SQSClient
from ..domain.job import JobStatus
from .job_processor import JobProcessor


logger = logging.getLogger(__name__)

class WorkerPool:
    """Worker Pool 관리 - 멀티프로세싱 기반"""

    def __init__(self, worker_count: int, sqs_client: SQSClient,
                 job_processor: JobProcessor):
        self.worker_count = worker_count
        self.sqs_client = sqs_client
        self.job_processor = job_processor
        self.workers: List[Process] = []
        self.job_queue = Queue(maxsize=worker_count * 2)
        self.should_stop = mp.Event()

    def start(self) -> None:
        """Worker Pool 시작"""
        logger.info(f"Starting worker pool with {self.worker_count} workers")

        # Signal 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Worker 프로세스 시작
        for i in range(self.worker_count):
            worker = Process(target=self._worker_loop, args=(i,))
            worker.start()
            self.workers.append(worker)
            logger.info(f"Worker {i} started (PID: {worker.pid})")

        # SQS Polling 루프
        self._polling_loop()

    def _polling_loop(self) -> None:
        """SQS 메시지 polling"""
        logger.info("Starting SQS polling loop")
        
        no_message_count = 0  # 메시지가 없을 때 카운터 (로그 스팸 방지)

        while not self.should_stop.is_set():
            try:
                # SQS에서 메시지 수신
                jobs = self.sqs_client.receive_messages(
                    max_messages=self.worker_count
                )

                if jobs:
                    logger.info(f"Polling result: received {len(jobs)} job(s) from SQS")
                    no_message_count = 0  # 카운터 리셋
                    
                    for job in jobs:
                        # Job Queue에 추가
                        self.job_queue.put(job, timeout=5)
                        logger.info(f"Job {job.job_id} queued (task_type={job.task_type}, source_s3_key={job.source_s3_key})")
                else:
                    # 메시지가 없을 때 주기적으로 로그 (60초마다)
                    no_message_count += 1
                    if no_message_count >= 60:  # 약 60초마다 로그 (1초 sleep * 60)
                        logger.debug("No messages in SQS queue, continuing to poll...")
                        no_message_count = 0
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)
                no_message_count = 0  # 에러 시 카운터 리셋
                time.sleep(5)

        logger.info("Polling loop stopped")

    def _worker_loop(self, worker_id: int) -> None:
        """Worker 프로세스 메인 루프"""
        logger.info(f"Worker {worker_id} started")

        while not self.should_stop.is_set():
            try:
                # Job Queue에서 작업 가져오기
                job = self.job_queue.get(timeout=5)

                logger.info(f"Worker {worker_id} processing job {job.job_id}")

                # 1. 즉시 삭제 (At-most-once delivery 보장, 중복 실행 방지)
                self.sqs_client.delete_message(job.receipt_handle)
                logger.info(f"Job {job.job_id} message deleted from SQS immediately to prevent duplicate processing")

                # 2. Job 처리
                result = self.job_processor.process_job(job)

                # 3. 실패 시 재시도 처리 (Application-level retry)
                if result.status != JobStatus.COMPLETED:
                    if job.retry_count < self.job_processor.max_retries:
                        # 재시도 횟수 증가하여 새 메시지 발행
                        retry_payload = job.to_dict()
                        retry_payload['customRetryCount'] = job.retry_count + 1
                        
                        logger.warning(f"Job {job.job_id} failed. Re-queueing for retry ({retry_payload['customRetryCount']}/{self.job_processor.max_retries})")
                        self.sqs_client.send_message(retry_payload, delay_seconds=60) # 60초 딜레이
                    else:
                        logger.error(f"Job {job.job_id} exceeded max retries ({job.retry_count}). Dropping message.")
                
                # 성공 시에는 이미 삭제했으므로 추가 동작 없음
                if result.status == JobStatus.COMPLETED:
                    logger.info(f"Job {job.job_id} completed successfully")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    def _signal_handler(self, signum, frame):
        """Graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.should_stop.set()

        # Worker들이 종료될 때까지 대기
        for worker in self.workers:
            worker.join(timeout=30)
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker.pid}")
                worker.terminate()

        logger.info("Shutdown complete")