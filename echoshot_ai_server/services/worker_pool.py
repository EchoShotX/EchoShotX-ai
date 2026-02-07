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

                # Job 처리
                result = self.job_processor.process_job(job)

                # SQS 메시지 삭제 (성공/실패 관계없이 항상 삭제)
                # 재시도는 SQS의 ApproximateReceiveCount로 관리됨
                deleted = self.sqs_client.delete_message(job.receipt_handle)
                
                if result.status == JobStatus.COMPLETED:
                    logger.info(f"Job {job.job_id} completed successfully, SQS message deleted={deleted}")
                else:
                    logger.warning(f"Job {job.job_id} failed, SQS message deleted={deleted}")

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