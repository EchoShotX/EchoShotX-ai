import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import List
import signal
import time
import logging
from echoshot_ai_server.core.sqs_client import SQSClient
from echoshot_ai_server.domain.job import JobStatus
from echoshot_ai_server.services.job_processor import JobProcessor


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

        while not self.should_stop.is_set():
            try:
                # SQS에서 메시지 수신
                jobs = self.sqs_client.receive_messages(
                    max_messages=self.worker_count
                )

                for job in jobs:
                    # Job Queue에 추가
                    self.job_queue.put(job, timeout=5)
                    logger.info(f"Job {job.job_id} queued")

                # 메시지가 없으면 짧은 대기
                if not jobs:
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)
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

                # 성공 시 SQS 메시지 삭제
                if result.status == JobStatus.COMPLETED:
                    self.sqs_client.delete_message(job.receipt_handle)
                else:
                    # 실패 시 재시도 로직
                    if job.retry_count < self.job_processor.max_retries:
                        # 가시성 타임아웃 연장 (재시도)
                        self.sqs_client.change_visibility(
                            job.receipt_handle,
                            timeout=60
                        )
                    else:
                        # 최대 재시도 초과 시 삭제
                        self.sqs_client.delete_message(job.receipt_handle)
                        logger.error(f"Job {job.job_id} exceeded max retries")

            except Queue.Empty:
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