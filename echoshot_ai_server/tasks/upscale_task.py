import cv2
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import logging
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

from echoshot_ai_server.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class UpscaleTask(BaseTask):
    """
    비디오 업스케일링 작업
    
    지원 기능:
    - GPU/CPU 선택적 사용 (구독 서비스에 따라 선택 가능)
    - 메모리 최적화 (타일링 처리)
    - 오디오 트랙 자동 처리 (있는 경우만 병합)
    - 진행률 추적 및 로깅
    """

    # 구독 티어별 설정
    DEVICE_CONFIGS = {
        "gpu": {
            "tile": 256,  # GPU는 더 큰 타일 사용 가능
            "tile_pad": 10,
            "half_precision": True  # FP16 사용
        },
        "cpu": {
            "tile": 128,  # CPU는 작은 타일로 메모리 절약
            "tile_pad": 5,
            "half_precision": False  # CPU는 FP32만 지원
        }
    }

    def _process(self) -> Path:
        """업스케일링 수행"""
        # 파라미터 검증 및 추출
        scale_factor = self._validate_scale_factor(
            self.job.parameters.get("scale_factor", 2)
        )
        device = self.job.parameters.get("device", "cpu").lower()  # 기본값: CPU

        logger.info(f"업스케일 작업 시작 - Scale: {scale_factor}x, Device: {device.upper()}")

        output_file = self.temp_dir / f"{self.job.job_id}_upscaled.mp4"

        # 업스케일링 로직 (Real-ESRGAN)
        self._upscale_video(
            input_path=self.input_path,
            output_path=output_file,
            scale=scale_factor,
            device=device
        )

        return output_file

    def _validate_scale_factor(self, scale: int) -> int:
        """스케일 팩터 검증 (2 또는 4만 지원)"""
        if scale not in [2, 4]:
            logger.warning(f"지원하지 않는 scale_factor: {scale}, 기본값 2 사용")
            return 2
        return scale

    def _upscale_video(self, input_path: Path, output_path: Path,
                       scale: int, device: str):
        """
        Real-ESRGAN 업스케일링을 단일 ffmpeg 인코딩 파이프라인으로 수행
        - ffmpeg: 디코드(raw RGB24 stdout)
        - Python: RealESRGAN 업스케일
        - ffmpeg: 단일 인코딩(libx264) + 오디오 copy/재인코딩
        """
        import numpy as np
        import subprocess
        import cv2
        import time
        import torch

        # ---------- 1) 원본 메타 추출 ----------
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
        cap.release()

        out_w, out_h = width * scale, height * scale
        logger.info(f"비디오 정보 - {width}x{height}@{fps:.3f} -> {out_w}x{out_h} (x{scale})")

        # ---------- 2) 업스케일러 초기화 ----------
        upscaler = self._initialize_upscaler(scale, device)
        if device.lower() == "gpu" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # ---------- 3) 오디오 추출(가능하면 copy) ----------
        temp_audio = self.temp_dir / "temp_audio.aac"
        has_audio = self._extract_audio(input_path, temp_audio)

        # ---------- 4) ffmpeg 파이프 준비 (디코더/인코더) ----------
        # 디코더: 원본 → raw RGB24
        dec = subprocess.Popen([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)

        # 인코더: raw RGB24 → h264 (단일 인코딩) + (옵션)오디오
        enc_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{out_w}x{out_h}",
            "-r", f"{max(1, int(round(fps)))}",
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20"
        ]
        if has_audio:
            # 우선 copy 시도(호환 안 되면 _extract_audio에서 재인코딩하도록 처리)
            enc_cmd += ["-i", str(temp_audio), "-c:a", "copy", "-shortest"]
        enc_cmd += [str(output_path)]

        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, bufsize=10 ** 8)

        # ---------- 5) 프레임 스트리밍 처리 ----------
        frame_bytes = width * height * 3  # rgb24
        processed = 0
        t0 = time.time()

        try:
            while True:
                buf = dec.stdout.read(frame_bytes)
                if not buf:
                    break

                # RGB24 → numpy 프레임
                frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)

                # 업스케일 (RealESRGAN은 RGB 입력/출력)
                up_rgb, _ = upscaler.enhance(frame, outscale=scale)

                # 인코더에 raw RGB24로 밀어넣기
                enc.stdin.write(up_rgb.tobytes())

                processed += 1
                if total_frames > 0 and processed % max(1, total_frames // 20) == 0:
                    logger.info(f"업스케일 진행률: {processed}/{total_frames} "
                                f"({processed / total_frames * 100:.1f}%)")
        finally:
            # 파이프 정리
            try:
                dec.stdout.close()
            except Exception:
                pass
            try:
                enc.stdin.close()
            except Exception:
                pass
            dec.wait()
            enc.wait()

        elapsed = time.time() - t0
        logger.info(f"스트리밍 업스케일 완료: frames={processed}, elapsed={elapsed:.1f}s")

        # 임시 오디오 정리
        temp_audio.unlink(missing_ok=True)

    def _initialize_upscaler(self, scale: int, device: str) -> RealESRGANer:
        """
        Real-ESRGAN 업스케일러 초기화
        
        Args:
            scale: 업스케일 배율
            device: 처리 장치 ("gpu" or "cpu")
        
        Returns:
            초기화된 RealESRGANer 인스턴스
        """
        # 디바이스 설정 가져오기
        config = self.DEVICE_CONFIGS.get(device, self.DEVICE_CONFIGS["cpu"])

        # GPU 사용 가능 여부 확인
        use_gpu = device == "gpu" and torch.cuda.is_available()

        if device == "gpu" and not torch.cuda.is_available():
            logger.warning("GPU가 요청되었지만 사용 불가능합니다. CPU로 대체합니다.")
            use_gpu = False
            config = self.DEVICE_CONFIGS["cpu"]

        # RRDBNet 모델 생성
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4  # 모델은 x4로 고정, outscale로 조정
        )

        # RealESRGANer 초기화
        upscaler = RealESRGANer(
            scale=4,
            model_path='weights/RealESRGAN_x4plus.pth',
            model=model,
            tile=config["tile"],  # 타일 크기 (메모리 최적화)
            tile_pad=config["tile_pad"],
            pre_pad=0,
            half=config["half_precision"] and use_gpu,  # FP16은 GPU에서만
            device='cuda' if use_gpu else 'cpu'
        )

        logger.info(
            f"업스케일러 초기화 완료 - "
            f"Device: {'GPU' if use_gpu else 'CPU'}, "
            f"Tile: {config['tile']}, "
            f"Half Precision: {config['half_precision'] and use_gpu}"
        )

        return upscaler

    def _extract_audio(self, input_path: Path, temp_audio: Path) -> bool:
        """
        오디오 트랙 추출 (있는 경우만)
        
        Returns:
            bool: 오디오 추출 성공 여부
        """
        try:
            extract_audio_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vn",  # 비디오 제외
                "-acodec", "copy",  # 오디오 원본 유지
                str(temp_audio)
            ]
            result = subprocess.run(
                extract_audio_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                text=True
            )

            # 오디오 파일이 실제로 생성되었는지 확인
            if temp_audio.exists() and temp_audio.stat().st_size > 0:
                logger.info("오디오 트랙 추출 완료")
                return True
            else:
                logger.info("오디오 트랙이 없습니다")
                return False

        except subprocess.CalledProcessError as e:
            logger.warning(f"오디오 추출 실패 (비디오에 오디오가 없을 수 있음): {e}")
            return False

    # todo 삭제 예정
    def _process_frames(self, input_path: Path, temp_video: Path,
                        upscaler: RealESRGANer, scale: int,
                        width: int, height: int, fps: float, total_frames: int):
        """
        프레임별 업스케일 처리
        
        Args:
            input_path: 입력 비디오 경로
            temp_video: 출력 비디오 경로 (오디오 제외)
            upscaler: RealESRGANer 인스턴스
            scale: 업스케일 배율
            width: 원본 너비
            height: 원본 높이
            fps: 프레임레이트
            total_frames: 총 프레임 수
        """
        cap = cv2.VideoCapture(str(input_path))
        out_width = width * scale
        out_height = height * scale

        # H.264 코덱 사용 (더 나은 압축률)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (out_width, out_height))

        frame_count = 0
        log_interval = max(1, total_frames // 20)  # 5% 단위로 로깅

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # OpenCV → RGB → 업스케일 → BGR 변환
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                upscaled_rgb, _ = upscaler.enhance(img_rgb, outscale=scale)
                upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
                out.write(upscaled_bgr)

                frame_count += 1

                # 진행률 로깅
                if frame_count % log_interval == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"업스케일 진행률: {progress:.1f}% ({frame_count}/{total_frames})")

        finally:
            cap.release()
            out.release()
            logger.info(f"프레임 업스케일 완료: {frame_count}/{total_frames} 프레임 처리")

    # todo 삭제 예정
    def _merge_video_audio(self, temp_video: Path, temp_audio: Path,
                           output_path: Path, has_audio: bool):
        """
        업스케일된 비디오와 오디오 병합
        
        Args:
            temp_video: 업스케일된 비디오 경로 (오디오 없음)
            temp_audio: 추출된 오디오 경로
            output_path: 최종 출력 경로
            has_audio: 오디오 존재 여부
        """
        if has_audio:
            # 오디오가 있는 경우: 비디오 + 오디오 병합
            merge_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video),
                "-i", str(temp_audio),
                "-c:v", "libx264",  # H.264 인코딩
                "-preset", "medium",  # 인코딩 속도/품질 밸런스
                "-crf", "23",  # 품질 설정 (낮을수록 고품질)
                "-c:a", "aac",  # 오디오 AAC 인코딩
                "-b:a", "128k",  # 오디오 비트레이트
                "-shortest",  # 짧은 쪽에 맞춤
                str(output_path)
            ]
            logger.info("비디오 인코딩 및 오디오 병합 중...")
        else:
            # 오디오가 없는 경우: 비디오만 인코딩
            merge_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                str(output_path)
            ]
            logger.info("비디오 인코딩 중 (오디오 없음)...")

        subprocess.run(
            merge_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("최종 비디오 생성 완료")

    def _generate_output_key(self) -> str:
        """출력 S3 키 생성"""
        original_key = self.job.source_s3_key
        base_name = Path(original_key).stem
        return f"processed/upscaled/{self.job.job_id}/{base_name}_upscaled.mp4"

    def _generate_metadata(self) -> Dict[str, Any]:
        """메타데이터 생성"""
        metadata = super()._generate_metadata()

        # 비디오 정보 추가
        cap = cv2.VideoCapture(str(self.output_path))
        metadata.update({
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        })
        cap.release()

        return metadata
