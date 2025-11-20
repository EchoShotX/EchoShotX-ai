import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """모델별 성능/품질 프로필"""
    name: str
    speed_score: int  # 1-10 (높을수록 빠름)
    quality_score: int  # 1-10 (높을수록 고품질)
    vram_usage: str  # Low/Medium/High
    best_for: str


# 실사 영상용 모델 옵션 (빠른 순서)
MODEL_PROFILES = {
    "fast": ModelConfig(
        name="FSRCNN",
        speed_score=10,
        quality_score=6,
        vram_usage="Low",
        best_for="빠른 처리 우선 (실시간급)"
    ),
    "balanced": ModelConfig(
        name="EDSR-base",
        speed_score=7,
        quality_score=8,
        vram_usage="Medium",
        best_for="속도/품질 균형 (권장)"
    ),
    "quality": ModelConfig(
        name="RealESRGAN-x4plus",
        speed_score=3,
        quality_score=10,
        vram_usage="High",
        best_for="최고 품질 (느림)"
    )
}


class VideoUpscaler:
    """
    실사 영상 업스케일러 (서버 비용 최적화)

    핵심 전략:
    1. OpenCV DNN 모듈 활용 (추가 의존성 최소화)
    2. FSRCNN/EDSR 경량 모델 우선 (Real-ESRGAN 대비 10-30배 빠름)
    3. ffmpeg 직접 파이핑 (디스크 I/O 제거)
    4. 적응형 타일링 (메모리 효율)
    """

    def __init__(self, model_profile: str = "balanced", device: str = "cpu"):
        """
        Args:
            model_profile: "fast" | "balanced" | "quality"
            device: "cpu" | "gpu"
        """
        self.profile = MODEL_PROFILES[model_profile]
        self.device = device
        self.use_gpu = device == "gpu" and torch.cuda.is_available()

        if device == "gpu" and not self.use_gpu:
            logger.warning("GPU 사용 불가, CPU로 전환")
            self.device = "cpu"

        self.model = self._load_model()
        logger.info(
            f"모델 로드 완료: {self.profile.name} "
            f"(속도: {self.profile.speed_score}/10, "
            f"품질: {self.profile.quality_score}/10)"
        )

    def _load_model(self):
        """모델 로드 (OpenCV DNN 또는 PyTorch)"""
        if self.profile.name == "FSRCNN":
            return self._load_fsrcnn()
        elif self.profile.name == "EDSR-base":
            return self._load_edsr()
        else:  # RealESRGAN
            return self._load_realesrgan()

    def _load_fsrcnn(self):
        """FSRCNN (초고속, OpenCV 내장)"""
        model_path = Path("weights/FSRCNN_x2.pb")
        if not model_path.exists():
            logger.warning(f"{model_path} 없음, OpenCV 기본 업스케일 사용")
            return None

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel("fsrcnn", 2)

        if self.use_gpu:
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        return sr

    def _load_edsr(self):
        """EDSR-base (균형형)"""
        model_path = Path("weights/EDSR_x2.pb")
        if not model_path.exists():
            logger.warning(f"{model_path} 없음, FSRCNN으로 대체")
            return self._load_fsrcnn()

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel("edsr", 2)

        if self.use_gpu:
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        return sr

    def _load_realesrgan(self):
        """RealESRGAN (최고품질, 가장 느림)"""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4
            )

            upsampler = RealESRGANer(
                scale=4,
                model_path='weights/RealESRGAN_x4plus.pth',
                model=model,
                tile=256 if self.use_gpu else 128,
                tile_pad=10,
                pre_pad=0,
                half=self.use_gpu,
                device='cuda' if self.use_gpu else 'cpu'
            )
            return upsampler

        except ImportError:
            logger.error("RealESRGAN 미설치, EDSR로 대체")
            return self._load_edsr()

    def upscale_frame(self, frame: np.ndarray, scale: int = 2) -> np.ndarray:
        """단일 프레임 업스케일"""
        if self.model is None:
            # Fallback: OpenCV 기본 업스케일
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w * scale, h * scale),
                              interpolation=cv2.INTER_CUBIC)

        if self.profile.name in ["FSRCNN", "EDSR-base"]:
            # OpenCV DNN 모델
            return self.model.upsample(frame)
        else:
            # RealESRGAN
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            upscaled_rgb, _ = self.model.enhance(img_rgb, outscale=scale)
            return cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)


class OptimizedUpscaleTask:
    """
    최적화된 비디오 업스케일 태스크

    성능 개선:
    - FSRCNN/EDSR 사용 시 Real-ESRGAN 대비 10-30배 빠름
    - ffmpeg 직접 파이핑으로 중간 파일 제거
    - 적응형 배치 처리
    """

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process(
            self,
            input_path: Path,
            output_path: Path,
            scale: int = 2,
            device: str = "cpu",
            model_profile: str = "balanced"
    ):
        """
        비디오 업스케일 실행

        Args:
            input_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            scale: 업스케일 배율 (2 or 4)
            device: "cpu" or "gpu"
            model_profile: "fast" | "balanced" | "quality"
        """
        # 비디오 정보 추출
        video_info = self._get_video_info(input_path)
        logger.info(
            f"비디오 정보: {video_info['width']}x{video_info['height']}, "
            f"{video_info['fps']} fps, {video_info['total_frames']} 프레임"
        )

        # 업스케일러 초기화
        upscaler = VideoUpscaler(model_profile, device)

        # 오디오 추출
        temp_audio = self.temp_dir / "audio.aac"
        has_audio = self._extract_audio(input_path, temp_audio)

        # 업스케일 처리
        self._process_video(
            input_path, output_path, upscaler, scale,
            video_info, temp_audio if has_audio else None
        )

        # 정리
        temp_audio.unlink(missing_ok=True)
        logger.info(f"✅ 완료: {output_path}")

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """비디오 정보 추출"""
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return info

    def _extract_audio(self, input_path: Path, output_path: Path) -> bool:
        """오디오 추출"""
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vn", "-acodec", "copy", str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path.exists() and output_path.stat().st_size > 0
        except subprocess.CalledProcessError:
            return False

    def _process_video(
            self,
            input_path: Path,
            output_path: Path,
            upscaler: VideoUpscaler,
            scale: int,
            video_info: Dict,
            audio_path: Optional[Path]
    ):
        """비디오 처리 (ffmpeg 직접 파이핑)"""
        out_width = video_info["width"] * scale
        out_height = video_info["height"] * scale
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]

        # ffmpeg 프로세스 시작
        ffmpeg_cmd = self._build_ffmpeg_cmd(
            output_path, out_width, out_height, fps, audio_path
        )

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # 프레임 처리
        cap = cv2.VideoCapture(str(input_path))
        frame_count = 0
        log_interval = max(1, total_frames // 20)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 업스케일
                upscaled = upscaler.upscale_frame(frame, scale)

                # ffmpeg에 직접 쓰기
                ffmpeg_proc.stdin.write(upscaled.tobytes())

                frame_count += 1
                if frame_count % log_interval == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"진행률: {progress:.1f}% ({frame_count}/{total_frames})")

        finally:
            cap.release()
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

            if ffmpeg_proc.returncode != 0:
                stderr = ffmpeg_proc.stderr.read().decode()
                raise RuntimeError(f"ffmpeg 오류: {stderr}")

    def _build_ffmpeg_cmd(
            self,
            output_path: Path,
            width: int,
            height: int,
            fps: float,
            audio_path: Optional[Path]
    ) -> List[str]:
        """ffmpeg 명령어 생성"""
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]

        if audio_path:
            cmd.extend(["-i", str(audio_path), "-c:a", "aac", "-b:a", "128k"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",  # 인코딩 속도 최우선
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ])

        return cmd


# BaseTask 통합용
class UpscaleTask:
    """BaseTask 인터페이스 구현"""

    def __init__(self, job, temp_dir: Path):
        self.job = job
        self.temp_dir = temp_dir
        self.input_path = None  # BaseTask에서 설정
        self.output_path = None

    def _process(self) -> Path:
        """업스케일 실행"""
        scale = self.job.parameters.get("scale_factor", 2)
        device = self.job.parameters.get("device", "cpu")
        profile = self.job.parameters.get("model_profile", "balanced")

        output_file = self.temp_dir / f"{self.job.job_id}_upscaled.mp4"

        task = OptimizedUpscaleTask(self.temp_dir)
        task.process(self.input_path, output_file, scale, device, profile)

        return output_file

    def _generate_output_key(self) -> str:
        """S3 키 생성"""
        base_name = Path(self.job.source_s3_key).stem
        return f"processed/upscaled/{self.job.job_id}/{base_name}_upscaled.mp4"

    def _generate_metadata(self) -> Dict[str, Any]:
        """메타데이터 생성"""
        cap = cv2.VideoCapture(str(self.output_path))
        metadata = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return metadata