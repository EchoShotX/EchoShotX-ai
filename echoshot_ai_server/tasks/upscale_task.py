import cv2
import torch
from pathlib import Path
from typing import Dict, Any
import subprocess
from realesrgan import RealESRGANer
from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet

from echoshot_ai_server.tasks.base import BaseTask


class UpscaleTask(BaseTask):
    """비디오 업스케일링 작업"""

    def _process(self) -> Path:
        """업스케일링 수행"""
        scale_factor = self.job.parameters.get("scale_factor", 2)
        model_type = self.job.parameters.get("model", "ESRGAN")

        output_file = self.temp_dir / f"{self.job.job_id}_upscaled.mp4"

        # 업스케일링 로직 (Real-ESRGAN)
        self._upscale_video(
            input_path=self.input_path,
            output_path=output_file,
            scale=scale_factor,
            model=model_type
        )

        return output_file

    def _upscale_video(self, input_path: Path, output_path: Path,
                       scale: int, model: str):
        """Real-ESRGAN으로 비디오 업스케일링 후 오디오 포함 병합"""
        temp_video = self.temp_dir / "temp_upscaled_no_audio.mp4"
        temp_audio = self.temp_dir / "temp_audio.aac"
        # ---------------------------
        # 1. 원본 비디오 정보
        # ---------------------------
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # ---------------------------
        # 2. Real-ESRGAN 모델 초기화
        # ---------------------------
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)

        upscaler = RealESRGANer(
            scale=4,
            model_path='weights/RealESRGAN_x4plus.pth',
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available()
        )

        # ---------------------------
        # 3. 오디오 트랙 추출 (ffmpeg)
        # ---------------------------
        extract_audio_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",  # 비디오 제외
            "-acodec", "copy",  # 오디오 원본 유지
            str(temp_audio)
        ]
        subprocess.run(extract_audio_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # ---------------------------
        # 4. 프레임 업스케일 (Real-ESRGAN)
        # ---------------------------
        cap = cv2.VideoCapture(str(input_path))
        out_width = width * scale
        out_height = height * scale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (out_width, out_height))

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

        finally:
            cap.release()
            out.release()

        # ---------------------------
        # 5. 업스케일된 비디오 + 오디오 병합
        # ---------------------------
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video),
            "-i", str(temp_audio),
            "-c:v", "copy",  # 비디오는 그대로
            "-c:a", "aac",  # 오디오는 aac로 인코딩
            "-shortest",
            str(output_path)
        ]
        subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # ---------------------------
        # 6. 임시 파일 삭제 (선택)
        # ---------------------------
        temp_video.unlink(missing_ok=True)
        temp_audio.unlink(missing_ok=True)


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
