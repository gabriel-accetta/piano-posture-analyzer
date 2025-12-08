import tempfile
import shutil
import os
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from .features import BodyPreprocessor
from .inference import classify_posture


class BodyService:
    def __init__(self):
        self.processor = BodyPreprocessor()

    def analyze_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single BGR image and return detected body features and
        classification label.

        Returns: {"features": [...], "label": str}
        """
        features = self.processor.extract_features(image)

        if not features:
            return {"features": None, "label": "NO_POSE_DETECTED"}

        # classify_posture expects a 2D array-like
        pred_label, label_text = classify_posture(np.array([features]))

        return {"features": features, "label": str(label_text)}

    def analyze_video(self, file_path: str, every_n_seconds: int = 5) -> Dict[str, List[Tuple[int, str]]]:
        """
        Read a video from `file_path`, sample frames every `every_n_seconds`,
        run analysis and return list of timestamped classifications.
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval_frames = max(1, int(round(fps * every_n_seconds)))

        results: List[Dict[str, Any]] = []

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                timestamp = int(round(frame_idx / fps))
                analysis = self.analyze_frame(frame)

                results.append({
                    "timestamp": timestamp,
                    "features": analysis.get("features"),
                    "label": analysis.get("label")
                })

            frame_idx += 1

        cap.release()

        return {"body_classification": results}

    def analyze_uploadfile_video(self, upload_file) -> Dict[str, List[Tuple[int, str]]]:
        """
        Convenience wrapper to accept a FastAPI UploadFile-like object, save
        it to a temp file and call `analyze_video`.
        """
        suffix = os.path.splitext(upload_file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            try:
                shutil.copyfileobj(upload_file.file, tmp)
                tmp.flush()
                tmp_path = tmp.name
            finally:
                # keep file until processing finishes
                pass

        try:
            result = self.analyze_video(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return result


body_service = BodyService()
