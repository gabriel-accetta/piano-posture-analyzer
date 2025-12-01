import tempfile
import shutil
import os
import math
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from .features import HandPreprocessor
from .inference import classify_posture


class HandService:
    def __init__(self):
        self.processor = HandPreprocessor()

    def analyze_frame(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze a single BGR image and return a list of detected hands with
        their handedness, features and classification.

        Returns list ordered with Left first (if present) then Right.
        Each entry: {"hand": "Left"|"Right", "features": [...], "label": (int, str)}
        """
        pairs = self.processor.extract_features(image)

        results = []
        for handedness, features in pairs:
            # classify_posture expects a 2D array-like
            pred_label, label_text = classify_posture(np.array([features]))
            results.append({
                "hand": handedness,
                "features": features,
                "label": (int(pred_label), str(label_text))
            })

        return results

    def analyze_video(self, file_path: str, every_n_seconds: int = 5) -> Dict[str, List[Tuple[int, str]]]:
        """
        Read a video from `file_path`, sample frames every `every_n_seconds`,
        run analysis and return two lists for right and left hands containing
        tuples of (timestamp_seconds, label_text).
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        interval_frames = max(1, int(round(fps * every_n_seconds)))

        left_list: List[Tuple[int, str]] = []
        right_list: List[Tuple[int, str]] = []

        frame_idx = 0
        sample_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                timestamp = int(round(frame_idx / fps))
                analysis = self.analyze_frame(frame)

                # map hands to lists
                for item in analysis:
                    hand = item.get("hand")
                    label_text = item.get("label", (None, ""))[1]
                    if hand == "Left":
                        left_list.append((timestamp, label_text))
                    else:
                        right_list.append((timestamp, label_text))

                sample_idx += 1

            frame_idx += 1

        cap.release()

        # If you want lists to be aligned by timestamps (i.e., both have value for each sampled time)
        # one could merge by timestamp; for now we return what was detected per hand.

        return {"right_hand_classification": right_list, "left_hand_classification": left_list}

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
                # keep the file until processing finishes
                pass

        try:
            result = self.analyze_video(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return result


hand_service = HandService()
