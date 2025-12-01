from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any

from src.hand_analyzer.service import hand_service


router = APIRouter(
    prefix="/hand",
    tags=["hand"],
    responses={404: {"description": "Not found"}},
)


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accepts a video file, samples frames every 5 seconds and returns
    left/right hand classifications as timestamped lists.
    """
    try:
        # delegate blocking work to threadpool
        result = await run_in_threadpool(hand_service.analyze_uploadfile_video, file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/realtime-frame")
async def realtime_frame(frame: UploadFile = File(...)) -> Dict[str, Any]:
    """Accepts a single image frame upload and returns per-hand features and classification.

    The client (realtime Next.js) should POST image/jpeg or image/png as form-data `frame`.
    """
    try:
        contents = await frame.read()
        # Convert bytes to OpenCV image (BGR)
        import numpy as np
        import cv2

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # run analysis in threadpool
        analysis = await run_in_threadpool(hand_service.analyze_frame, img)

        # Build response mapping hands to results
        resp = {"Left": None, "Right": None, "detected_hands": analysis}
        for item in analysis:
            hand = item.get("hand")
            resp[hand] = {"features": item.get("features"), "label": item.get("label")}

        return resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
