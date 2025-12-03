from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any
import cv2
import numpy as np
import base64
import json

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
        result = await run_in_threadpool(hand_service.analyze_uploadfile_video, file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """WebSocket endpoint that accepts binary image frames (JPEG/PNG bytes)
    and returns JSON analysis results for each received frame.

    The client should send binary frames (Blob) whenever possible. This
    handler also includes a fallback for text messages that contain a
    base64-encoded image (data URL or JSON with `image` field).
    """
    await websocket.accept()
    try:
        while True:
            # receive() lets us handle both bytes and text payloads
            message = await websocket.receive()

            data_bytes = None
            if "bytes" in message and message["bytes"] is not None:
                data_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                text = message["text"]
                # data URL (data:image/jpeg;base64,....)
                if text.startswith("data:") and "," in text:
                    try:
                        data_bytes = base64.b64decode(text.split(",", 1)[1])
                    except Exception:
                        data_bytes = None
                else:
                    # Try JSON with an `image` key containing base64
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and parsed.get("image"):
                            data_bytes = base64.b64decode(parsed["image"])
                    except Exception:
                        data_bytes = None

            if not data_bytes:
                # nothing usable received; wait for next frame
                continue

            # decode image bytes into numpy array -> BGR image
            nparr = np.frombuffer(data_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # decoding failed
                continue

            # Run the (potentially blocking) analysis in a threadpool
            analysis = await run_in_threadpool(hand_service.analyze_frame, img)

            # Make the result JSON-serializable
            safe_analysis = []
            for item in analysis:
                safe_analysis.append({
                    "hand": item.get("hand"),
                    "features": [float(x) for x in item.get("features", [])],
                    "label": item.get("label"),
                })

            await websocket.send_json({"status": "OK", "analysis": safe_analysis})

    except WebSocketDisconnect:
        # client disconnected cleanly
        return
    except Exception:
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

