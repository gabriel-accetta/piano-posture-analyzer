from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(
    prefix="/posture",
    tags=["posture"],
    responses={404: {"description": "Not found"}},
)

@router.post("/analyze-video")
async def analyze_posture(request):
    return request
