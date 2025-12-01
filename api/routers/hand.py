from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(
    prefix="/hand",
    tags=["hand"],
    responses={404: {"description": "Not found"}},
)

@router.post("/analyze-video")
async def analyze_hand(request):
    return request
