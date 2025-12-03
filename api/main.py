from fastapi import FastAPI
from .routers import hand, posture

app = FastAPI(
    title="Piano Posture Analyzer API",
    description="API for analyzing piano playing posture and hand technique.",
    version="0.1.0",
)

app.include_router(hand.router)
app.include_router(posture.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Piano Posture Analyzer API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
