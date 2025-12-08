from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import hand, body

app = FastAPI(
    title="Piano Posture Analyzer API",
    description="API for analyzing piano playing posture and hand technique.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hand.router)
app.include_router(body.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Piano Posture Analyzer API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
