from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.inference import router as Inference_router
from app.api.ws import router as ws_router

app=FastAPI()

@app.get("/")
def road_root():
    return{"massage":"안녕하는 패스트에이피아이"}

app.include_router(health_router)  #app/api/health
app.include_router(Inference_router)
app.include_router(ws_router)
