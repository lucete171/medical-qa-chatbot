from fastapi import APIRouter
from app.schemas.request import InferenceRequest
from app.schemas.response import InferenceResponse
from app. services.inference_service import InferenceService

router = APIRouter()
service = InferenceService()

@router.post("/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest):
    answer = service.run(req.question)
    return InferenceResponse(answer=answer)