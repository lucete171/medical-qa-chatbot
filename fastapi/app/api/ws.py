from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.inference_service import InferenceService
import json

router = APIRouter()
service = InferenceService()

@router.websocket("/ws/inference")
async def websocket_inference(ws: WebSocket):
    await ws.accept() # 수락 
    print("[WS] accepted")

    try:
        while True:
            raw = await ws.receive_text() # 클라이언트 메세지 대기 
            print("[WS] raw 수신:", raw)
            payload = json.loads(raw)

            question = payload["content"]
            answer = service.run(question)

            await ws.send_text(json.dumps({ #응답
                "type": "answer",
                "content": answer
            }))
            print("[WS] sent")

    except WebSocketDisconnect:
        #연결 종료 시 정리 작업
        print("WebSocket disconnected")