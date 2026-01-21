from app.models.predictor import Predictor

class InferenceService:
    def __init__(self):
        self.predictor = Predictor()

    def run(self, question:str) -> str:

        # 
        
        answer = self.predictor.predict(question)
        return answer