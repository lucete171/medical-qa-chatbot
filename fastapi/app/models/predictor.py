from app.models.loader import load_model

class Predictor:
    def __init__(self):
        self.model = load_model()

    def predict(self, question: str) -> str:
        return self.model.query(question)
