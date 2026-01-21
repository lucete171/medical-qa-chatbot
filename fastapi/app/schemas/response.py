from pydantic import BaseModel

class InferenceResponse(BaseModel):
    answer : str