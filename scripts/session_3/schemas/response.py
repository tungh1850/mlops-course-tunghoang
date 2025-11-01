from pydantic import BaseModel

class PredictResponse(BaseModel):
    result: float

class IrisPredictResponse(BaseModel):
    result: float