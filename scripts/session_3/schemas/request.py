from pydantic import BaseModel

class PredictRequest(BaseModel):
     avg_area_income: float
     avg_area_house_age: float
     avg_area_number_of_rooms: float
     avg_area_number_of_bedrooms: float
     area_population: float

class IrisPredictRequest(BaseModel):
     sepal_length: float
     sepal_width: float
     petal_length: float
     petal_width: float
