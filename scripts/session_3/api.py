from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from enum import Enum
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import APIRouter


from scripts.session_3.router import housing, iris

app = FastAPI()
app.include_router(housing.housing_router)
app.include_router(iris.iris_router)

class Method(str, Enum):
    add = "add"
    subtract = "subtract"
    multiply = "multiply"
    divide = "divide"
 
class CalculateResquest(BaseModel):
    method: Method
    num1:float
    num2:float

class CalculateResponse(BaseModel):
    result: float



@app.get("/")
def root():
    return{"message":"Hello, FastAPI!"}

@app.get("/health")
def health(dump_input:int):
    if dump_input > 10:
        return  {"message":f"this is large number. Your input was {dump_input}"}
    else: 
        return  {"message":f"this is small number. Your input was {dump_input}"}



if __name__ == "__main__":
    uvicorn.run("api:app",host="0.0.0.0",port=3000, reload=True)