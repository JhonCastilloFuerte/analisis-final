from fastapi import FastAPI
from joblib import load

# Por error se instala con pip install -U scikit-learn
import sklearn

app = FastAPI()

# Se prueba con este array 
#[
#  7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47
#]

# instentar desplegar en Vercel, mlflow, mlops

@app.get("/")
async def root():
    return {"message": "Running api test"}

@app.post("/predict")
async def predict(data: list):
    model = load('modelored_final.joblib')
    result = model.predict([data])

    print(f"El resultado es {result} {type(str(result))}")
    return {"result": str(result[0])}
