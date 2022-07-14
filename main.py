from imp import reload
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import uvicorn
import datetime

class Request(BaseModel):
    array:list

app = FastAPI()

model = joblib.load("models/main_model.pkl")

def prepare(array:list):
    array[5] = 1 if array[5] == "normal" else 0
    array[11] = 1 if array[11] == "yes" else 0

    new_array = []

    new_array.append(array[0]) # age
    new_array.append(array[1]) # bp
    new_array.append(array[2]) # sg
    new_array.append(array[3]) # al
    new_array.append(array[4]) # su
    new_array.append(array[6]) # bgr
    new_array.append(array[7]) # bu
    new_array.append(array[8]) # sc
    new_array.append(array[9]) # hemo
    new_array.append(array[10]) # pcv

    new_array.append(array[1]/array[0]) # bp_per_age
    new_array.append(array[4]/array[6]) # bgr_per_su
    new_array.append(array[3]/array[7]) # bu_per_al

    new_array.append(array[5]) # pc
    new_array.append(array[11]) # htn

    return new_array

def log(err):
    time = int(datetime.datetime.now().timestamp())
    log_text = f"Timestampe: {time}"
    log_text += f"\nError: {err}\n"
    with open("log.txt", "a") as f:
        f.write(log_text)
        f.write("\n")

def predict(array:list):
    try:
        array = prepare(array)
        pred = model.predict_proba([array]).tolist()[0]
        proba = pred[0] if pred[0] > pred[1] else pred[1]
        pred = pred.index(proba)

        proba*=100
        proba = 99.0 if proba == 100 else round(proba, 1)

        return [True, pred, proba]
    except Exception as e:
        log(e)
        return [False]

@app.post("/predict/")
async def predict_method(req:Request):
    if req.array == None or len(req.array) < 1:
        return {"Error": True, "Message":"Invalid request"}
    
    prediction = predict(req.array)
    if not prediction[0]:
        return {"Error": True, "Message":"Error occured in prediction"}

    return {"Error": False, "Prediction": prediction[1], "Probability": prediction[2]}


# if __name__=="__main__":
#     uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)


# uvicorn main:app --reload