from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import datetime
import os

from fastapi.middleware.cors import CORSMiddleware


class Request(BaseModel):
    array: list


class DataResponse(BaseModel):
    error: bool
    reason: str
    pred: int
    proba: float


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("./models/main_model.pkl")


def prepare(array: list) -> list:
    array[5] = 1 if array[5] == "normal" else 0
    array[11] = 1 if array[11] == "yes" else 0
    print(array)
    new_array = [array[0], array[1], array[2], array[3], array[4], array[6], array[7], array[8], array[9], array[10],
                 array[1] / array[0], array[4] / array[6], array[3] / array[7], array[5], array[11]]

    return new_array


def log(err: str):
    time = int(datetime.datetime.now().timestamp())
    log_text = f"Timestampe: {time}"
    log_text += f"\nError: {err}\n"
    with open("log.txt", "a") as f:
        f.write(log_text)
        f.write("\n")

    return


def predict(array: list):
    try:
        array = prepare(array)
        pred = model.predict_proba([array]).tolist()[0]
        proba = pred[0] if pred[0] > pred[1] else pred[1]
        pred = pred.index(proba)

        proba *= 100
        proba = 99.0 if proba == 100 else round(proba, 1)

        return [True, pred, proba]
    except Exception as e:
        log(str(e))
        return [False, 0, 0]


@app.post("/predict", response_model=DataResponse)
async def predict_method(req: Request):
    if req.array is None or len(req.array) < 1:
        return DataResponse(error=True, pred=0, proba=0, reason="Invalid request")
    prediction = predict(req.array)
    if not prediction[0]:
        return DataResponse(error=True, pred=prediction[1], proba=prediction[2], reason="Failure to get prediction")

    return DataResponse(error=False, pred=prediction[1], proba=prediction[2], reason="Success")

# run the fastapi
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        debug=True,
        reload=True,
    )
