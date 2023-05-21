import dill

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.schema import Optional


app = FastAPI()

with open('../data/model.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: Optional[str]
    utm_medium: Optional[str]
    utm_campaign: Optional[str]
    utm_adcontent: Optional[str]
    utm_keyword: Optional[str]
    device_category: Optional[str]
    device_os: Optional[str]
    device_brand: Optional[str]
    device_model: Optional[str]
    device_screen_resolution: str
    device_browser: Optional[str]
    geo_country: Optional[str]
    geo_city: Optional[str]


class Prediction(BaseModel):
    target: int


@app.get('/status')
def status():
    return "I'm OK"


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model.predict(df)

    return {
        'target': y[0]
    }
