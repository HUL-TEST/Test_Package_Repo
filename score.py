import json
import numpy as np
import os
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('stack_regressor')
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    preds = model.predict(data)
    return json.dumps(preds)