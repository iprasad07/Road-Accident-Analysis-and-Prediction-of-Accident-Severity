import joblib
import numpy as np
from config import *
import pandas as pd

def get_prediction(model, data):
    """
    Predict the class of a given data point
    """
    prediction = model.predict(data)[0]
    print("=============================== \n",prediction,type(prediction),prediction.argmax())
    return Accident_severity_dict[prediction.argmax()]







