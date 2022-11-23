import logging
import azure.functions as func
import json
from fastbook import load_learner
from fastbook import PILImage

def predict_instrument(file):
    learn_inf = load_learner('instrument_classifier_export.pkl')
    img = PILImage.create(file)
    pred,pred_idx,probs = learn_inf.predict(img)
    return { 'prediction': pred, 'probability': float(probs[pred_idx]) }

def main(req: func.HttpRequest) -> func.HttpResponse:
    # get file from req
    file = req.files['file']
    # get prediction
    response = predict_instrument(file)
    return func.HttpResponse(json.dumps(response), mimetype='application/json')

        