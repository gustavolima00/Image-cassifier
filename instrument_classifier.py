from fastbook import load_learner
from fastbook import PILImage

def predict_instrument(file):
    learn_inf = load_learner('instrument_classifier_export.pkl')
    img = PILImage.create(file)
    pred,pred_idx,probs = learn_inf.predict(img)
    return { 'prediction': pred, 'probability': float(probs[pred_idx]) }

with open('images/guitar.jpg', "rb") as f:
    file = f.read()
    print(predict_instrument(file))