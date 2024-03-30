from flask import Flask,render_template,request
import pandas as pd 
from src.cnnClassifier.pipeline.prediction import PredictionPipeline
from io import BytesIO

app = Flask(__name__)
pred = PredictionPipeline()

@app.route('/')
def start():
    return render_template('index.html')

@app.route('/predict', methods = ['POST','GET'] )
def predict():
    if request.method == "GET":
        return render_template('start.html')
    
    else:
        img = request.files['image']
        img_bytes = BytesIO(img.read())
        result = pred.prediction(img_bytes,(224,224))
        if result[0]== 0 :
            x = 'Normal'
        else:
            x = 'Tumor'
        return render_template('start.html',result = x)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

