from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        logging.info('Collecting the User Data')
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        logging.info('Collected the User Data ')

        pred_df = data.get_data_as_data_frame()
        logging.info('Data Converted into DataFrame')
        logging.info("Initialising Prediction Pipeline")
        
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        logging.info('Prediction Done')
        return render_template('home.html',results=result[0])


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)