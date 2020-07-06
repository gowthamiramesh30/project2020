#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  27 22:36:20 2020

@author: rrc
"""

from flask import Flask,abort,jsonify,request
import pandas as pd 
import numpy as np
import pickle

my_regression=pickle.load(open('final_model_27Jun2020.pkl','rb'))

app = Flask(__name__)

#decorator (@app.route('/')) to specify the URL that should trigger the execution of the home function

@app.route('/api', methods=['POST'])
def make_predict():
    data=request.get_json(force='True')
    predict_request=[data['Month'],data['Day'],data['Hour'],data['Minute'],data['ProTypeAir'],data['ProTypeHot'],data['IteTypeDomes']]
    predict_request=np.array(predict_request)
    y_hat=my_regression.predict(predict_request)
    output=[y_hat[7]]
    return jsonify(request=output)

if __name__ == '__main__':
	app.run(port=9000,debug=True)