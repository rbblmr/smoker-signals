#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'
# host="smoker-serving-env.eba-pvppz392.ap-southeast-1.elasticbeanstalk.com"
# url= f"http://{host}/predict"

patient_id = 'xyz-123'
patient = {
    'age': 40,
    'waist(cm)': 84.0,
    'eyesight(left)': 1.2,
    'eyesight(right)': 1.2,
    'hearing(left)': 1,
    'hearing(right)': 1,
    'systolic': 130,
    'relaxation': 89,
    'fasting blood sugar': 107,
    'Cholesterol': 200,
    'triglyceride': 186,
    'HDL': 49,
    'LDL': 115,
    'hemoglobin': 14.2,
    'Urine protein': 1,
    'serum creatinine': 0.9,
    'AST': 19,
    'ALT': 25,
    'Gtp': 32,
    'dental caries': 1,
    'cholesterol_ratio': 4.08,
    'de_ritis_ratio': 0.76,
    'bmi_class': 'Overweight',
}


response = requests.post(url, json=patient).json()
print(response)

if response['smoker?']== True:
    print('Patient %s is a smoker.' % patient_id)
else:
    print('Patient %s is not a smoker.' % patient_id)