!pip install kaggle --upgrade

import os
from getpass import getpass
from google.colab import userdata
os.environ["KAGGLE_USERNAME"]="name"
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
# Enter key with no " "

!kaggle competitions download -c playground-series-s3e24

!unzip -q "./playground-series-s3e24.zip" -d .

# general libraries
import time
import pickle
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# modelling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder, StandardScaler

# ensemble
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# hypertuning
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

#Metrics
from sklearn.metrics import f1_score, log_loss, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score, cross_validate

# interpretability
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

pd.set_option('display.max_columns', None)

##############################################################################
# Feature Engineering
df=pd.read_csv('train.csv')

# Cholesterol Ratio : Total Cholesterol / HDL - ration of total cholesterol to good cholesterol
# AST/ALT ratio or De Ritis ratio - liver function test

def preprocess(df):

  # Cholesterol Ratio (Higher -> Bad)
  df['cholesterol_ratio'] = round(df['Cholesterol']/df['HDL'], 2)

  #AST/ALT ratio or De Ritis ratio (Higher => Bad)
  df['de_ritis_ratio'] = round(df['AST']/df['ALT'], 2)

  # BMI
  def bmi(height,weight):
    bmi=weight/(height/100)**2

    return round(bmi,2)

  df['bmi']=bmi(df['height(cm)'], df['weight(kg)'])
  df['bmi_class']=pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf], labels=["Underweight", "Normal", "Overweight", "Obese 1", "Obese 2", "Obese 3"])

  return df

df=preprocess(df)

###############################################################################
# Set the columns to respective variables
X=df.loc[:, df.columns != 'smoking']
y=df['smoking']
categorical = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries','bmi_class']
numerical = df.columns[~df.columns.isin(categorical+['id','smoking', 'weight(kg)','height(cm)','bmi'])]
###############################################################################
# Pipeline for transformation
# Function for yeojohnson
from scipy import stats
def stats_boxcox(df,numerical):
  for col in numerical:
    df[col]=stats.boxcox(df[col])[0]

# Pipelines for numerical and categorical
num_pipeline = make_pipeline(
    FunctionTransformer(func=stats_boxcox(numerical=numerical, df=X), feature_names_out="one-to-one"), #np_log_1(numerical=numerical, df=X)
    StandardScaler())

cat_pipeline = make_pipeline(
    OneHotEncoder(sparse=False, handle_unknown="ignore"), verbose=True)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, numerical),
    ("cat", cat_pipeline, categorical),
   ])

preprocessing_pipeline = make_pipeline(preprocessing)
###############################################################################
# Train-Test Function for Unprocessed data

def train_val_test(df, val_size=0.2, test_size=0.1):

  # Choosing Initial Features
  val_pct = val_size/(1-test_size)

  X=df.loc[:, df.columns != 'smoking']
  y=df['smoking']

  # 70-20-10
  X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y) # 90-10
  X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=val_pct, random_state=42, stratify=y_full_train) # 20 is __% of 90

  dict_df = {'X_full_train':X_full_train,'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_full_train':y_full_train, 'y_train': y_train, 'y_val':y_val, 'y_test':y_test}

  return dict_df
###############################################################################
# Creating the train - val - test for unprocessed data
dict_df=train_val_test(df)

X_full_train=dict_df['X_full_train'].reset_index(drop=True)

X_train=dict_df['X_train'].reset_index(drop=True)

X_val=dict_df['X_val'].reset_index(drop=True)

X_test=dict_df['X_test'].reset_index(drop=True)

y_full_train=dict_df['y_full_train'].reset_index(drop=True)

y_train=dict_df['y_train'].reset_index(drop=True)

y_val=dict_df['y_val'].reset_index(drop=True)

y_test=dict_df['y_test'].reset_index(drop=True)

###############################################################################

# Training the best model
def train(X_train, y_train, model):
    skf = StratifiedKFold(n_splits=3)

    model=make_pipeline(preprocessing, model) # Change the model here
    scoring = {'acc':'accuracy','prec': 'precision', 'f1':'f1', 'roc':'roc_auc'}

    scores=cross_validate(model, X_full_train, y_full_train, cv=skf, scoring=scoring, return_train_score=True)
    
    scores_average={}
    for key in scores.keys():
        scores_average[key]=np.mean(scores[key])
    
    print(scores_average)
    
    model.fit(X_full_train, y_full_train)
    
    return model

def predict(X_test, model):
    df_test=preprocess(X_test)
    y_pred=model.predict(df_test)
    
    return y_pred

# Final model
print("Training the final model")
# Best Params
params={
    'booster': 'gbtree',
    'learning_rate': 0.013310002716978346,
    'max_depth': 50,
    'min_child_weight': 75.38444619706124,
    'n_estimators': 483,
    'objective': 'binary:logistic',
    'random_state': 42,

}
classifier=xgb.XGBClassifier(**params)

model=train(X_train=X_full_train, y_train=y_full_train, model=classifier)

y_pred = predict(X_test, model)

auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# Saving the model
print("Saving the model to current directory")
pickle.dump(model, open('./xgb_smoking_model.pkl', 'wb'))
print("Saved!")