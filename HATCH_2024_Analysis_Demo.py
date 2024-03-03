import numpy as np
import joblib
import pandas as pd
import sklearn
from sklearn import svm, preprocessing
from sklearn.utils import shuffle


classs = []
stage = []
origin = []
returns = []

clf = joblib.load('tumor_class_mapping.pkl')

dp =pd.read_csv("data.csv", index_col=0)#test



var = 'cgc_sample_sample_type'
tumor_stage_mapping = {0: 'Primary Tumor', 1: 'Solid Tissue Normal', 2: 'Recurrent Tumor'}
if dp[var].dtype != 'int':
    if var in tumor_stage_mapping.values():
        dp[var] = dp[var].map({v: k for k, v in tumor_stage_mapping.items()})
for column in dp.columns:
    if dp[column].dtype != 'int':
        dp[column] = dp[column].astype("category").cat.codes
        



dp = sklearn.utils.shuffle(dp)
dp = dp.dropna()
Xp = dp.drop(var, axis=1).values
Xp = preprocessing.scale(Xp)
Yp = dp[var].values

x_test = Xp#[:-test_size]
y_test = Yp#[:-test_size:]


for X,y in zip(x_test, y_test):
    prediction = clf.predict([X])[0]
    rounded = int(np.round(prediction))
    og = tumor_stage_mapping.get(rounded, "Unknown")
    ogact = tumor_stage_mapping.get(y, "Unknown")
    #print(original_value)
    #print(f"Model: {prediction}, Actual:{y} ")
    #print(f"Original Value: {og}")
    classs.append(og)
    
clf = joblib.load('tumor_stage_mapping.pkl')

dp =pd.read_csv("data.csv", index_col=0)#test



var = 'gdc_cases.diagnoses.tumor_stage'
tumor_stage_mapping = {0: 'stage iia', 1: 'stage ib', 2: 'stage ia', 3: 'stage iv', 4: 'stage iiia', 5: 'not reported', 6: 'stage iib', 7: 'stage iiib', 8: 'stage i', 9: 'stage ii'}
if dp[var].dtype != 'int':
    if var in tumor_stage_mapping.values():
        dp[var] = dp[var].map({v: k for k, v in tumor_stage_mapping.items()})
for column in dp.columns:
    if dp[column].dtype != 'int':
        dp[column] = dp[column].astype("category").cat.codes
        



dp = sklearn.utils.shuffle(dp)
dp = dp.dropna()
Xp = dp.drop(var, axis=1).values
Xp = preprocessing.scale(Xp)
Yp = dp[var].values

x_test = Xp#[:-test_size]
y_test = Yp#[:-test_size:]


for X,y in zip(x_test, y_test):
    prediction = clf.predict([X])[0]
    rounded = int(np.round(prediction))
    og = tumor_stage_mapping.get(rounded, "Unknown")
    ogact = tumor_stage_mapping.get(y, "Unknown")
    #print(original_value)
    #print(f"Model: {prediction}, Actual:{y} ")
    #print(f"Original Value: {og}")
    stage.append(og)





clf = joblib.load('tumororigin.pkl')

dp =pd.read_csv("data.csv", index_col=0)#test



var = 'cgc_follow_up_new_tumor_event_after_initial_treatment'
tumor_stage_mapping = {0: 'c34.1', 1: 'c34.3', 2: 'c34.9', 3: 'c34.2', 4: 'c34.8', 5: 'c34.0'}
if dp[var].dtype != 'int':
    if var in tumor_stage_mapping.values():
        dp[var] = dp[var].map({v: k for k, v in tumor_stage_mapping.items()})
for column in dp.columns:
    if dp[column].dtype != 'int':
        dp[column] = dp[column].astype("category").cat.codes
        



dp = sklearn.utils.shuffle(dp)
dp = dp.dropna()
Xp = dp.drop(var, axis=1).values
Xp = preprocessing.scale(Xp)
Yp = dp[var].values

x_test = Xp#[:-test_size]
y_test = Yp#[:-test_size:]


for X,y in zip(x_test, y_test):
    prediction = clf.predict([X])[0]
    rounded = int(np.round(prediction))
    og = tumor_stage_mapping.get(rounded, "Unknown")
    ogact = tumor_stage_mapping.get(y, "Unknown")
    #print(original_value)
    #print(f"Model: {prediction}, Actual:{y} ")
    #print(f"Original Value: {og}")
    origin.append(og)


clf = joblib.load('tumorreturn.pkl')

dp =pd.read_csv("data.csv", index_col=0)#test



var = 'cgc_follow_up_new_tumor_event_after_initial_treatment'
tumor_stage_mapping = {0: 'NO', 1: 'YES', 2: 'nan'}
if dp[var].dtype != 'int':
    if var in tumor_stage_mapping.values():
        dp[var] = dp[var].map({v: k for k, v in tumor_stage_mapping.items()})
for column in dp.columns:
    if dp[column].dtype != 'int':
        dp[column] = dp[column].astype("category").cat.codes
        



dp = sklearn.utils.shuffle(dp)
dp = dp.dropna()
Xp = dp.drop(var, axis=1).values
Xp = preprocessing.scale(Xp)
Yp = dp[var].values

x_test = Xp#[:-test_size]
y_test = Yp#[:-test_size:]


for X,y in zip(x_test, y_test):
    prediction = clf.predict([X])[0]
    rounded = int(np.round(prediction))
    og = tumor_stage_mapping.get(rounded, "Unknown")
    ogact = tumor_stage_mapping.get(y, "Unknown")
    #print(original_value)
    #print(f"Model: {prediction}, Actual:{y} ")
    #print(f"Original Value: {og}")
    returns.append(og)
x=0
for id in classs:
    
    print(f"Patient {x} most likely has a {classs[x]}, {stage[x]} cancer, originating from {origin[x]}, with a liklihood of return: {returns[x]} ")
    x +=1