# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:11:05 2022

@author: USER
"""

import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.impute import KNNImputer
import datetime
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from modules import ExploratoryDataAnalysis,ModelCreation, ModelEvaluation

#paths
DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'new_customers.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
OHE_PATH = os.path.join(os.getcwd(), 'saved_model', 'ohe.pkl')
SAVE_RESULT = os.path.join(os.getcwd(), 'datasets', 'new_customers_result.csv')

# STEP 1: Model Loading

# model Loading
ohe_scaler = pickle.load(open(OHE_PATH,'rb'))
model = load_model(MODEL_PATH)
model.summary()

# STEP 2: Data Loading
df = pd.read_csv(DATA_PATH)

# STEP 3: Data Inspection
# a) to display the first 10 rows of data
print(df.head(10))

# b) to view the summary, non-null
print(df.info())
# Observation: Ever_Married, Graduated, Profession, Work_Experience, 
#               Family_Size, Var_1 contain Null Values

# STEP 4: Data Cleaning

# a) drop data with same ID
df = df.drop_duplicates(subset=['ID'])

# b) set ID column as index
df_new = df.copy()
df_new = df.set_index('ID')

# c) encode the categorical data using label encoder approach
numerical_data = df_new[['Age', 'Work_Experience', 'Family_Size']]
cat_data = df_new[['Gender','Ever_Married', 'Graduated', 'Profession', 
            'Spending_Score', 'Var_1']]

eda = ExploratoryDataAnalysis()
cat_data = eda.label_encoder(cat_data)

# d) fill nan using KNNImputer

X = pd.concat([cat_data, numerical_data], axis=1)

imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean') 
X = pd.DataFrame(imputer.fit_transform(X))

#%%
# STEP 4: Data Preprocessing

# a) scale the features using Min Max Scaler approach
X = eda.min_max_scaler(X)

#%%

# STEP 5: Deployment
predicted_y = np.empty([len(X), 4])

for index, test in enumerate(X):
    predicted_y[index,:] = model.predict(np.expand_dims(test, axis=0))

y2 = ohe_scaler.inverse_transform(predicted_y)
df['Segmentation'] = y2
df.to_csv(SAVE_RESULT, index=False)








