# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:35:34 2022

@author: USER
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from modules import ExploratoryDataAnalysis,ModelCreation, ModelEvaluation
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.preprocessing import OneHotEncoder

#%% Paths
DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'train.csv')
LOG_PATH = os.path.join(os.getcwd(),'log')
OHE_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model','ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
#%% EDA
# STEP 1: Data Loading
df = pd.read_csv(DATA_PATH)

# STEP 2: Data Inspection
# a) to display the first 10 rows of data
print(df.head(10))

# b) to view the summary, non-null
print(df.info())
# Observation: Ever_Married, Graduated, Profession, Work_Experience, 
#               Family_Size, Var_1 contain Null Values

#%%
# STEP 3: Data Cleaning

# a) drop data with same ID
df = df.drop_duplicates(subset=['ID'])

# b) set ID column as index
df = df.set_index('ID')

# c) encode the categorical data using label encoder approach
numerical_data = df[['Age', 'Work_Experience', 'Family_Size']]
cat_data = df[['Gender','Ever_Married', 'Graduated', 'Profession', 
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

# b) encode the target data using One Hot encoder approach
y = df['Segmentation']
enc = OneHotEncoder(sparse=False) 
y = enc.fit_transform(np.expand_dims(y,axis=-1))
pickle.dump(enc, open(OHE_SAVE_PATH, 'wb'))
# #%%
# # STEP 5: DL Model
# # a) split train & test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
#                                                     random_state=42)
# X_train = np.expand_dims(X_train,-1)
# X_test = np.expand_dims(X_test,-1)

# # b) create model
# mc = ModelCreation()
# model = mc.sequential(input_shape=9, output_shape=4, nb_nodes=256)

# plot_model(model)

# model.compile(optimizer = 'adam', 
#               loss = 'categorical_crossentropy', 
#               metrics = ['acc'])

# # c) Callbacks 
# log_files = os.path.join(LOG_PATH, 
#                          datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
# tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)

# # e) train the model
# hist = model.fit(X_train, y_train, epochs=100, 
#                     validation_data=(X_test,y_test), 
#                     callbacks=[tensorboard_callback,early_stopping_callback])


# #STEP 7: Model Evaluation

# predicted_y = np.empty([len(X_test), 4])

# for index, test in enumerate(X_test):
#     predicted_y[index,:] = model.predict(np.expand_dims(test, axis=0))

# # STEP 9: Model analysis
# y_pred = np.argmax(predicted_y, axis=1)
# y_true = np.argmax(y_test, axis=1)

# ModelEvaluation().report_metrics(y_true, y_pred)

# model.save(MODEL_SAVE_PATH)







