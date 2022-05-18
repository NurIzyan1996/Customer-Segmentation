# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:09:21 2022

@author: Nur Izyan Binti Kamarudin
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score


class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def label_encoder(self, input_data):  
        label_encoder = LabelEncoder() 
        return input_data.apply(lambda series: pd.Series(
            label_encoder.fit_transform(series[series.notnull()]), 
            index=series[series.notnull()].index))

    
    def one_hot_encoder(self, input_data):  
        enc = OneHotEncoder(sparse=False) 
        return enc.fit_transform(np.expand_dims(input_data,axis=-1))
       
    
    def min_max_scaler(self, input_data):  
        mms = MinMaxScaler() 
        return mms.fit_transform(input_data)
        
    

class ModelCreation():
    def __init__(self):
        pass
    
    def sequential(self, input_shape, output_shape, nb_nodes):
        model = Sequential()
        model.add(Input(shape=(input_shape), name='input_layer'))
        model.add(Flatten()) # to flatten the data
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_3'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_4'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation='softmax', name='output_layer'))
        model.summary()
        return model

class ModelEvaluation():
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))





