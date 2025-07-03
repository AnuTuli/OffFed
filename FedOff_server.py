#Server only code
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import socket
import tqdm
import json
import os
from _thread import *
import threading
from mlsocket import MLSocket
import time
import matplotlib.pyplot as plt
import pickle

# Load global data
global_data_path = 'Act.csv'
global_data = pd.read_csv(global_data_path)
X_global = global_data.iloc[:, :-1].astype(np.float32).values
y_global = global_data.iloc[:, -1]

lb = LabelEncoder()
y_global = lb.fit_transform(y_global).astype(np.int32)
y_global = to_categorical(y_global)

sc = StandardScaler()
X_global = sc.fit_transform(X_global)
X_global = X_global.reshape((X_global.shape[0], 1, X_global.shape[1]))  # Reshape for LSTM

X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state=4)

SERVER_HOST = "10.0.4.102"
SERVER_PORT = 65432
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

local_weights=[]
all_addr=[]

def init_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_global.shape[1], X_global.shape[2])))
    model.add(Dense(y_global.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

t=0
num_cl=5
num_round=3


def update_model():
    local_weights=[]
    accuracy=0
    t1=time.time()
    model=init_model()
    model1=model

    for i in range(num_cl):
        local_weights.append(model1)  #initialize model for offloaded data

    for r in range(num_round):
        for k in range(num_cl):
            s.listen(2)
            client_socket, addr=s.accept()
            all_addr.append(client_socket)
        for k in range(num_cl):
            model1=local_weights[k]
            cd=pd.read_csv("CD"+str(k+1)+".csv")   #retrive received client data
            X_cd = cd.iloc[:, :-1].values
            y_cd = cd.iloc[:, -1]
            lb = LabelEncoder()
            y_cd = lb.fit_transform(y_cd)
            y_cd = to_categorical(y_cd)
            sc = StandardScaler()
            X_cd = sc.fit_transform(X_cd)
            X_cd = X_cd.reshape((X_cd.shape[0], 1, X_cd.shape[1]))  # Reshape
            model1.fit(X_cd, y_cd, epochs=100, batch_size=100, verbose=0)     # fit data
            local_weights[k]=model1
        for conn in all_addr:                        
            received = conn.recv(1024)
            print("Model Received from Client"+str(i)) 
            local_weights.append(received)                   #receive model update for local data from clients

        average_weights = [np.mean([model.get_weights()[k] for model in local_weights], axis=0) for k in range(len(local_weights[0].get_weights()))]
        model.set_weights(average_weights)

        for conn in all_addr:
            conn.send(model)
            print("Model sent to Client"+str(i))  #send model to client

    t2=time.time()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_classes, y_pred)
    print("Accuracy of the global model:", accuracy)
    CF=classification_report(y_test_classes, y_pred)
    print("Model is updated finally")
    print(t2-t1)
    print(CF)
    modelfile='Activitymodel.sav'
    pickle.dump(model, open(modelfile, 'wb'))
    

update_model()


