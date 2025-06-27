#FL-based Decision-making to Offload
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
import sys

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 16384

print_lock = threading.Lock()

# Load global data
global_data = pd.read_csv('offg.csv')
X_global = global_data.iloc[:, 1:9].astype(np.float32).values
y_global = global_data.iloc[:, 9]

lb = LabelEncoder()
y_global = lb.fit_transform(y_global).astype(np.int32)
y_global = to_categorical(y_global)

sc = StandardScaler()
X_global = sc.fit_transform(X_global)
X_global = X_global.reshape((X_global.shape[0], 1, X_global.shape[1]))  # Reshape for LSTM

X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state=4)

#model_params={'Sequential()', 'LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]))','Dense(y_train.shape[1], activation=softmax)','optimizer=adam, loss=categorical_crossentropy, metrics=[accuracy]'}
model_params=['start',50,'softmax','adam','categorical_crossentropy','end']
# global model training using LSTM

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
num_cl=4
num_round=10

s = MLSocket()
s.bind((SERVER_HOST, SERVER_PORT))


def printreport(model):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_classes, y_pred)
    con_mat = confusion_matrix(y_test_classes, y_pred)
    clreport=classification_report(y_test_classes, y_pred)
    print(con_mat)
    print(clreport)

def update_model():
    accuracy=0
    t1=time.time()
    model=init_model()
    all_addr=[]
    for i in range(num_cl):
        s.listen(2)
        print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
        print("Waiting for the client to connect... ")
        client_socket, addr=s.accept()
        all_addr.append(client_socket)
        print(f"[+] {addr} is connected.")
        acc=0
    for i in range(num_round):
        local_weights=[]
        i=1
        for conn in all_addr:
            conn.send(model)
            print("Model sent to Client"+str(i))
            i=i+1
        i=1
        for conn in all_addr:
            received = conn.recv(1024)
            print("Model Received from Client"+str(i)) 
            local_weights.append(received)
            i=i+1
        average_weights = [np.mean([model.get_weights()[k] for model in local_weights], axis=0) for k in range(len(local_weights[0].get_weights()))]
        model.set_weights(average_weights)
        tr1=time.time()
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred)
        tr2=time.time()
        print("Accuracy of the global model:", accuracy)
        if(abs(accuracy-acc)<=0.001):
            print("Model is updated finally")
            t2=time.time()
            print("Train time:",(t2-t1))
            print("RT:",(tr2-tr1))
            printreport(model)
            for conn in all_addr:
                conn.send(model)
                conn.close()
            sys.exit("Model is updated finally")
        elif accuracy>0.99:
            t2=time.time()
            print("Train time:",(t2-t1))
            print("RT:",(tr2-tr1))
            printreport(model)
            for conn in all_addr:
                conn.send(model)
                conn.close()
            modelfile='finalmodel.sav'
            pickle.dump(model4, open(modelfile, 'wb'))
            sys.exit("Model is updated finally")
    t2=time.time()
    print("Train time:",(t2-t1))
    print("RT:",(tr2-tr1))
    printreport(model)
    for conn in all_addr:
        conn.send(model)
        conn.close()
    modelfile='finalmodel.sav'
    pickle.dump(model4, open(modelfile, 'wb'))
    sys.exit("Model is updated finally")

    

update_model()


