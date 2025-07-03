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
from sklearn.neural_network import MLPClassifier

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 16384

print_lock = threading.Lock()

# Load global data
def preprocess_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Encode labels
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    # Standardize features
    sc = StandardScaler()
    X = sc.fit_transform(X)

    return X, y

data = pd.read_csv('taskdata.csv')
peer_data = preprocess_data(data)


X_local = peer_data[0]
y_local = peer_data[1]
X_test = [None] * 1
y_test = [None] * 1


X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.2, random_state=41)

SERVER_HOST = "10.0.4.102"
SERVER_PORT = 65432
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

local_weights=[]
all_addr=[]

def init_model():
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100)
    return model

t=0
num_cl=4
num_round=10

s = MLSocket()
s.bind((SERVER_HOST, SERVER_PORT))

def update_model():
    accuracy=0
    t1=time.time()
    model=init_model()
    model.fit(X_train, y_train)
    all_addr=[]
    for i in range(num_cl):
        s.listen(2)
        print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
        print("Waiting for the client to connect... ")
        client_socket, addr=s.accept()
        all_addr.append(client_socket)
        print(f"[+] {addr} is connected.")
    for r in range(num_round):
        local_models=[]
        i=1
        for conn in all_addr:
            conn.send(model)
            print("Model sent to Client"+str(i))
            i=i+1
        i=1
        for conn in all_addr:
            received = conn.recv(1024)
            print("Model Received from Client"+str(i)) 
            local_models.append(received)
            i=i+1
        average_weights = [np.mean([model.coefs_[k] for model in local_models], axis=0) for k in range(len(local_models[0].coefs_))]
        model.coefs_=average_weights
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_test_classes = y_test
        accuracy = accuracy_score(y_test_classes, y_pred)
        print("Accuracy of the peer model:", accuracy)
    print("Model is updated finally")
    for conn in all_addr:
        conn.send(model)
        conn.close()
    

update_model()
