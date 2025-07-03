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
import threading
from mlsocket import MLSocket
import time
from sklearn.neural_network import MLPClassifier

HOST = "10.0.4.102"
PORT = 65432
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

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

data = pd.read_csv('tasklocaldata.csv')
peer_data = preprocess_data(data)


X_local = peer_data[0]
y_local = peer_data[1]
X_test = [None] * 1
y_test = [None] * 1

X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.2, random_state=41)

t=0

with MLSocket() as s:
    s.connect((HOST, PORT)) # Connect to the port and host
    t1=time.time()
    while True:
        model=s.recv(2024)
        print("Received model")
        model.fit(X_train, y_train)
        s.send(model)
        y_pred = model.predict(X_test)
        y_test_classes = y_test
        accuracy = accuracy_score(y_test_classes, y_pred)
        print("Accuracy of the local model:", accuracy)
        t2=time.time()
        t=t+(t2-t1)
        print(t)
