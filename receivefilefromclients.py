import socket
import tqdm
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd
import pickle
    
def recfile(n, j):
    SERVER_HOST = "10.0.4.102"
    SERVER_PORT = 5001
    BUFFER_SIZE = 4096
    SEPARATOR = "<SEPARATOR>"
    s1 = socket.socket()
    s1.bind((SERVER_HOST, SERVER_PORT))
    s1.listen(10)
    print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
    print("Waiting for the client to connect... ")
    client_socket, address = s1.accept()
    print(f"[+] {address} is connected.")
    filename="CDe.csv"
    received = client_socket.recv(1024).decode()
    with open(filename, "wb") as f:
        while True:
            bytes_read = client_socket.recv(1024)
            if not bytes_read:
                break
            f.write(bytes_read)
    client_socket.close()
    s1.close()
