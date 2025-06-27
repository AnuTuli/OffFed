import numpy as np
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
import pickle

EHOST = "10.0.4.102"
EPORT = 65013

def calculator():
    c=int(input('Enter choice: 1. Add 2. Multiply 3. Subract 4. Divide'))
    a=input('Number1')
    b=input('Number2')
    if c==1:
        print(float(a)+float(b))
    elif c==2:
        print(float(a)*float(b))
    elif c==3:
        print(float(a)-float(b))
    elif c==4:
        print(float(a)/float(b))
    else:
        print("Invalid choice")

def predict(model):
    test=pd.read_csv('Test.csv')
    X_test = test.iloc[:,1:9].astype(np.float32).values
    lb = LabelEncoder()
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # Reshape for LSTM
    result = np.argmax(model.predict(X_test), axis=1)
    return result

def execute():
    print('Enter matrix1:')
    R1 = int(input("Enter the number of rows:"))
    C1 = int(input("Enter the number of columns:"))
    print("Enter the entries in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    m1 = np.array(entries).reshape(R1, C1)
    print('Enter matrix2:')
    R2 = int(input("Enter the number of rows:"))
    C2 = int(input("Enter the number of columns:"))
    print("Enter the entries in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    m2 = np.array(entries).reshape(R2, C2)
    res=m1 @ m2
    print("Result:", res)
    

def offload():
    s=MLSocket()
    s.connect((EHOST, EPORT))
    print('Enter matrix1:')
    R1 = int(input("Enter the number of rows:"))
    C1 = int(input("Enter the number of columns:"))
    print("Enter the entries in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    m1 = np.array(entries).reshape(R1, C1)
    print('Enter matrix2:')
    R2 = int(input("Enter the number of rows:"))
    C2 = int(input("Enter the number of columns:"))
    print("Enter the entries in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    m2 = np.array(entries).reshape(R2, C2)
    s.send(np.array([1]))
    s.send(m1)
    s.send(m2)
    res=s.recv(1024)
    print("Result:", res)


def filecr():
    content=input("Enter your content:")
    try:
        with open('file2.txt', 'w') as gfg:
            gfg.write(content)
    except Exception as e:
        print("There is a Problem", str(e)) 
    

def offfile():
    s=MLSocket()
    s.connect((EHOST, EPORT))
    s.send(np.array([2]))
    L=input("Enter your content:")
    print("Ok")
    s.send(np.array(L))
    print(s.recv(2024))
    

def choice():
    print("Enter choice: 1. Arithmatic calculation 2. Matrix multiplication 3. File creation")
    ch=int(input( ))
    if ch==1:
        calculator()
    elif ch==2:
        finalmodel=pickle.load(open('finalmodel.sav','rb'))
        result=predict(finalmodel)
        if result==1:
            execute()
        else:
            offload()
    else:
        print("1. Create local file 2. Create file online")
        ch1=int(input( ))
        if ch1==1:
            filecr()
        else:
            offfile()


choice()

        
