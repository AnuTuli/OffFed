#Server only code
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
import time

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 16384


# Load global data
global_data_path = 'off.csv'
global_data = pd.read_csv(global_data_path)
global_data=global_data.sample(frac=0.01)
X_global = global_data.iloc[:, 1:9].astype(np.float32).values
y_global = global_data.iloc[:, 9]

lb = LabelEncoder()
y_global = lb.fit_transform(y_global).astype(np.int32)
y_global = to_categorical(y_global)

sc = StandardScaler()
X_global = sc.fit_transform(X_global)
X_global = X_global.reshape((X_global.shape[0], 1, X_global.shape[1]))  # Reshape for LSTM

X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state=4)

model = Sequential()
model.add(LSTM(50, input_shape=(X_global.shape[1], X_global.shape[2])))
model.add(Dense(y_global.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=0)
    
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred)
print("Accuracy of the global model:", accuracy)
    

