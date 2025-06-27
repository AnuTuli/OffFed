import pandas as pd
import numpy as np
import tqdm
import socket
import rsa
import time
import pickle

t1=time.time()

pubkey=pickle.load(open('PUBKEY.pem','rb'))

filename="Key1.key"

with open(filename, "wb") as f:
     while True:
           bytes_read = cs.recv(4096)
           if not bytes_read:
              break
           f.write(bytes_read)

newk=rsa.encrypt(bytes_read, pubkey)

with open('Key1.key', 'wb') as file:
     file.write(newk)

t2=time.time()

print(t2-t1)
