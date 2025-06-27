import pandas as pd
import numpy as np
import tqdm
import socket
from cryptography.fernet import Fernet
import time

HOST = "10.0.4.102"
PORT = 64345

s1=socket.socket()
s1.connect((HOST, PORT))

filename="Key5.key"

t1=time.time()
#progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
with open(filename, "rb") as f:
    while True:
        bytes_read = f.read(4096)
        if not bytes_read:
            break
        s1.sendall(bytes_read)
        #progress.update(len(bytes_read))
s1.close()

t2=time.time()

print(t2-t1)
