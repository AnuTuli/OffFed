import socket
import tqdm
import os
import time
import pandas as pd
from cryptography.fernet import Fernet
import time


def sfile(filename):
    with open('Key1.key', 'rb') as filekey:
        key=filekey.read()

    fernet= Fernet(key)

    with open(filename, "rb") as f:
        org=f.read()

    encrypted=fernet.encrypt(org)

    filename="CDe1.csv"

    with open(filename, "wb") as ef:
        ef.write(encrypted)
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096
    s1 = socket.socket()
    host = "10.0.4.102"
    port = 5001
    print(f"[+] Connecting to {host}:{port}")
    s1.connect((host, port))
    print("[+] Connected to ", host)
    filesize = os.path.getsize(filename)
    s1.send(f"{filename}{SEPARATOR}{filesize}".encode())
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        while True:
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            s1.sendall(bytes_read)
        progress.update(len(bytes_read))
    s1.close()


