import tqdm
import socket
import time
from cryptography.fernet import Fernet
import rsa

SERVER_HOST = "10.0.4.102"
SERVER_PORT = 64345

s1 = socket.socket()
s1.bind((SERVER_HOST, SERVER_PORT))
s1.listen(5)
print("Waiting for the client to connect... ")

x=0
t=0

num_cl=int(input('Enter number of clients:'))

for i in range(num_cl):
    cs, addr1=s1.accept()
    x=x+1
    filename = "Key"+str(x)+".key"
    t1=time.time()
    with open(filename, "wb") as f:
        while True:
            bytes_read = cs.recv(4096)
            if not bytes_read:
                break
            f.write(bytes_read)
    t2=time.time()
    print("Key Received")
    t=t+(t2-t1)

print(t)
