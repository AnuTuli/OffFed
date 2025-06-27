import time
from cryptography.fernet import Fernet
import rsa
import pickle

prikey=pickle.load(open('PRIKEY.pem','rb'))
t1=time.time()

num_cl=int(input('Enter number of clients:'))

x=0

for i in range(num_cl):
    x=x+1
    filename="Enkey"+str(x)+".key"
    with open('Enkey.key', 'rb') as filekey:
        key=filekey.read()
        
    newk=rsa.decrypt(key, prikey)
    filename="Key"+str(x)+".key"
    with open(filename, 'wb') as file:
        file.write(newk)
    
t2=time.time()

print(t2-t1)
