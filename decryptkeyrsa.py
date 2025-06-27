import time
from cryptography.fernet import Fernet
import rsa
import pickle

prikey=pickle.load(open('PRIKEY.pem','rb'))
t1=time.time()

num_cl=int(input('Enter number of clients:'))

for i in range(num_cl):
    with open('Enkey.key', 'rb') as filekey:
        key=filekey.read()
        
    newk=rsa.decrypt(key, prikey)
    filename="Key"+str(i+1)+".key"
    with open(filename, 'wb') as file:
        file.write(newk)
    
t2=time.time()

print(t2-t1)
