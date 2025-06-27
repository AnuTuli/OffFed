import time
from cryptography.fernet import Fernet
import rsa
import pickle

prikey=pickle.load(open('PRIKEY.pem','rb'))
t1=time.time()
filename="Key1.key"
with open('key1.key', 'rb') as filekey:
    key=filekey.read()
        
newk=rsa.decrypt(key, prikey)
with open('Key1.key', 'wb') as file:
    file.write(newk)
    
t2=time.time()

print(t2-t1)
