from cryptography.fernet import Fernet
import time

t1=time.time()
key = Fernet.generate_key()

# string the key in a file
with open('Key5.key', 'wb') as filekey:
    filekey.write(key)

t2=time.time()

print(t2-t1)
