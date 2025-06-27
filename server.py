from mlsocket import MLSocket
import numpy as np

SHOST = "10.0.4.115"
SPORT = 65006


def filecr(cs):
    file1=open('file.txt','w')
    r=cs.recv(1024)
    print(r)
    file1.write(np.array_str(r))
    file1.close()
    cs.send(np.array(["Saved successfully"]))
    print("File created")
    
s=MLSocket()
s.bind((SHOST, SPORT))

s.listen(10)
print("Waiting ... ")

cs, addr=s.accept()
        
r=cs.recv(1024)

if r==1:
    filecr(cs)
    
    
