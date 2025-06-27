from mlsocket import MLSocket
import numpy as np

EHOST = "10.0.4.102"
EPORT = 65013

SHOST = "10.0.4.115"
SPORT = 65006



def conser(cs, con):
    s1=MLSocket()
    s1.connect((SHOST, SPORT))
    s1.send(np.array([1]))
    s1.send(con)
    cs.send(s1.recv(1024))
    print("File created remotely")
  
    
s=MLSocket()
s.bind((EHOST, EPORT))
s.listen()
print("Waiting ... ")
cs, addr=s.accept()
r=cs.recv(1024)

if r==1:
    m1=cs.recv(1024)
    print(m1)
    m2=cs.recv(1024)
    print(m1)
    res=m1 @ m2
    print("Result:", res)
    cs.send(res)
elif r==2:
    con=cs.recv(1024)
    conser(cs, con)
