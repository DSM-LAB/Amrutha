from numpy import array
from numpy import diag
from numpy import dot
from scipy.linalg import svd
#define a mtrix
A=array([[1,2,3],[4,5,6],[7,8,9]])
print(A)
U,s,VT=svd(A)
print(U)
print(s)
print(VT)
#create n x n Sigma matrix
Sigma=diag(s)
B=U.dot(Sigma.dot(VT))
print(B)
