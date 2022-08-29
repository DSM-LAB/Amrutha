import numpy as np
arr1=np.array([[1,2,3],[2,3,4],[6,7,8]])
arr2=np.array([[5,6,7],[5,6,7],[9,8,6]])
print("original 3-D matrix")
print(arr1)
print(arr2)
print("outer product of the 2 matrix")
r=np.outer(arr1,arr2)
print(r)
              
