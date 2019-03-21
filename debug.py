from torchreid.data import Dataset
from torchreid.data.datasets import *

dataset1 = Market1501(root='data', combineall=False, transform='dummy', verbose=False)
dataset2 = Market1501(root='data', combineall=False, transform='dummy', verbose=False)
dataset3 = dataset1 + dataset2
print(type(dataset3))
print(dataset3)

print('** After addition **')
print(type(dataset1))
print(dataset1)

print(type(dataset2))
print(dataset2)

print(type(dataset3))
print(dataset3)