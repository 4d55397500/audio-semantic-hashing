# local_index.py
""""
    A local index over binary sequences using the Hamming distance
    function in the annoy library

    the library requires audio chunks (or whatever object being index)
    to be identified by an integer
"""

# example use
from annoy import AnnoyIndex
import random

f = 40
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10)  # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 1000))  # will find the 1000 nearest neighbors




