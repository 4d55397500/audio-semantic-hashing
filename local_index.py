# local_index.py
""""
    A local index over binary sequences using the Hamming distance
    function in the annoy library

    the library requires audio chunks (or whatever object being index)
    to be identified by an integer
"""
from annoy import AnnoyIndex

from constants import ENCODED_BITSEQ_LENGTH

# example use
# import random
#
#
# f = 40
# t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
# for i in range(1000):
#     v = [random.gauss(0, 1) for z in range(f)]
#     t.add_item(i, v)
#
# t.build(10)  # 10 trees
# t.save('test.ann')
#
# # ...
#
# u = AnnoyIndex(f, 'angular')
# u.load('test.ann')  # super fast, will just mmap the file
# print(u.get_nns_by_item(0, 1000))  # will find the 1000 nearest neighbors
#


def initialize_index():
    return AnnoyIndex(ENCODED_BITSEQ_LENGTH, 'hamming')


def build_index(index, n_trees):
    # no more items can be added once build is called
    index.build(n_trees)


def save_to_disk(index, filename):
    # no more items can be added once save is called
    index.save(filename)


def load_from_disk(filename):
    index = AnnoyIndex(ENCODED_BITSEQ_LENGTH, 'hamming')
    index.load(filename)
    return index


def add_to_index(index, int_id, bitseq):
    index.add_item(int_id, bitseq)


def query_by_id(index, int_id, n):
    # query n nearest neighbors passing an id
    pass


def query_by_vector(index, v, n):
    # query n nearest neighbors passing a vector
    pass

