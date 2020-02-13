# local_index.py
""""
    A local index over binary sequences using the Hamming distance
    function in the annoy library

    the library requires audio chunks (or whatever object being indexed)
    to be identified by an integer
"""
from annoy import AnnoyIndex
import os
import pickle
import torch

from audio_ops import chunks_to_torch_tensor
from constants import ENCODED_BITSEQ_LENGTH, \
    LOCAL_CHUNK_FILEPATHS, MODEL_SAVE_PATH, \
    INDEX_DIR, INDEX_SAVE_PATH, ID_MAPPING_SAVE_PATH,\
    INDEX_NUM_TREES



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


def create_index():

    index = initialize_index()
    x = chunks_to_torch_tensor(LOCAL_CHUNK_FILEPATHS)
    id_mapping = int_id_mapping()
    with open(ID_MAPPING_SAVE_PATH, 'wb') as handle:
        pickle.dump(dict(id_mapping), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    binary_enc = run_inference(x)
    N = binary_enc.size()[0]
    for i in range(N):
        add_to_index(index=index,
                     int_id=i,
                     bitseq=binary_enc[i])
    build_index(index, INDEX_NUM_TREES)
    save_to_disk(index, INDEX_SAVE_PATH)


def run_inference(x):
    model = torch.load(MODEL_SAVE_PATH)
    binary_enc = model.binary_encoding(x)
    return binary_enc


def int_id_mapping():
    chunks = sorted(os.listdir(LOCAL_CHUNK_FILEPATHS))
    return enumerate(chunks)


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


create_index()

