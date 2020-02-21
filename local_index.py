# local_index.py
""""
    A local index over binary sequences using the Hamming distance
    function in the annoy library

    the library requires audio chunks (or whatever object being indexed)
    to be identified by an integer
"""
from annoy import AnnoyIndex
import io
import os
import pickle
import torch

from audio_ops import chunks_dir_to_torch_tensor, \
    chunk_audio, chunks_to_torch_tensor
from constants import ENCODED_BITSEQ_LENGTH, \
    LOCAL_CHUNK_FILEPATHS, MODEL_SAVE_PATH, \
    INDEX_DIR, INDEX_SAVE_PATH, ID_MAPPING_SAVE_PATH,\
    INDEX_NUM_TREES, WAV_CHUNK_SIZE
from custom_exceptions import ModelNotFoundException


def create_index():

    if not os.path.exists(MODEL_SAVE_PATH):
        raise ModelNotFoundException
    index = initialize_index()
    x = chunks_dir_to_torch_tensor(LOCAL_CHUNK_FILEPATHS)
    id_mapping = int_id_mapping()
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)
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


def run_search(wav_bytes, n_neighbors=2, top_k=25):
    index = load_from_disk(INDEX_SAVE_PATH)
    fp = io.BytesIO(wav_bytes)
    _, chunks = chunk_audio(fp, chunk_size=WAV_CHUNK_SIZE)
    x = chunks_to_torch_tensor(chunks)
    binary_enc = run_inference(x)
    top_k = top_k
    top_k_list = []
    for search_vector in binary_enc:
        neighbors = query_by_vector(index, search_vector, n_neighbors)
        indices, distances = neighbors
        for ix, dist in zip(indices, distances):
            if len(top_k_list) < top_k:
                top_k_list.append((ix, dist))
            top_k_list = sorted(top_k_list, key=lambda e: e[1])
            if dist < top_k_list[-1][1]:
                for j, e in enumerate(top_k_list):
                    if dist < e[1]:
                        head = top_k_list[:j]
                        tail = top_k_list[j+1:] if j < top_k - 1 else []
                        top_k_list = head + [(ix, dist)] + tail
    print(top_k_list)
    return top_k_list


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
    return index.get_nns_by_item(int_id, n)


def query_by_vector(index, v, n):
    # query n nearest neighbors passing a vector
    return index.get_nns_by_vector(v, n, include_distances=True)



