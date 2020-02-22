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
import base64


from audio_ops import chunks_dir_to_torch_tensor, \
    chunk_audio, chunks_to_torch_tensor
from constants import ENCODED_BITSEQ_LENGTH, \
    LOCAL_CHUNK_FILEPATHS, MODEL_SAVE_PATH, \
    INDEX_DIR, INDEX_SAVE_PATH, ID_MAPPING_SAVE_PATH,\
    INDEX_NUM_TREES, WAV_CHUNK_SIZE
from custom_exceptions import ModelNotFoundException, \
    IndexNotFoundException


def create_index():
    print("Creating index ...")
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
    print_index_statistics(binary_enc)
    N = binary_enc.size()[0]
    print(f"Writing {N} binary sequences to index ...")
    for i in range(N):
        add_to_index(index=index,
                     int_id=i,
                     bitseq=binary_enc[i])
    print("Finished writing to index")
    build_index(index, INDEX_NUM_TREES)
    save_to_disk(index, INDEX_SAVE_PATH)


def run_search(wav_bytes, n_neighbors=2, top_k=5):
    """
    Runs a nearest neighbor search on each chunk of the passed wav bytes.
    Computes a global top_k nearest neighbors across all chunks.
    Returns a descending ordered list of tuples (chunk bytes, distance)

    :param wav_bytes:
    :param n_neighbors:
    :param top_k:
    :return:
    """
    if not os.path.exists(INDEX_SAVE_PATH):
        raise IndexNotFoundException
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
            if len(top_k_list) < top_k and (ix, dist) not in top_k_list:
                top_k_list.append((ix, dist))
            top_k_list = sorted(top_k_list, key=lambda e: e[1])
            if dist < top_k_list[-1][1]:
                for j, e in enumerate(top_k_list):
                    if dist < e[1] and (ix, dist) not in top_k_list:
                        head = top_k_list[:j]
                        tail = top_k_list[j+1:] if j < top_k - 1 else []
                        top_k_list = head + [(ix, dist)] + tail
    with open(ID_MAPPING_SAVE_PATH, 'rb') as handle:
        id_mapping = pickle.load(handle)
    results = []
    for int_id, distance in top_k_list:
        with open(os.path.join(LOCAL_CHUNK_FILEPATHS, id_mapping[int_id]), 'rb') as chunk:
            chunk_bytes = base64.b64encode(chunk.read()).decode()
            results.append((chunk_bytes, distance))
    return results  # list of tuples (chunk bytes, distance)


def run_inference(x):
    model = torch.load(MODEL_SAVE_PATH)
    binary_enc = model.binary_encoding(x)
    return binary_enc


def print_index_statistics(binary_enc):
    _, counts = torch.unique(binary_enc, dim=0, return_counts=True)
    n_unique = counts.size()[0]
    probs = counts * 1. / torch.sum(counts)
    index_entropy = -torch.sum(probs * torch.log(probs))
    print(f"""
       index statistics:
           entropy: {index_entropy}
           num. unique rows: {n_unique} / {binary_enc.size()[0]}
       """)


def int_id_mapping():
    chunks = sorted(os.listdir(LOCAL_CHUNK_FILEPATHS))
    return enumerate(chunks)


def initialize_index():
    return AnnoyIndex(ENCODED_BITSEQ_LENGTH, 'hamming')


def build_index(index, n_trees):
    print("building index")
    # no more items can be added once build is called
    index.build(n_trees)


def save_to_disk(index, filename):
    print("saving index to disk")
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

