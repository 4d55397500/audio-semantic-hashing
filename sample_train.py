# sample_train.py

import numpy as np
import tensorflow as tf


from semantic_hashing import SemanticHashing
from constants import SAMPLE_AUDIO, WAV_CHUNK_SIZE, \
    ENCODED_BITSEQ_LENGTH
from misc_ops import build_keys_list, all_filenames, get_local_filepaths, \
    audio_to_numpy
from audio_ops import download_wav_files


def prepare_and_train(remote_file_paths, n_epochs):
    tf.compat.v1.disable_eager_execution()

    filenames = all_filenames(remote_file_paths)
    download_wav_files(remote_file_paths)

    local_filepaths = list(get_local_filepaths("./wavs", filenames))
    all_chunks, x_train = audio_to_numpy(local_filepaths)
    # print(f"training set shape: {x_train.shape}")
    ash = SemanticHashing(xdim=WAV_CHUNK_SIZE, hdim=ENCODED_BITSEQ_LENGTH)
    ash.train(x_train=x_train, batch_size=300, n_epochs=n_epochs)
    return local_filepaths, all_chunks, ash


def sample_train():

    tf.compat.v1.disable_eager_execution()
    remote_file_paths = [url for urls in SAMPLE_AUDIO.values() for url in urls]
    local_filepaths = all_chunks, ash = prepare_and_train(remote_file_paths,
                                                          n_epochs=100)
    keys = build_keys_list(local_filepaths)

    num_test_samples = 10

    test_indices = np.random.choice(len(all_chunks), num_test_samples)
    x_test = np.array(all_chunks)[test_indices]
    encoded_x = ash.encode(x_test)
    bit_seqs = set()
    bitseqmp = {}
    for i, encoded_vec in enumerate(encoded_x):
        key = keys[test_indices[i]]
        bitseq = ''.join([str(int(e)) for e in encoded_vec])
        bit_seqs.add(bitseq)
        try:
            bitseqmp[bitseq]
        except:
            bitseqmp[bitseq] = []
        bitseqmp[bitseq] += [key]
        print(f"{key} -> bit sequence: {bitseq}")
    print(f"{len(bit_seqs)} distinct bit sequences")
    for k in bitseqmp.keys():
        print(f"{k} -> {bitseqmp[k]}")


if __name__ == '__main__':
    sample_train()


