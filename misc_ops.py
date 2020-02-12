import os

import numpy as np
import scipy.io

from constants import WAV_CHUNK_SIZE, SAMPLE_AUDIO


def build_keys_list(local_filepaths):

    keys = []
    for fname in local_filepaths:
        rate, numpy_audio = scipy.io.wavfile.read(fname)
        x = numpy_audio.flatten()
        key = filename_to_key(fname.split("/")[-1])
        keys += [key] * int(x.shape[0] / WAV_CHUNK_SIZE)
    return keys


def filename_to_key(fname):
    for k in SAMPLE_AUDIO.keys():
        links = SAMPLE_AUDIO[k]
        if sum([int(fname in a) for a in links]) > 0:
            return k


def all_filenames(filepaths):
    return [url.split("/")[-1] for url in filepaths]


def get_local_filepaths(directory, filenames):
    for dirpath, _, fnames in os.walk(directory):
        for f in fnames:
            if f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))


def normalize_np_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def audio_to_numpy(local_filepaths):

    all_chunks = []
    for fname in local_filepaths:
        rate, numpy_audio = scipy.io.wavfile.read(fname)
        x = numpy_audio.flatten()
        all_chunks += [normalize_np_vector(x[i: i + WAV_CHUNK_SIZE]) for i in
                       range(int(x.shape[0] / WAV_CHUNK_SIZE))]
    x_train = np.vstack(all_chunks)
    return all_chunks, x_train