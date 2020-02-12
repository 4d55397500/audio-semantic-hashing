# audio_ops.py
import os
import urllib.request
import numpy as np
import scipy.io.wavfile

from constants import LOCAL_CHUNK_FILEPATHS, \
    LOCAL_WAV_FILEPATHS, WAV_CHUNK_SIZE


def chunk_audio(wav_infilepath,
                chunks_outdir=LOCAL_CHUNK_FILEPATHS,
                chunk_size=WAV_CHUNK_SIZE):
    """ chunk the given audio file into chunks of given size
     and write to disk """
    print(f"chunking audio for {wav_infilepath}...")
    if not os.path.exists(chunks_outdir):
        os.makedirs(chunks_outdir)
    wavname = wav_infilepath.split("/")[-1].split(".")[0]
    rate, np_audio = scipy.io.wavfile.read(wav_infilepath)
    length = np_audio.shape[0]
    i = 0
    chunks = []
    while i + chunk_size < length:
        np_chunk = np_audio[i: i + chunk_size]
        chunks.append(np_chunk)
        i += chunk_size
        fpath = os.path.abspath(os.path.join(chunks_outdir, f"{wavname}_{i}.wav"))
        scipy.io.wavfile.write(fpath, rate, np_chunk)


def chunks_to_numpy(chunks_dir):
    chunks = []
    for fp in os.listdir(chunks_dir):
        rate, v = scipy.io.wavfile.read(os.path.join(chunks_dir, fp))
        chunks.append(normalize_np_vector(v))
    return np.vstack(chunks)


def normalize_np_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def download_wav_files(remote_filepaths,
                       local_wav_filepaths=LOCAL_WAV_FILEPATHS):
    if not os.path.exists(local_wav_filepaths):
        os.mkdir(local_wav_filepaths)
    local_filepaths = []
    for url in remote_filepaths:
        fname = url.split("/")[-1]
        local_fp = os.path.join(local_wav_filepaths, fname)
        if not os.path.exists(local_fp):
            print(f"Downloading {url}...")

            urllib.request.urlretrieve(url, filename=local_fp)
        else:
            print(f"{fname} already downloaded")
        local_filepaths.append(local_fp)
    print("finished downloads")
    return local_filepaths



