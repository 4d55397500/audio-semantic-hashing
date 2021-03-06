# audio_ops.py
"""

 !!!! Currently normalizing into [-1, 1] by assuming
     16-bit PCM, so dividing by 2**15

"""
import os
import urllib.request
import numpy as np
import scipy.io.wavfile
import torch
import torchaudio


from constants import LOCAL_CHUNK_FILEPATHS, \
    LOCAL_WAV_FILEPATHS, WAV_CHUNK_SIZE



def chunk_write_audio(wav_infilepath,
                      chunks_outdir=LOCAL_CHUNK_FILEPATHS,
                      chunk_size=WAV_CHUNK_SIZE):
    """ chunk the given audio file into chunks of given size
     and write to disk """
    print(f"chunking audio for {wav_infilepath}...")
    if not os.path.exists(chunks_outdir):
        os.makedirs(chunks_outdir)
    wavname = wav_infilepath.split("/")[-1].split(".")[0]
    rate, chunks = chunk_audio(wav_infilepath, chunk_size)
    for i, np_chunk in enumerate(chunks):
        fpath = os.path.abspath(os.path.join(chunks_outdir,
                                             f"{wavname}_{i}.wav"))
        if not os.path.exists(fpath):
            scipy.io.wavfile.write(fpath, rate, np_chunk)


def chunk_audio(wav_filepath, chunk_size=WAV_CHUNK_SIZE):
    # normalize in this method
    rate, np_audio = scipy.io.wavfile.read(wav_filepath)
    np_audio = np_audio.flatten()
    length = np_audio.shape[0]
    i = 0
    chunks = []
    while i + chunk_size <= length:
        np_chunk = np_audio[i: i + chunk_size]
        chunks.append(np_chunk)
        i += chunk_size
    return rate, chunks


def chunks_to_torch_tensor(chunks):
    return torch.tensor(chunks_to_numpy(chunks)).float()


def chunks_dir_to_torch_tensor(chunks_dir):
    return torch.tensor(chunks_dir_to_numpy(chunks_dir)).float()


def chunks_dir_to_numpy(chunks_dir):
    chunks = []
    for fp in sorted(os.listdir(chunks_dir)):
        rate, v = scipy.io.wavfile.read(os.path.join(chunks_dir, fp))
        chunks.append(v)
    return chunks_to_numpy(chunks)


def chunks_to_numpy(chunks):
    mu_chunks = list(mu_transform(v) for v in chunks)
    z = np.vstack(mu_chunks)
    return z


def mu_transform(v, quantization_channels=256):
    #!! note the torchaudio method expects the input to already be normalized
    # into [-1, 1], otherwise it will not give an output in the quantization bounds
    # 0....quantization_channels - 1
    # !!!! Assuming 16-bit PCM, so dividing by 2**15 for normalization
    # additional performs a one-hot after the quantization
    normalized_v = v / 2. ** 15
    mu_v = torchaudio.functional.mu_law_encoding(
        torch.tensor(normalized_v),
        quantization_channels=quantization_channels)
    if len(mu_v.shape) == 1:
        mu_v = mu_v.unsqueeze(dim=0)
    return torch.nn.functional.one_hot(mu_v, 256).permute(0, 2, 1).float()


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

