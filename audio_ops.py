# audio_ops.py
import os
import urllib.request
import scipy.io.wavfile

from constants import LOCAL_CHUNK_FILEPATHS, LOCAL_WAV_FILEPATHS


def chunk_audio(wav_infilepath, chunks_outdir, chunk_size):
    """ chunk the given audio file into chunks of given size
     and write to disk """
    if not os.path.exists(chunks_outdir):
        os.makedirs(chunks_outdir)
    wavname = wav_infilepath.split("/")[-1].split(".")[0]
    rate, np_audio = scipy.io.wavfile.read(wav_infilepath)
    length = np_audio.shape[0]
    i = 0
    chunks = []
    while i < length:
        np_chunk = np_audio[i * chunk_size: (i+1) * chunk_size]
        chunks.append(np_chunk)
        i += chunk_size
        fpath = os.path.abspath(os.path.join(chunks_outdir, f"{wavname}_{i}.wav"))
        scipy.io.wavfile.write(fpath, rate, np_chunk)


def download_wav_files(remote_filepaths):
    if not os.path.exists(LOCAL_WAV_FILEPATHS):
        os.mkdir(LOCAL_WAV_FILEPATHS)
    local_filepaths = []
    for url in remote_filepaths:
        fname = url.split("/")[-1]
        local_fp = os.path.join(LOCAL_WAV_FILEPATHS, fname)
        #local_fp = f'wavs/{fname}'
        if not os.path.exists(local_fp):
            print(f"Downloading {url}...")

            urllib.request.urlretrieve(url, filename=local_fp)
        else:
            print(f"{fname} already downloaded")
        local_filepaths.append(local_fp)
    print("finished downloads")
    return local_filepaths



