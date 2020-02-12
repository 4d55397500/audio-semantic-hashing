# audio_ops.py
import scipy

def chunk_audio(wav_infilepath, chunks_outfilepath, chunk_size:
    """ chunk the given audio file into chunks of given size """
    rate, np_audio = scipy.io.wavefile.read(wav_infilepath)




def chunk_audio(local_filepaths):

    CHUNK_SIZE = 10000
    all_chunks = []
    for fname in local_filepaths:
        rate, numpy_audio = scipy.io.wavfile.read(fname)
        x = numpy_audio.flatten()
        all_chunks += [normalize(x[i: i + CHUNK_SIZE]) for i in
                       range(int(x.shape[0] / CHUNK_SIZE))]
    x_train = np.vstack(all_chunks)
    return all_chunks, x_train

