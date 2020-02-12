# training.py
import tensorflow as tf

from semantic_hashing import SemanticHashing
from constants import WAV_CHUNK_SIZE, \
    ENCODED_BITSEQ_LENGTH, LOCAL_CHUNK_FILEPATHS

from audio_ops import chunks_to_numpy


def train(batch_size, n_epochs):

    x_train = chunks_to_numpy(LOCAL_CHUNK_FILEPATHS)
    assert x_train.shape[1] == WAV_CHUNK_SIZE, \
        "incorrect training input dimensions"

    tf.compat.v1.disable_eager_execution()

    SemanticHashing(
        xdim=WAV_CHUNK_SIZE,
        hdim=ENCODED_BITSEQ_LENGTH).train(
        x_train=x_train,
        batch_size=batch_size,
        n_epochs=n_epochs)

