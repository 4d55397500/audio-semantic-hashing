# training.py

import os
import tensorflow as tf
import torch
from pytorch_sh import SemanticHashing, \
        DenseEncoder, DenseDecoder

# from semantic_hashing import SemanticHashing
from constants import WAV_CHUNK_SIZE, \
    ENCODED_BITSEQ_LENGTH, LOCAL_CHUNK_FILEPATHS, MODEL_SAVE_DIR, \
    MODEL_SAVE_PATH

from audio_ops import chunks_dir_to_numpy


# def train(batch_size, n_epochs):
#
#     x_train = chunks_dir_to_numpy(LOCAL_CHUNK_FILEPATHS)
#     assert x_train.shape[1] == WAV_CHUNK_SIZE, \
#         "incorrect training input dimensions"
#
#     tf.compat.v1.disable_eager_execution()
#
#     model = SemanticHashing(
#         xdim=WAV_CHUNK_SIZE,
#         hdim=ENCODED_BITSEQ_LENGTH)
#     model.train(
#         x_train=x_train,
#         batch_size=batch_size,
#         n_epochs=n_epochs)
#     return model


def train_pytorch(batch_size, n_epochs):

    if not os.path.exists(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    x_train = torch.tensor(chunks_dir_to_numpy(LOCAL_CHUNK_FILEPATHS)).float()
    assert x_train.shape[1] == WAV_CHUNK_SIZE, \
        "incorrect training input dimensions"

    de = DenseEncoder()
    dd = DenseDecoder()
    model = SemanticHashing(
        encoder=de,
        decoder=dd
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    N = x_train.size()[0]
    for epoch in range(n_epochs):
        permutation = torch.randperm(N)
        epoch_loss = 0.
        for i in range(0, N, batch_size):
            optimizer.zero_grad()
            if i + batch_size >= N:
                k = i + batch_size - N
                indices = torch.cat((permutation[i:], permutation[:k]), 0)
                assert (len(indices) == batch_size)
            else:
                indices = permutation[i:i+batch_size]
            batch_x = x_train[indices]
            output = model.forward(batch_x)
            loss = criterion(output, batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            noise_sigma = model.noise_sigma
        if epoch % 10 == 0:
            print(f"epoch: {epoch} epoch loss: {epoch_loss} noise sigma: {noise_sigma}")
            print(f"saving model to {MODEL_SAVE_PATH}")
            torch.save(model, MODEL_SAVE_PATH)


