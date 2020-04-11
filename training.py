# training.py

import os
import torch
from semantic_hashing import SemanticHashing
from conv_decoder import ConvDecoder
from conv_encoder import ConvEncoder

from constants import WAV_CHUNK_SIZE, \
    LOCAL_CHUNK_FILEPATHS, MODEL_SAVE_DIR, \
    MODEL_SAVE_PATH, TRAINING_BATCH_SIZE

from audio_ops import chunks_dir_to_numpy

celoss = torch.nn.CrossEntropyLoss()


def loss_criterion(output, target):
    target_index_vector = torch.argmax(target, dim=1)
    loss_value = celoss(
        input=output,
        target=target_index_vector)
    return loss_value


def train_pytorch(batch_size, n_epochs):

    if not os.path.exists(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    x_train = torch.tensor(
        chunks_dir_to_numpy(LOCAL_CHUNK_FILEPATHS))

    de = ConvEncoder()
    dd = ConvDecoder()
    model = SemanticHashing(
        encoder=de,
        decoder=dd
    )

    criterion = loss_criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    N = x_train.size()[0]
    print("starting training loop ...")
    for epoch in range(n_epochs):
        permutation = torch.randperm(N)
        epoch_loss = 0.
        n_batches = N / batch_size
        for i in range(0, N, batch_size):
            optimizer.zero_grad()
            if i + batch_size >= N:
                k = i + batch_size - N
                indices = torch.cat((permutation[i:], permutation[:k]), 0)
            else:
                indices = permutation[i:i+batch_size]
            batch_x = x_train[indices]
            output = model.forward(batch_x)
            encoded_entropy = model.encoded_entropy(batch_x)
            loss = criterion(output, batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            noise_sigma = model.noise_sigma
            print(
                f"batch loss: {loss.item()} "
                f"batch encoded entropy: {encoded_entropy} "
                f"noise sigma: {noise_sigma}\n"
                f"----------------------------")
        if epoch % 1 == 0:
            loss_criterion(output, target=batch_x)
            print(f"""
            epoch: {epoch} epoch loss: {epoch_loss / n_batches}
            noise sigma: {noise_sigma}
            encoded entropy: {encoded_entropy}
            """)
            print(f"saving model to {MODEL_SAVE_PATH}")
            print("-" * 20)
            torch.save(model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    train_pytorch(batch_size=TRAINING_BATCH_SIZE, n_epochs=100)

