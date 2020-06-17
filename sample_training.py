# sample_training.py


import constants
import ops
import torchaudio_datasets
import training


def sample_training():
    ops.ensure_dirs()
    torchaudio_datasets.download_yes_no()
    ops.prepare_audio()
    training.train_pytorch(batch_size=constants.TRAINING_BATCH_SIZE,
                           n_epochs=100)
    ops.clean_dirs()


if __name__ == "__main__":
    sample_training()
