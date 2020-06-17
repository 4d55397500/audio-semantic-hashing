import os
import shutil

import audio_ops
import constants


def prepare_audio():
    for root, dirs, files in os.walk(constants.LOCAL_WAV_FILEPATHS):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith('.wav'):
                audio_ops.chunk_write_audio(path)

def ensure_dirs():
    for dir in constants.RUNTIME_DIRS:
        if not os.path.exists(dir):
            os.mkdir(dir)


def clean_dirs():
    for dir in constants.RUNTIME_DIRS:
        try:
            shutil.rmtree(dir)
        except:
            pass

