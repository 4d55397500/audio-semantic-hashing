# torchaudio_datasets.py
"""

 download sample torchaudio datasets

"""

import os
import torchaudio

import ops


def download_yes_no():
    ops.ensure_dirs()
    yesno_data = torchaudio.datasets.YESNO('wavs',
                                           download=True)



