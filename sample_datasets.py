# sample_datasets.py
"""

 download sample torchaudio datasets

"""

import os
import torchaudio


import constants


def download_yes_no():
    #if not os.path.exists(constants.LOCAL_WAV_FILEPATHS):
        #os.mkdir(constants.LOCAL_WAV_FILEPATHS)
    yesno_data = torchaudio.datasets.YESNO(
        root=os.path.curdir,
        #folder_in_archive='wavs',
        download=True)


download_yes_no()

