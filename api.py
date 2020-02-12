# api.py
"""
    /train call takes path to wav files and writes local chunks,
    then trains and writes to index

"""
from flask import Flask, jsonify, request
import os

import audio_ops
import constants
import sample_train

app = Flask(__name__)


@app.route('/datasetinfo', methods=['GET'])
def dataset_info():
    """ returns the number of wav files and chunks present locally """
    if request.method == "GET":
        num_wavs = len(os.listdir("./wavs"))
        num_chunks = len(os.listdir("./chunks"))
        return jsonify(({'num_wavs': num_wavs, 'num_chunks': num_chunks}))


@app.route('/add', methods=['POST'])
def add():
    """ Add wavs files"""
    if request.method == "POST":
        content = request.json
        remote_filepaths = content["filepaths"]
        local_filepaths = audio_ops.download_wav_files(remote_filepaths)
        return jsonify({'status': 'success',
                        'local_filepaths': local_filepaths})


@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        content = request.json
        remote_filepaths = content["filepaths"]
        n_epochs = content['n_epochs']
        sample_train.prepare_and_train(remote_filepaths,
                                       n_epochs=n_epochs)
        return jsonify({'status': 'success'})


if __name__ == "__main__":
    app.run()


