# api.py
"""
    /train call takes path to wav files and writes local chunks,
    then trains and writes to index

"""
from flask import Flask, jsonify, request
import os

import audio_ops
import local_index
import training


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
    """ Add wav files and create chunks"""
    if request.method == "POST":
        content = request.json
        remote_filepaths = content["filepaths"]
        local_filepaths = audio_ops.download_wav_files(remote_filepaths)
        for wav_fp in local_filepaths:
            audio_ops.chunk_audio(wav_fp)
        return jsonify({'status': 'success',
                        'local_filepaths': local_filepaths})


@app.route('/train', methods=['POST'])
def train():
    """ Train a semantic hashing model over existing audio chunks
     in local directory
     """
    if request.method == 'POST':
        content = request.json
        n_epochs = content['n_epochs']
        training.train_pytorch(batch_size=300, n_epochs=n_epochs)
        return jsonify({'status': 'success'})


@app.route('/index', methods=["POST"])
def index():
    """ Index existing audio chunks in local directory
    using their binary encoding given by the model
    """
    if request.method == "POST":
        local_index.build_index()
        return jsonify({'status': 'success'})


@app.route("/search", method=["GET"])
def search():
    """ Search for nearest matching elements in the index,
    by chunking a passed audio wav and querying the index
    for each such chunk.
    """
    pass


if __name__ == "__main__":
    app.run()


