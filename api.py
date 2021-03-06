# api.py
"""
    /train call takes path to wav files and writes local chunks,
    then trains and writes to index

"""
from flask import Flask, jsonify, request
import os
import atexit


import audio_ops
import local_index
import ops
import training
import constants
import custom_exceptions


app = Flask(__name__)


@app.before_first_request
def activate_job():
    ops.ensure_dirs()

@app.route('/datasetinfo', methods=['GET'])
def dataset_info():
    """ returns the number of wav files and chunks present locally """
    if request.method == "GET":
        num_wavs = len(os.listdir(constants.LOCAL_WAV_FILEPATHS))
        num_chunks = len(os.listdir("./chunks"))
        return jsonify(({'num_wavs': num_wavs, 'num_chunks': num_chunks}))


@app.route('/add', methods=['POST'])
def add():
    """ Add wav files and create chunks"""
    if request.method == "POST":
        content = request.json
        print(content)
        remote_filepaths = content["filepaths"]
        local_filepaths = audio_ops.download_wav_files(remote_filepaths)
        for wav_fp in local_filepaths:
            audio_ops.chunk_write_audio(wav_fp)
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
        try:
            local_index.create_index()
            return jsonify({'status': 'success'})
        except custom_exceptions.ModelNotFoundException:
            return jsonify({'status': 'model not found'})


@app.route("/search", methods=["POST"])
def search():
    """ Search for nearest matching elements in the index,
    by chunking a passed audio wav and querying the index
    for each such chunk.
    """
    if request.method == "POST":
        wav = request.files['file']
        wav_bytes = wav.read()
        try:
            results = local_index.run_search(wav_bytes, n_neighbors=10, top_k=10)
            return jsonify({'results': results})
        except custom_exceptions.IndexNotFoundException:
            print("Index does not exist")
            return jsonify({'results': []})


if __name__ == "__main__":
    atexit.register(ops.clean_dirs)
    app.run()


