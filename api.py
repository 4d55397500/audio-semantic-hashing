# api.py
from flask import Flask, jsonify, request

import audio_semantic_hashing

app = Flask(__name__)

SAMPLE_FILEPATHS = [url for urls in audio_semantic_hashing.SAMPLE_AUDIO.values()
                    for url in urls]

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        content = request.json
        remote_filepaths = content["filepaths"]
        train_on_paths(remote_filepaths)

def train_on_paths(remote_filepaths):
    # download and then train on the wav audio files of the given remote filepaths
    audio_semantic_hashing.prepare_and_train(remote_filepaths)


def encode_persist(remote_filepaths):
    # encode chunks and persist byte sequences in bigtable / column-store with the corresponding
    # audio chunk as a column value
    # download if necessary else use local copy for files
    pass


if __name__ == "__main__":
    app.run()


