# api.py
from flask import Flask, jsonify, request

import audio_semantic_hashing

app = Flask(__name__)
SAMPLE_FILEPATHS = [url for urls in audio_semantic_hashing.SAMPLE_AUDIO.values()
                    for url in urls]

# train -> bit sequences, audio chunk map structure

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        content = request.json
        remote_filepaths = content["filepaths"]
        local_filepaths = train_on_paths(remote_filepaths)
        return jsonify({'local_filepaths': local_filepaths})

def train_on_paths(remote_filepaths):
    # download and then train on the wav audio files of the given remote filepaths
    local_filepaths, all_chunks, ash = audio_semantic_hashing.prepare_and_train(remote_filepaths)
    return local_filepaths


def encode_persist(remote_filepaths):
    # encode chunks and persist byte sequences in bigtable / column-store with the corresponding
    # audio chunk as a column value
    # download if necessary else use local copy for files
    pass


if __name__ == "__main__":
    app.run()


