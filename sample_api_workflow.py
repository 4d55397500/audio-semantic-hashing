# sample_api_workflow.py

import requests
import json

HOST = 'http://localhost:5000'
SAMPLE_SEARCH_CHUNK = 'chunks/0_0_0_0_1_1_1_1_2.wav'

requests.get(f'{HOST}/datasetinfo')
requests.post(f'{HOST}/add',
                      json={'filepaths': []})
requests.post(f'{HOST}/train',
                      json={'n_epochs': 100})
requests.post(f'{HOST}/index',
                      json={})

with open(SAMPLE_SEARCH_CHUNK, 'rb') as handle:
    files = {'file': handle }
    resp5 = requests.post(f'{HOST}/search',
                          files=files)
    results = json.loads(resp5.content)['results']
    for e in results:
        wav_bytes, distance = tuple(e)
        print(f"Found matched chunk: {wav_bytes[:10]}... distance: {distance}")

