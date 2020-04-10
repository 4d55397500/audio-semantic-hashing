# sample_workflow.py

import requests

HOST = 'http://localhost:5000'
#
# resp = requests.get(f'{HOST}/datasetinfo')
# print(resp.content)
#
#
# resp2 = requests.post(f'{HOST}/add',
#                       json={'filepaths': []})
# print(resp2.content)
#
#
# resp3 = requests.post(f'{HOST}/train',
#                       json={'n_epochs': 10})
# print(resp3.content)
#
# resp4 = requests.post(f'{HOST}/index',
#                       json={})
# print(resp4.content)

sample_search_chunk = 'chunks/0_0_0_0_1_1_1_1_2.wav'
files = {'file': open(sample_search_chunk, 'rb')}
resp5 = requests.post(f'{HOST}/search',
                      files=files)
print(resp5.content)

