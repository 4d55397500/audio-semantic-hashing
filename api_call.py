import requests
from constants import SAMPLE_AUDIO

remote_file_paths = [url for urls in SAMPLE_AUDIO.values() for url in urls]

resp = requests.post("http://localhost:5000/train",
                     json={"filepaths": remote_file_paths})
print(resp.json())