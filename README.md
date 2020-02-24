# audio semantic hashing
[![Build Status](https://travis-ci.com/4d55397500/audio-semantic-hashing.svg?branch=master)](https://travis-ci.org/4d55397500/audio-semantic-hashing)


An API for semantic hashing and indexing for local search of audio wav files.

Background
---

Semantic hashing consists of an encoder and decoder.
Increasing amounts of noise are introduced after the encoder prior to a sigmoid activation, forcing the network to saturate the sigmoid output while also preserving information for the decoder.

The sigmoid saturated output constitutes a binary compressed representation of the data.


An index over binary compressed representations of audio chunks facilitates neighbor
search using a Hamming distance.


This project is for the purpose of demonstrating semantic hashing. A more appropriate approach would be to train a continuous embedding and then use a vector-based index like [faiss](https://github.com/facebookresearch/faiss).

Implementation
---
#### Pre-processing
Ideally pre-process as done in Wavenet, with mu transform. Currently pre-processing consists solely of normalizing the numpy vector for each chunk.

#### Autoencoder
The current implementation consists of dense encoders and decoders.
A convolutional autoencoder (tbd) would be more appropriate for audio.

#### Indexing
Currently using the Spotify [annoy](https://github.com/spotify/annoy) library to index binary codes and run local search by the Hamming distance criterion.

#### Parameters

Specifiable parameters are stored in `constants.py`. These include values determining the audio pre-processing, as well as neural network and index structures.

Requirements
--
* Python 3
* See `requirements.txt`. 

Tests
---
From the project root run

```python3 -m unittest discover```

API
--

* `/datasetinfo`: get number of wav files and chunks present locally
* `/add`: add wav files and create chunks
* `/train`: train semantic hashing model over local wav files
* `/index`: index existing audio chunks in local directory using their learned binary encoding
* `/search`: search for nearest matching elements in the index, by chunking a passed audio wav and querying the index for each such chunk

