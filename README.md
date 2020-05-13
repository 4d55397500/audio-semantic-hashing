# audio semantic hashing
[![Build Status](https://travis-ci.com/4d55397500/audio-semantic-hashing.svg?branch=master)](https://travis-ci.org/4d55397500/audio-semantic-hashing)


Semantic hashing and indexing for local search of audio wav files.

Architecture
---

### Network Diagrams

semantic hashing architecture

```

	input -> encoder -|
					  | -> (+)-> [sigmoid activation] -> decoder -> output
	noise -> |--------|
	
```	

audio preprocessing

```
raw input -> [normalization] -> [mu transform] -> [chunking] -> input

```

encoder architecture

```
left-padded conv (d=2^0) -> left-padded conv (d=2^1) -> ...
```

decoder architecture

```
left-padded conv (d=2^N) -> left-padded conv (d=2^(N-1)) -> ...
```





insert system architecture diagram here

Background
---

Semantic hashing consists of an encoder and decoder.
Increasing amounts of noise are introduced after the encoder prior to a sigmoid activation, forcing the network to saturate the sigmoid output while also preserving information for the decoder. The original paper dating back to 2007 can be found 
[here](https://www.cs.utoronto.ca/~rsalakhu/papers/semantic_final.pdf).

The sigmoid saturated output constitutes a binary compressed representation of the data.


An index over binary compressed representations of audio chunks facilitates neighbor
search using a Hamming distance.


This project is for the purpose of demonstrating semantic hashing. A more appropriate approach would be to train a continuous embedding and then use a vector-based index like [faiss](https://github.com/facebookresearch/faiss).

Implementation
---
#### Pre-processing
Audio is broken up into chunks, normalized, and quantized into one of 256 disrete values according to to the mu companding transform.

#### Encoder & Decoder
The encoder and decoder are stacks of increasingly/decreasingly dilated convolutional layers, as in Wavenet. The convolutional layers in theory enable similar bit representations for similar events but at different times in the audio.; the Wavenet-style composition of dilations in theory enables efficient long-range correlations. However we will keep the effective kernel length less than the chunk size we use.

#### Indexing
Currently using the Spotify [annoy](https://github.com/spotify/annoy) library to index binary codes and run local search by the Hamming distance criterion.

#### Parameters

Specifiable parameters are stored in `constants.py`. These include values determining the audio pre-processing, as well as neural network and index structures.

Data Representations
--
Single channel audio are assumed given as 16-bit PCM wav files, read into numpy arrays/torch tensors of shape
`[num_samples, audio_length]`. 

These are normalized into [-1, 1] by dividing by 2^15, then fed into a mu companding transform of 256 quantization levels. This will define for each audio sample point a one-hot 256-dimensional vector.

Each audio vector is broken up into segments of length `CHUNK_SIZE`. The formal input to the network is vectors of 256 dimensional one-hots, of shape `[batch_size, 256, CHUNK_SIZE]`.

The encoder performs a series of dilated left-padded convolutions, mapping a batch of audio chunks to an encoded representation of shape `[batch_size, ENCODED_BITSEQ_LENGTH]`.

The decoder reverses this, outputting a batch of shape
`[batch_size, 256, CHUNK_SIZE]`. A per-256 dim vector cross entropy with corresponding softmax is used as the loss function.

Audio Sources
--

tbd

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

