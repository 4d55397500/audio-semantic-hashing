# audio semantic hashing



An API for semantic hashing and indexing for local search of audio wav files.

Background
---

Semantic hashing consists of an encoder and decoder.
Increasing amounts of noise are introduced after the encoder prior to a sigmoid activation, forcing the network to saturate the sigmoid output while also preserving information for the decoder.

The sigmoid saturated output constitutes a binary compressed representation of the data.


An index over binary compressed representations of audio chunks facilitates neighbor
search using a hamming distance.

Implementation
---
#### Pre-processing
Ideally pre-process as done in Wavenet, with mu transform

#### Autoencoder
The current implementation consists of dense encoders and decoders.
More appropriate for audio timeseries would be a convolutional autoencoder.

#### Indexing
Currently using the Spotify annoy library to index binary codes and run local search by the Hamming distance criterion.

Requirements
--
See the `requirements.txt`

API
--
