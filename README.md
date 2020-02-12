# audio semantic hashing



An API for semantic hashing and indexing for local search of audio wav files.

Background
---

Semantic hashing consists of an encoder and decoder.
Increasing amounts of noise are introduced after the encoder prior to a sigmoid activation, forcing the network to saturate the sigmoid output while also preserving information for the decoder.

The sigmoid saturated output constitutes a binary compressed representation of the data.

Requirements
--

API
--
