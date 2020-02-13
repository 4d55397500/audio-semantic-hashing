# constants.py

WAV_CHUNK_SIZE = 10000
SAMPLE_AUDIO = {
    "harpsichord": ["https://ccrma.stanford.edu/~jos/wav/harpsi-cs.wav",
                    "https://ccrma.stanford.edu/~jos/wav/Harpsichord.wav"],
    "cello": ["https://ccrma.stanford.edu/~jos/wav/cello.wav"],
    "trumpet": ["https://ccrma.stanford.edu/~jos/wav/trumpet.wav"],
    "piano": ["https://ccrma.stanford.edu/~jos/wav/pno-cs.wav"]
}

INTERMEDIATE_LAYER_DIMS = [20, 10, 10]
INITIAL_NOISE_STD = 0.2
NOISE_MULTIPLICATIVE_INCREMENT = 1. + 1e-5

# length of encoded bit sequence representation
ENCODED_BITSEQ_LENGTH = 15

LOCAL_WAV_FILEPATHS = "wavs"
LOCAL_CHUNK_FILEPATHS = "chunks"

MODEL_SAVE_DIR = "model"
MODEL_SAVE_PATH = f"{MODEL_SAVE_DIR}/model.pth"

INDEX_DIR = "index"
INDEX_SAVE_PATH = f"{INDEX_DIR}/index.ann"
ID_MAPPING_SAVE_PATH = f"{INDEX_DIR}/ids.pkl"

INDEX_NUM_TREES = 100  # how to choose?

LOCAL_SEARCH_CHUNKS_DIR = "tmp"
TEST_RESOURCES_DIR = "test_resources"
