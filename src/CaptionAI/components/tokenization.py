from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from CaptionAI import logger
import pandas as pd
import pickle
from collections import Counter
from CaptionAI.entity.config_entity import TokenizationConfig

class Tokenization:
    def __init__(self, config: TokenizationConfig):
        self.config = config

    def init_tokenizer(self):
        logger.info("Initializing the Tokenizer.")
        self.tokenizer = get_tokenizer(self.config.tokenizer_type)
        self.counter = Counter()

    def build_vocab(self):
        logger.info("Building the vocab.")
        lines = pd.read_csv(self.config.caption_file)
        for line in lines["caption"].tolist():
            self.counter.update(self.tokenizer(line))

        self.vocab = vocab(self.counter, min_freq = 5)

        self.vocab.insert_token(self.config.unk_token, 0)
        self.vocab.insert_token(self.config.pad_token, 1)
        self.vocab.insert_token(self.config.sos_token, 2)
        self.vocab.insert_token(self.config.eos_token, 3)

        self.vocab.set_default_index(self.vocab[self.config.unk_token])

        logger.info("Finished Creating the vocab.")

    def save_vocab_pickle(self):
        with open(self.config.token_file, "wb") as f:
            pickle.dump(self.vocab, f)
        logger.info(f"Vocabulary saved to {self.config.token_file}.")