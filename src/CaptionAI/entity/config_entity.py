from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir: Path
    dataset_link: str
    local_data_file: Path

@dataclass(frozen = True)
class TokenizationConfig:
    root_dir: Path
    token_file: Path
    caption_file: Path
    tokenizer_type: str
    unk_token: str
    pad_token: str
    sos_token: str
    eos_token: str

@dataclass(frozen = True)
class CustomDatasetConfig:
    root_dir: Path
    image_dir: Path
    caption_file: Path
    vocab: Path
    save_file_path: Path
    transform: bool

@dataclass(frozen = True)
class ModelTrainerConfig:
    root_dir: Path
    data_dir: Path
    token_dir: Path
    emb_size: int
    attn_size: int
    enc_hidden_size: int
    dec_hidden_size: int
    learning_rate: float