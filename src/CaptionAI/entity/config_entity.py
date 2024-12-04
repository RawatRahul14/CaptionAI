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