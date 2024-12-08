from CaptionAI.constants import *
from CaptionAI.utils.common import read_yaml, create_directories
from CaptionAI.entity.config_entity import (DataIngestionConfig,
                                            TokenizationConfig,
                                            CustomDatasetConfig)

class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            dataset_link = config.dataset_link,
            local_data_file = config.local_data_file
        )

        return data_ingestion_config
    
    def get_tokenization_config(self):
        config = self.config.tokenization
        create_directories([config.root_dir])

        tokenization_config = TokenizationConfig(
            root_dir = config.root_dir,
            token_file = config.token_file,
            caption_file = config.caption_file,
            tokenizer_type = config.tokenizer_type,
            unk_token = config.unk_token,
            pad_token = config.pad_token,
            sos_token = config.sos_token,
            eos_token = config.eos_token
        )

        return tokenization_config
    
    def get_dataset_config(self):
        config = self.config.custom_dataset
        create_directories([config.root_dir])

        dataset_config = CustomDatasetConfig(
            root_dir = config.root_dir,
            image_dir = config.image_dir,
            caption_file = config.caption_file,
            vocab = config.vocab,
            save_file_path = config.save_file_path,
            transform = False
        )

        return dataset_config