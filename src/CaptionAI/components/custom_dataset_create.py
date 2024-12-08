from CaptionAI.utils.dataset import FlickrDataset, generate_batch_captions
from torch.utils.data import DataLoader
from CaptionAI import logger
import pickle
from tqdm import tqdm
from CaptionAI.entity.config_entity import (CustomDatasetConfig)

class DatasetCreation:
    def __init__(self, config: CustomDatasetConfig):
        self.config = config
        self.flickr_dataset = None
        self._create_dataset()

    def _create_dataset(self):
        logger.info("Creating the Custom Dataset.")
        self.flickr_dataset = FlickrDataset(
            image_dir = self.config.image_dir,
            caption_file = self.config.caption_file,
            vocab_file = self.config.vocab
        )
        logger.info("Custom Dataset is created.")

    def create_dataloader(self, batch_size: int = 512):
        logger.info("Data Loader is getting created.")
        self.data_loader = DataLoader(
            dataset = self.flickr_dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = generate_batch_captions(pad_idx = 1,
                                                 batch_first = True)
        )
        logger.info("Data Loader is created.")

    def save_dataset(self):
        all_batches = []
        total_batches = len(self.data_loader)
        half_batches = total_batches // 2

        logger.info("Saving first half of the dataset batches...")
        
        for idx, batch in tqdm(enumerate(self.data_loader), desc = "Saving Batches", unit = "batch"):
            if idx >= half_batches:
                break
            all_batches.append(batch)

        with open(self.config.save_file_path, "wb") as f:
            pickle.dump(all_batches, f)

        logger.info(f"DataLoader batches saved to {self.config.save_file_path}")