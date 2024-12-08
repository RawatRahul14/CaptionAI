import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchtext.data.utils import get_tokenizer
from torchvision import transforms as T
import pickle
import torch
from CaptionAI import logger
from torch.nn.utils.rnn import pad_sequence

class FlickrDataset(Dataset):
    def __init__(self, 
                 image_dir, 
                 caption_file, 
                 vocab_file, 
                 transform = None):
        
        self.image_dir = image_dir
        self.caption_file = caption_file
        self.transform = transform
        self.vocab_file = vocab_file

        df = pd.read_csv(caption_file)

        df = df.dropna(subset = ["caption", "image"])

        self.captions = df["caption"]
        self.img_names = df["image"]

        self.tokenizer = get_tokenizer("basic_english")
        self._load_vocab()
        self._get_transform()

    def _load_vocab(self):
        try:
            with open(self.vocab_file, "rb") as f:
                self.vocab = pickle.load(f)
            logger.info(f"Vocabulary loaded from {self.vocab_file}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary: {e}")

    def _get_transform(self):
        
        if self.transform is None:
            self.transform = T.Compose([
            T.Resize(226),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):

        image_name, caption = self.img_names[idx], self.captions[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image {image_name} not found in {self.image_dir}.")
        
        if self.transform:
            image = self.transform(image)

        caption_text_to_index = lambda x: [self.vocab.get(token, self.vocab["<unk>"]) for token in self.tokenizer(x)]
        caption_vec = [self.vocab["<sos>"]]
        caption_vec += caption_text_to_index(caption)
        caption_vec += [self.vocab["<eos>"]]

        return image, torch.tensor(caption_vec)
    
class generate_batch_captions:
    def __init__(self,
                 pad_idx,
                 batch_first = False):
        
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)

        targets = (item[1] for item in batch)
        targets = pad_sequence(sequences = targets, batch_first = self.batch_first, padding_value = self.pad_idx)

        return imgs, targets