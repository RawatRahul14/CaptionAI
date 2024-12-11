from CaptionAI.utils.model import EncoderCNN, DecoderRNN, Attention
from CaptionAI.utils.common import get_device
from CaptionAI import logger
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from tqdm import tqdm
from CaptionAI.entity.config_entity import ModelTrainerConfig

class Img2Caption(nn.Module):
    def __init__(self,
                 emb_size,
                 vocab_size,
                 attn_size,
                 enc_hidden_size,
                 dec_hidden_size,
                 drop_prob = 0.3):
        super(Img2Caption, self).__init__()

        self.encoder = EncoderCNN()

        self.decoder = DecoderRNN(
            embd_size = emb_size,
            vocab_size = vocab_size,
            attn_size = attn_size,
            enc_hidden_state = enc_hidden_size,
            dec_hidden_state = dec_hidden_size
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)
        return output
    
class ModelTrain:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.data_loader = None
        self._get_vocab()
        self._read_data_loader()
        self._init_model()

    def _get_vocab(self):
        with open(self.config.token_dir, "rb") as f:
            self.vocab = pickle.load(f)

    def _init_model(self):

        self.device = get_device()

        self.model = Img2Caption(
            emb_size = self.config.emb_size,
            vocab_size = len(self.vocab),
            attn_size = self.config.attn_size,
            enc_hidden_size = self.config.enc_hidden_size,
            dec_hidden_size = self.config.dec_hidden_size
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index = self.vocab["<pad>"])
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = self.config.learning_rate
        )
        logger.info("Model is Initiated.")

    def save_model(self, model, num_epochs):

        model_state = {
            "num_epcohs": num_epochs,
            "emb_size": self.config.emb_size,
            "vocab_size": len(self.vocab),
            "attn_size": self.config.attn_size,
            "enc_hidden_size": self.config.enc_hidden_size,
            "dec_hidden_size": self.config.dec_hidden_size,
            "state_dict": model.state_dict()
        }
        torch.save(model_state, f"{self.config.root_dir}/attention_model_state.pth")

    def _read_data_loader(self):
        with open(self.config.data_dir, "rb") as f:
            self.data_loader = pickle.load(f)

    def train_model(self, num_epochs: int = 10, print_every: int = 100):

        logger.info("Model training has been started.")
        for epoch in range(num_epochs):
            with tqdm(enumerate(self.data_loader), total = len(self.data_loader), desc = f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for idx, (image, captions) in pbar:
                    image, captions = image.to(self.device), captions.to(self.device)
                    self.model.train()
                    self.optimizer.zero_grad()

                    outputs, attentions = self.model(image, captions)

                    targets = captions[:, 1:]

                    loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.reshape(-1))

                    loss.backward()
                    self.optimizer.step()
                    if (idx + 1) % print_every == 0:
                        logger.info(f"Epoch [{epoch}/{num_epochs}], Step [{idx + 1}/{len(self.data_loader)}], Loss: {loss.item()}")

                        self.model.eval()
                        with torch.no_grad():
                            img, _ = next(iter(self.data_loader))
                            features = self.model.encoder(img[0:1].to(self.device))
                            caps, attn_weights = self.model.decoder.generate_caption(features, self.vocab)

                            caption = " ".join(caps)
                            print(caption)

                        self.model.train()

                self.save_model(self.model, epoch)