import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from CaptionAI.utils.common import get_device

device = get_device()

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        for params in resnet.parameters():
            params.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):

        features = self.resnet(images)
        batchs, feature_maps, size_1, size_2 = features.size()

        features = features.permute(0, 2, 3, 1)
        features = features.view(batchs, size_1 * size_2, feature_maps)
        return features
    
class Attention(nn.Module):
    def __init__(self,
                 enc_hidden_size,
                 dec_hidden_state,
                 attn_size):
        super(Attention, self).__init__()

        self.attn_size = attn_size

        self.enc_U = nn.Linear(enc_hidden_size, attn_size)
        self.dec_W = nn.Linear(dec_hidden_state, attn_size)

        self.full_A = nn.Linear(attn_size, 1)

    def forward(self, features, decoder_hidden_state):

        decoder_hidden_state = decoder_hidden_state.unsqueeze(1)

        enc_attn = self.enc_U(features)
        dec_attn = self.dec_W(decoder_hidden_state)

        combined_state = torch.tanh(enc_attn + dec_attn)

        attn_scores = self.full_A(combined_state)
        attn_scores = attn_scores.squeeze(2)

        attn_weight = F.softmax(attn_scores, dim = 1)
        context = torch.sum(attn_weight.unsqueeze(2) * features, dim = 1)

        return attn_weight, context
    
class DecoderRNN(nn.Module):
    def __init__(self,
                 embd_size, vocab_size, attn_size,
                 enc_hidden_state, dec_hidden_state,
                 drop_prob: float = 0.3):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.attn_size = attn_size
        self.dec_hidden_state = dec_hidden_state

        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.attention = Attention(enc_hidden_state, dec_hidden_state, attn_size)

        self.init_h = nn.Linear(enc_hidden_state, dec_hidden_state)
        self.init_c = nn.Linear(enc_hidden_state, dec_hidden_state)

        self.lstm_cell = nn.LSTMCell(embd_size + enc_hidden_state, dec_hidden_state, bias = True)
        self.fc = nn.Linear(dec_hidden_state, vocab_size)
        self.drop_prob = nn.Dropout(drop_prob)

    def init_hidden_state(self, features):
        mean_features = torch.mean(features, dim = 1)
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)

        return h, c
    
    def forward(self, features, captions):

        embeds = self.embedding(captions)
        h, c = self.init_hidden_state(features)

        seq_len = len(captions[0]) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        attn_weights = torch.zeros(batch_size, seq_len, num_features).to(device)

        for t in range(seq_len):

            attn_weight, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, t], context), dim = 1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fc(self.drop_prob(h))
            preds[:, t, :] = output
            attn_weights[:, t] = attn_weight

        return preds, attn_weights
    
    def generate_caption(self, features, max_len: int = 20, vocab = None):

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)

        attn_weights = []

        word = torch.tensor(vocab["<sos>"]).view(1, -1).to(device)
        embds = self.embedding(word)

        captions = []

        for i in range(max_len):
            attn_weight, context = self.attention(features, h)
            attn_weights.append(attn_weight.cpu().detach().numpy())

            lstm_input = torch.cat((embds[:, 0], context), dim = 1)

            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fc(self.drop_prob(h))
            output = output.view(batch_size, -1)

            predicted_word_idx = output.argmax(dim = 1)
            captions.append(predicted_word_idx.item())

            if vocab.get_itos()[predicted_word_idx.item()] == "<eos>":
                break

            embds = self.embedding(predicted_word_idx.unsqueeze(0))
        return [vocab.get_itos()[idx] for idx in captions], attn_weights