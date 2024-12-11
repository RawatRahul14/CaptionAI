import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
from pathlib import Path
from CaptionAI.utils.model import EncoderCNN, DecoderRNN, Attention
import torch.nn as nn
import pickle

# Loading Vocab
def load_vocab():
    vocab_path = Path("artifacts/tokenization/data")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab

# Model Initiate
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

# Function to preprocess the image
def preprocess_image(image, transform):
    """Preprocess the image for the model."""
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to generate a caption
def generate_captions(model, image, vocab, max_caption_length = 20):
    """Generate a caption for an image using the trained model."""
    # Define the transform for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    image_tensor = preprocess_image(image, transform)

    # Ensure the model is in evaluation mode
    model.eval()

    # Move image tensor to the same device as the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_tensor = image_tensor.to(device)
    model = model.to(device)

    # Get the image features from the encoder
    with torch.no_grad():
        features = model.encoder(image_tensor)
        caps, _ = model.decoder.generate_caption(features, vocab = vocab)
        caption = " ".join(caps)

    return caption

# Streamlit UI
def load_model():
    model_path = Path("artifacts/model_trainer/attention_model_state.pth")
    checkpoint = torch.load(model_path, map_location = torch.device('cpu'))

    emb_size = checkpoint["emb_size"]
    vocab_size = checkpoint["vocab_size"]
    attn_size = checkpoint["attn_size"]
    enc_hidden_size = checkpoint["enc_hidden_size"]
    dec_hidden_size = checkpoint["dec_hidden_size"]

    model = Img2Caption(
        emb_size = emb_size,
        vocab_size = vocab_size,
        attn_size = attn_size,
        enc_hidden_size = enc_hidden_size,
        dec_hidden_size = dec_hidden_size
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def main():
    # Set the page title
    st.set_page_config(page_title = "Caption Creation")

    # Sidebar
    with st.sidebar:
        st.header("Upload Your Image")
        uploaded_image = st.file_uploader("Choose an image file", type = ["jpg", "png"])

    # Main Content
    st.title("Caption Creation")
    st.write("Upload an image in the sidebar to create a caption.")

    model = load_model()
    vocab = load_vocab()

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption = "Uploaded Image", use_container_width=True)
        st.success("Image uploaded successfully!")

        # Generate and display the caption
        caption = generate_captions(model, image, vocab)
        st.write(f"**Caption:** {caption[:-6]}")
    else:
        st.info("Please upload an image to generate a caption.")

if __name__ == "__main__":
    main()