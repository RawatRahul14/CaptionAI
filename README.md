# Image Captioning with Attention Mechanism

This project is a deep learning-based solution for generating captions for images. It uses a Convolutional Neural Network (CNN) as an encoder with transfer learning, a Recurrent Neural Network (RNN) as a decoder, and an Attention Mechanism to dynamically focus on relevant parts of the image during caption generation. The implementation is done using PyTorch.


## Features
* **Transfer Learning for Encoder**: Utilizes pre-trained CNN model (like ResNet50) to extract detailed visual features from images efficiently.
* **RNN-Based Decoder**: Converts the visual features into coherent, meaningful textual descriptions.
* **Attention Mechanism**: Enhances the model's ability to focus on specific regions of an image while generating captions.
* **End-to-End Training Pipeline**: Seamlessly integrates vision and natural language processing to generate captions directly from images.

## Setup and Installation
1. Clone the repository:
```bash
git clone https://github.com/RawatRahul14/CaptionAI.git
cd CaptionAI
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv/Scripts/activate
```

3. Install the required packages for the project:
```bash
pip install -r requirements.txt
```

4. Run the main.py file:
```bash
python main.py
```

5. Run the Application:
```bash
streamlit run app.py
```

## Technologies Used
* **PyTorch**: For building and training the deep learning models.
* **Transfer Learning**: Pre-trained CNNs for efficient feature extraction.
* **Attention Mechanism**: For improved focus on relevant image regions.

