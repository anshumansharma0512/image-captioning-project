>>**Image Captioning using ResNet-50 Encoder + Attention LSTM Decoder (Flickr8k Dataset)**

This project implements a multimodal deep learning system that generates natural language captions for images.
It uses a CNN–RNN Encoder–Decoder architecture with Attention, trained on the Flickr8k dataset.

This is a complete project containing:

-Fully modular PyTorch code
-Dataset preprocessing pipeline
-Vocabulary builder
-Encoder–Decoder model with attention
-Training & inference scripts
-Environment setup scripts (Windows & Linux/Mac)
-Professional folder structure

**Project report**

🏗 Project Structure
image-captioning-project/
│
├── README.md
├── requirements.txt
├── setup.bat                  # Windows environment setup
│
├── vocab.py                   # Vocabulary builder + tokenizer
├── dataset.py                 # Flickr8k dataset loader
├── models.py                  # EncoderCNN, Attention & Decoder models
├── train.py                   # Model training script
├── evaluate.py                # Inference script
│
├── reports/
│   └── Project_Report.txt     # A short project report
│
└── scripts/
    └── run_sample_inference.bat

**📊 Dataset: Flickr8k (Kaggle)**

Dataset Name:

-Flickr8k Dataset by adityajn105

Contains:
-8,000+ labeled images
-Each image has 5 human-written captions
-Captions stored in:
-Flickr8k.token.txt

Images stored in:
-Flickr8k_images/

Place the dataset exactly in this format for training.

**🤖 Model Architecture**
1. Encoder (Image Feature Extractor)
-Uses ResNet-50 pretrained on ImageNet
-Removes the classification head
-Extracts a feature map
-Applies Adaptive Average Pooling to get fixed spatial dimensions
-Flattens features → produces num_pixels × 2048 feature vectors
2. Attention Mechanism
-Computes soft attention weights over spatial features
-Helps decoder focus on relevant parts of the image
-Inspired by: Show, Attend and Tell (Xu et al., 2015)
3. Decoder (Caption Generator)
-Embedding layer for word tokens
-LSTMCell for sequential caption generation
-Inputs per timestep:
-Previous word embedding
-Attention context vector
-Outputs vocabulary probability distribution
-Ends at <EOS> token

**🧪 Training Details**
Hyperparameters
Component	Value
Encoder	ResNet-50 (frozen)
Attention Dim	512
Embedding Dim	512
Decoder Hidden Dim	512
Optimizer	Adam
Learning Rate	1e-4
Loss	CrossEntropy (PAD ignored)
Epochs (default)	5
Batch Size	64
