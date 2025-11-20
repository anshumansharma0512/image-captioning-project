# evaluate.py
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from models import EncoderCNN, DecoderWithAttention
from vocab import Vocabulary
import numpy as np
import json
import os

def load_vocab(vocab_stoi):
    vocab = Vocabulary()
    vocab.stoi = vocab_stoi
    vocab.itos = {v:k for k,v in vocab.stoi.items()}
    return vocab

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def greedy_decode(encoder, decoder, image_tensor, vocab, max_len=30, device='cpu'):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        encoder_out = encoder(image_tensor.to(device))
        num_pixels = encoder_out.size(1)
        h, c = decoder.init_hidden_state(encoder_out)
        seq = []
        alphas = []
        word = torch.tensor([vocab.stoi["<SOS>"]]).to(device)
        for t in range(max_len):
            embedding = decoder.embedding(word).squeeze(0)
            context, alpha = decoder.attention(encoder_out, h)
            input_lstm = torch.cat([embedding, context], dim=1)
            h, c = decoder.decode_step(input_lstm, (h, c))
            out = decoder.fc(h)
            pred = out.argmax(dim=1)
            word = pred
            w = vocab.itos[pred.item()] if pred.item() in vocab.itos else "<UNK>"
            if w == "<EOS>":
                break
            seq.append(w)
            alphas.append(alpha.cpu().numpy())
        return " ".join(seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    vocab_stoi = ckpt['vocab']
    vocab = Vocabulary()
    vocab.stoi = vocab_stoi
    vocab.itos = {v:k for k,v in vocab_stoi.items()}

    encoder = EncoderCNN().to(args.device)
    decoder = DecoderWithAttention(attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=len(vocab.stoi)).to(args.device)
    encoder.load_state_dict(ckpt['encoder_state'])
    decoder.load_state_dict(ckpt['decoder_state'])

    img_tensor = preprocess_image(args.img_path)
    caption = greedy_decode(encoder, decoder, img_tensor, vocab, device=args.device)
    print("Predicted caption:", caption)
