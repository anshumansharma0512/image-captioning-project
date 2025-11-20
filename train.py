import os
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import FlickrDataset
from vocab import Vocabulary
from models import EncoderCNN, DecoderWithAttention
from collections import defaultdict
import math
from tqdm import tqdm
import random

def collate_fn(data):
    # data: list of tuples (image, caption_tensor, length, img_name, raw_caption)
    data.sort(key=lambda x: x[2], reverse=True)
    images, captions, lengths, names, raw = zip(*data)
    images = torch.stack(images, 0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap
    return images, padded, lengths, names, raw

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Read all captions
    captions_file = os.path.join(args.data_dir, "Flickr8k.token.txt")
    images_dir = os.path.join(args.data_dir, "Flickr8k_images")
    # Read captions
    sentences = []
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split('\t')
            if len(parts)!=2: continue
            sentences.append(parts[1])

    vocab = Vocabulary(freq_threshold=args.freq_threshold, max_size=args.vocab_size)
    vocab.build_vocabulary(sentences)
    print(f"Vocab size: {len(vocab.stoi)}")

    dataset = FlickrDataset(images_dir, captions_file, vocab=vocab)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(attention_dim=512, embed_dim=args.embed_dim, decoder_dim=args.decoder_dim, vocab_size=len(vocab.stoi)).to(device)
    params = list(decoder.parameters()) + list(filter(lambda p: p.requires_grad, encoder.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

    best_loss = float('inf')
    global_step = 0
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, caps, lengths, names, raw in pbar:
            images = images.to(device)
            caps = caps.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            encoder_out = encoder(images)  # (B, num_pixels, enc_dim)
            preds, targets, lengths_sorted, alphas = decoder(encoder_out, caps, lengths)
            # flatten predictions and targets
            # preds: (B, max_len, vocab_size)
            preds = preds.view(-1, preds.size(-1))
            targets = targets[:, :preds.size(0)//preds.size(1)].contiguous() if False else targets  # workaround but we can flatten properly
            targets = targets.contiguous().view(-1)
            loss = criterion(preds, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)
            optimizer.step()
            running_loss += loss.item()
            global_step += 1
            if global_step % args.log_step == 0:
                pbar.set_postfix({'loss': running_loss / args.log_step})
                running_loss = 0.0

        # Save checkpoint every epoch
        ckpt = {
            'epoch': epoch+1,
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'vocab': vocab.stoi
        }
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch+1}.pth'))
        print(f"Saved checkpoint epoch {epoch+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./Flickr8k', help='Flickr8k dataset dir')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--freq_threshold', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_step', type=int, default=20)
    args = parser.parse_args()
    train(args)
