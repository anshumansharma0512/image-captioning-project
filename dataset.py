import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from vocab import Vocabulary
import numpy as np

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab=None, transform=None, max_len=30):
        """
        root_dir: dir with images (e.g., Flickr8k_images)
        captions_file: Flickr8k.token.txt path (format: image#0\tcaption)
        """
        self.root_dir = root_dir
        self.captions = []  # list of (img_name, caption)
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line)==0:
                    continue
                img_caps = line.split('\t')
                if len(img_caps) != 2:
                    continue
                img_name = img_caps[0].split('#')[0]
                caption = img_caps[1]
                self.captions.append((img_name, caption))

        self.vocab = vocab
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name, caption = self.captions[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.vocab is not None:
            numericalized = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
            if len(numericalized) > self.max_len:
                numericalized = numericalized[:self.max_len]
            length = len(numericalized)
            return image, torch.tensor(numericalized), length, img_name, caption

        return image, caption, img_name
