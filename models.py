import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # remove avgpool & fc
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        features = self.resnet(images)  # (B, 2048, H/32, W/32)
        features = self.adaptive_pool(features)  # (B,2048, encoded_image_size, encoded_image_size)
        features = features.permute(0,2,3,1)  # (B, enc_size, enc_size, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (B, num_pixels, 2048)
        return features

    def fine_tune(self, fine_tune=False):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # optionally unfreeze some layers
        if fine_tune:
            for c in list(self.resnet.children())[7:]:
                for p in c.parameters():
                    p.requires_grad = True

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (B, num_pixels, encoder_dim)
        # decoder_hidden: (B, decoder_dim)
        att1 = self.encoder_att(encoder_out)  # (B, num_pixels, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B,1,att_dim)
        att = torch.tanh(att1 + att2)  # (B, num_pixels, att_dim)
        e = self.full_att(att).squeeze(2)  # (B, num_pixels)
        alpha = F.softmax(e, dim=1)  # attention weights
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (B, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Sort input by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (B, max_len, embed_dim)

        h, c = self.init_hidden_state(encoder_out)  # (B, dec_dim)
        max_len = max(caption_lengths).item()
        preds = torch.zeros(batch_size, max_len, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_len, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_len):
            batch_size_t = sum([l > t for l in caption_lengths])
            context, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            emb_t = embeddings[:batch_size_t, t, :]
            input_lstm = torch.cat([emb_t, context], dim=1)
            h_t, c_t = self.decode_step(input_lstm, (h[:batch_size_t], c[:batch_size_t]))
            preds[:batch_size_t, t, :] = self.fc(self.dropout(h_t))
            alphas[:batch_size_t, t, :] = alpha
            h[:batch_size_t] = h_t
            c[:batch_size_t] = c_t

        return preds, encoded_captions, caption_lengths, alphas
