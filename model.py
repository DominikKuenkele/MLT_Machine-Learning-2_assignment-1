import torch.nn as nn
import torch
import sys
import math
import numpy as np
from torch import optim


class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_dim, num_layers, padding_idx, glove_vectors=None) -> None:
        super(CaptionEncoder, self).__init__()
        if embedding_dim == -1:
            embedding_dim = glove_vectors.dim
            self.embeddings = nn.Embedding.from_pretrained(glove_vectors.vectors, freeze=True)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, out_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.15)

        self.out_dimension = out_dim * 2 * num_layers

    def forward(self, caption_batch):
        embeddings = self.embeddings(caption_batch)
        dropped_out = self.dropout(embeddings)
        _, (h_n, _) = self.rnn(dropped_out)


        return self.dropout(h_n)

class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super(ImageEncoder, self).__init__()
        IMAGE_SIZE = 100
        KERNEL_SIZE = 3
        CHANNELS = 3

        POOLING_SIZE = 4

        self.image_encoder = nn.Sequential(
            nn.Conv2d(CHANNELS, CHANNELS, KERNEL_SIZE),
            nn.BatchNorm2d(CHANNELS),
            nn.MaxPool2d(POOLING_SIZE),
            nn.Tanh()
        )

        conv2d_out_dimension = IMAGE_SIZE - KERNEL_SIZE + 1
        max_pool2d_out_dimension = (conv2d_out_dimension - POOLING_SIZE + 2) / POOLING_SIZE
        self.out_dimension = int(max_pool2d_out_dimension ** 2 * CHANNELS)

    def forward(self, image_batch):
        return self.image_encoder(image_batch)


class CaptionEvaluator(nn.Module):
    def __init__(self, caption_encoder, image_encoder, hidden_size) -> None:
        super(CaptionEvaluator, self).__init__()

        self.caption_encoder = caption_encoder
        self.image_encoder = image_encoder

        self.classifier = nn.Sequential(
            nn.Linear(caption_encoder.out_dimension + image_encoder.out_dimension, hidden_size),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.Tanh(),
            nn.Linear(int(hidden_size/2), 1),
            nn.Sigmoid()
        )

    def forward(self, image_batch, caption_batch):
        encoded_caption = self.caption_encoder(caption_batch)
        flattened_caption = torch.flatten(encoded_caption.permute(1,0,2), 1)

        encoded_image = self.image_encoder(image_batch)
        flattened_image = torch.flatten(encoded_image, 1)

        concatenated_encoding = torch.cat((flattened_caption, flattened_image), 1)
        classified = self.classifier(concatenated_encoding)
        
        return classified


def train(caption_evaluator, train_dataloader, hyperparameters, device):
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(caption_evaluator.parameters(), lr=hyperparameters['learning_rate'])

    print(f'{hyperparameters["epochs"]} EPOCHS - {math.floor(len(train_dataloader.dataset) / train_dataloader.batch_size)} BATCHES PER EPOCH')

    glove_training_epoch = int(0.5 * hyperparameters['epochs']) 

    for epoch in range(hyperparameters['epochs']):
        if epoch == glove_training_epoch:
            caption_evaluator.caption_encoder.embeddings.weight.requires_grad = True

        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            image_batch = batch['image'].to(device)
            if hyperparameters['embedding_dim'] == -1:
                caption_batch = batch['glove_encoded_caption'].to(device)
            else:
                caption_batch = batch['encoded_caption'].to(device)

            output = caption_evaluator(image_batch, caption_batch)
            loss = loss_function(output, batch['class'].view(len(image_batch),1).to(device))
            total_loss += loss.item()

            # print average loss for the epoch
            sys.stdout.write(f'\repoch {epoch}, batch {i}: {np.round(total_loss / (i + 1), 4)}')

            # compute gradients
            loss.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()
        print()