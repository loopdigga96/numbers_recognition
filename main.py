import os
import json
import argparse
from typing import Callable

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (word_error_rate, IterMeter, Metrics, ASRModel, greedy_decode, AudioDataset,
                   TextTransform, data_processing)


def train(model: nn.Module, device: torch.device, train_loader: DataLoader, criterion: nn.Module,
          optimizer: nn.Module, scheduler, epoch: int, iter_meter, tb_writer: SummaryWriter,
          log_every=20) -> Metrics:
    model.train()
    data_len = len(train_loader)
    epoch_loss = []
    print('Training')

    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()
        loss_scalar = loss.item()

        optimizer.step()
        if scheduler:
            scheduler.step()
        iter_meter.step()

        if batch_idx % log_every == 0 or batch_idx == data_len:
            print(f'Train Epoch: {epoch} \t batch: {batch_idx}/{data_len}')
            print(f'Loss: {loss_scalar}')

        epoch_loss.append(loss_scalar)
        tb_writer.add_scalar('batch_loss', loss_scalar, iter_meter.get())

    return Metrics(loss=np.mean(epoch_loss))


def test(model: nn.Module, device: torch.device, test_loader: DataLoader, criterion: nn.Module,
         text_transform: Callable, log_every=40):
    print('Evaluating...')
    model.eval()
    test_cer, test_wer, test_loss = [], [], []
    data_len = len(test_loader)

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss.append(loss.item())

            decoded_preds, decoded_targets = greedy_decode(output.transpose(0, 1), labels, label_lengths,
                                                           text_transform)
            test_cer.append(word_error_rate(decoded_targets, decoded_preds, use_cer=True))
            test_wer.append(word_error_rate(decoded_targets, decoded_preds))

            if i % log_every == 0:
                print(f'{i}/{data_len}')
                print(f'Test WER: {test_wer[-1]}; CER: {test_cer[-1]}')

                for p, t in zip(decoded_preds, decoded_targets):
                    print(f'Prediction: [{p}]\t Ground Truth: [{t}]')

    avg_cer = np.mean(test_cer)
    avg_wer = np.mean(test_wer)
    avg_loss = np.mean(test_loss)

    print(f'Test set: Average loss: {avg_loss}, Average CER: {avg_cer} Average WER: {avg_wer}')
    return Metrics(loss=avg_loss, cer=avg_cer, wer=avg_wer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    summary_writer = SummaryWriter()
    torch.manual_seed(config['seed'])
    epochs = config['epochs']

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
    text_transform = TextTransform()

    print('Loading model')
    model = ASRModel(config['n_cnn_layers'], config['n_rnn_layers'], config['rnn_dim'],
                     config['n_class'], config['n_feats'], config['stride'],
                     config['dropout']).to(device)

    # load dataset
    data_df = pd.read_csv(os.path.join(config['data_path'], 'train.csv'))
    train_df, val_df = train_test_split(data_df, test_size=config['val_fraction'])

    train_dataset = AudioDataset(config['data_path'], train_df)
    val_dataset = AudioDataset(config['data_path'], val_df)

    print('Setup loaders')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              collate_fn=lambda x: data_processing(x, text_transform, train_audio_transforms),
                              **kwargs)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            collate_fn=lambda x: data_processing(x, text_transform, valid_audio_transforms),
                            **kwargs)

    optimizer = optim.AdamW(model.parameters(), config['learning_rate'])
    blank_id = len(text_transform)
    criterion = nn.CTCLoss(blank=blank_id).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=config['epochs'],
                                              anneal_strategy='linear')
    all_metrics = []
    best_metric = None
    iter_meter = IterMeter()
    for epoch in range(epochs):
        train_metrics = train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter,
                              summary_writer)
        test_metrics = test(model, device, val_loader, criterion, text_transform)

        if best_metric is not None and test_metrics.loss < best_metric.loss:
            print('Found improvement')
            best_metric = test_metrics
            torch.save(model.state_dict(), f'./best_model_{epoch}.pt')
        else:
            torch.save(model.state_dict(), f'./best_model_{epoch}.pt')
            best_metric = test_metrics

        all_metrics.append(test_metrics)

        summary_writer.add_scalars('Loss', {'train': train_metrics.loss,
                                            'val': test_metrics.loss}, epoch)
        summary_writer.add_scalars('Metric_val', {'WER': test_metrics.wer,
                                                  'CER': test_metrics.cer}, epoch)
