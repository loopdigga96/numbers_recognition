import os
import json
import argparse

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pd

from utils import (ASRModel, greedy_decode, AudioDataset, TextTransform, data_processing)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        hparams = json.load(f)

    summary_writer = SummaryWriter()
    torch.manual_seed(hparams['seed'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    audio_transforms = torchaudio.transforms.MelSpectrogram()
    text_transform = TextTransform()

    print('Loading model')
    model = ASRModel(hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
                     hparams['n_class'], hparams['n_feats'], hparams['stride'],
                     hparams['dropout']).to(device)

    if args.checkpoint:
        print(f'Loading checkpoint {args.checkpoint}')
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    # load dataset
    data_df = pd.read_csv(args.data_path)
    data_folder = os.path.dirname(args.data_path)

    dataset = AudioDataset(data_folder, data_df)

    print('Setup loaders')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    loader = DataLoader(dataset=dataset,
                        batch_size=hparams['batch_size'],
                        shuffle=False,
                        collate_fn=lambda x: data_processing(x, text_transform, audio_transforms),
                        **kwargs)

    blank_id = len(text_transform)
    preds = []

    print('Making prediction')
    data_len = len(loader)
    for i, batch in enumerate(loader):
        print(f'{i}/{data_len}')
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        decoded_preds, decoded_targets = greedy_decode(output.transpose(0, 1), labels,
                                                       label_lengths, text_transform)
        preds.extend(decoded_preds)

    submission = pd.DataFrame({'number': preds, 'path': data_df['path'].tolist()})
    submission.to_csv('submission.csv')
