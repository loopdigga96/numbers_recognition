import os
from typing import List, Callable, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        0 2
        1 3
        2 4
        3 5
        4 6
        5 7
        6 8
        7 9
        8 10
        9 11
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text: str):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string)

    def __len__(self):
        return len(self.char_map)


class AudioDataset(Dataset):
    def __init__(self, data_path: str, df: pd.DataFrame, audio_col='path', utter_col='number'):
        self.df = df
        self.audio_col = audio_col
        self.utter_col = utter_col
        self.data_path = data_path

    def __getitem__(self, item: int) -> List[torch.tensor]:
        row = self.df.iloc[item]
        file_audio = row[self.audio_col]
        utterance = str(row[self.utter_col] if self.utter_col in row else '')
        cwd = os.getcwd()
        waveform, sample_rate = torchaudio.load(os.path.join(cwd, self.data_path, file_audio))
        return waveform, sample_rate, utterance

    def __len__(self) -> int:
        return len(self.df)


def data_processing(batch: List[torch.tensor], text_transform: Callable,
                    audio_transform: Callable) -> List[torch.tensor]:
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, sample_rate, utterance in batch:
        spec = audio_transform(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def greedy_decode(output: torch.tensor, labels: str, label_lengths: List[int], text_transform: Callable,
                  blank_label=12, collapse_repeated=True) -> Tuple[str, str]:
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets
