import numpy as np
import os
import torchaudio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_desc_file(_desc_file, __class_labels):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        words = words[:len(words) - 1]  # Removing 'mixture a001 (audio file name)' after the class label
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append(__class_labels[words[-1]])
    return _desc_dict


class MelData(Dataset):
    def __init__(self, root, class_labels, sample_rate=32000, n_mels=64, n_fft=1024, hop_length=320):
        self.root = root
        self.class_labels = class_labels
        # Spectrogram parameters (the same as librosa.stft)
        self.sample_rate = sample_rate
        # sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        win_length = n_fft
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        is_mono = True

        # Mel parameters (the same as librosa.feature.melspectrogram)
        self.n_mels = n_mels
        fmin = 50
        fmax = 14000

        # Power to db parameters (the same as default settings of librosa.power_to_db
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.mel_tensor, self.label_tensor = None, None
        self.mel_list, self.label_list = [], []

        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                            n_fft=self.n_fft,
                                                            hop_length=self.hop_length,
                                                            n_mels=self.n_mels,
                                                            f_min=50, f_max=14000)

        meta = os.path.join(self.root + 'meta.txt')
        meta_dict = load_desc_file(meta, self.class_labels)

        for audio_file in tqdm(os.listdir(os.path.join(root + 'audio/'))):
            audio_path = os.path.join(root + 'audio/' + audio_file)
            #print(audio_path)
            y, sr = torchaudio.load(audio_path)
            # make it mono
            y = torch.mean(y, dim=0)
            if sr != self.sample_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)
            mels = self.melspec(y)
            mels = torch.transpose(mels, 0, 1)
            mels = torch.log(mels + torch.finfo(torch.float32).eps)
            mels = mels[None, None, :, :]

            label = torch.tensor(meta_dict[audio_path.split('/')[-1]])

            if self.mel_tensor is None:
                self.mel_tensor, self.label_tensor = mels, label
            else:
                self.mel_tensor, self.label_tensor = (torch.concat((self.mel_tensor, mels), dim=0),
                                                      torch.concat((self.label_tensor, label), dim=0))
            self.mel_list.append(mels)
            self.label_list.append(label)

        print("Total audio files =", len(self.mel_list))

