# coding: utf-8
from torch.utils.data import Dataset
import numpy as np
import scipy.signal as signal
import torch
import torchvision.transforms as transforms

'''for pre-training'''
class AddGaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, signal):
        noise = torch.randn_like(signal) * self.std + self.mean
        noisy_signal = signal + noise
        return noisy_signal


# 振幅缩放
class ScaleAmplitude(object):
    def __init__(self, scale_factor=1.5):
        self.scale_factor = scale_factor

    def __call__(self, signal):
        scaled_signal = signal * self.scale_factor
        return scaled_signal


# Y轴翻转
class FlipYAxis(object):
    def __call__(self, signal):
        flipped_signal = -signal
        return flipped_signal


# X轴翻转
class FlipXAxis(object):
    def __call__(self, signal):
        flipped_signal = torch.flip(signal, [0])
        return flipped_signal

class CMSCDataset(Dataset):
    def __init__(self, txt_path):
        fh = open(txt_path, 'r')
        signals = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if words[0].split('/')[2]=='Dataset':
                signals.append((words[0], int(words[1])))
            else:
                signals.append(('../../Dataset/' + words[0], int(words[1])))

        self.signals = signals

    def __getitem__(self, index):
        fn, label = self.signals[index]
        raw_signal = np.float32(np.loadtxt(fn))
        original_freq = 250
        target_freq = 96
        T = 1 / original_freq
        new_T = 1 / target_freq
        duration = T * len(raw_signal)
        resampled_signal = signal.resample(raw_signal, int(duration / new_T))

        # 将信号分成两个非重叠段
        mid_point = len(resampled_signal) // 2
        sig_1 = torch.Tensor(resampled_signal[:mid_point])
        sig_2 = torch.Tensor(resampled_signal[mid_point:])

        # 确保每段的长度一致，如果需要，可以在此处进行裁剪或填充
        max_len = min(len(sig_1), len(sig_2))
        sig_1 = sig_1[:max_len].unsqueeze(0)  # 添加通道维度
        sig_2 = sig_2[:max_len].unsqueeze(0)  # 添加通道维度

        return sig_1, sig_2

    def __len__(self):
        return len(self.signals)


'''for generalization experiment'''
class gerMyDataset(Dataset):
    def __init__(self, txt_path):
        fh = open(txt_path, 'r')
        signals = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if words[0].split('/')[2]=='Dataset':
                signals.append((words[0], int(words[1])))
            else:
                signals.append(('../../Dataset/' + words[0], int(words[1])))

        self.signals = signals

    def __getitem__(self, index):
        fn, label = self.signals[index]
        raw_signal = np.float32(np.loadtxt(fn))
        original_freq = 250
        target_freq = 48
        T = 1 / original_freq
        new_T = 1 / target_freq
        duration = T * len(raw_signal)
        new_signal = signal.resample(raw_signal, int(duration / new_T))
        new_signal = torch.from_numpy(new_signal)
        new_signal = new_signal.unsqueeze(0)

        return new_signal, label

    def __len__(self):
        return len(self.signals)


'''for personalization experiment'''
class perMyDataset(Dataset):
    def __init__(self, person, dataset, txt_path, mode, transform=None):
        fh = open(txt_path, 'r')
        inputs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            signalPath = '../../Dataset/' + words[0].split('./')[-1]
            inputs.append((signalPath, int(words[1])))

        self.inputs = inputs
        self.transform = transform

    def __getitem__(self, index):
        signalPath, label = self.inputs[index]

        raw_signal = np.float32(np.loadtxt(signalPath))
        original_freq = 250
        target_freq = 48
        T = 1 / original_freq
        new_T = 1 / target_freq
        duration = T * len(raw_signal)
        new_signal = signal.resample(raw_signal, int(duration / new_T))
        new_signal = torch.from_numpy(new_signal)
        new_signal = new_signal.unsqueeze(0)

        return new_signal, label

    def __len__(self):
        return len(self.inputs)
