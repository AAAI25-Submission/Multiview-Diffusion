import torch
import torch.nn as nn
import torch.nn.functional as F

class classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(classifier, self).__init__()
        self.signal_layer = nn.Sequential(
            nn.Conv1d(12, 3, 6, 1, 1),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.lorenz_layer = nn.Sequential(
            nn.Conv2d(12, 3, 6, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.frequency_layer = nn.Sequential(
            nn.Conv2d(12, 3, 6, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3618, 1024),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, signal, lorenz, frequency):
        signal = self.signal_layer(signal)
        lorenz = self.lorenz_layer(lorenz)
        frequency = self.frequency_layer(frequency)

        com = torch.cat([signal, lorenz, frequency], dim=1)

        x = self.fc1(com)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
