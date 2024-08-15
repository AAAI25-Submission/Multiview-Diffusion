import torch
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.sd_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm1d(4, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size = 2, stride=1),
            nn.Dropout(0.1),
        )
        self.sd_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm1d(16, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size = 2, stride=1),
            nn.Dropout(0.1),
        )
        self.sd_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm1d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size = 2, stride=1),
            nn.Dropout(0.1),
        )
        self.fc1 = nn.Linear(448, 128)

    def forward(self, x):
        x = self.sd_layer1(x)
        x = self.sd_layer2(x)
        x = self.sd_layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

# 定义SimCLR模型
class CLModel(nn.Module):
    def __init__(self):
        super(CLModel, self).__init__()
        self.encoder = BaseCNN()
        self.projector = nn.Sequential(
            nn.Linear(128, 128)
        )

    def forward(self, x):
        features = self.encoder(x.float())
        features = features.view(features.size(0), -1)
        embeddings = self.projector(features)
        return embeddings

class downStreamClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(downStreamClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        return x