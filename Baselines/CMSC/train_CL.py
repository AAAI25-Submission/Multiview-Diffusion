import sys
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cl_load_data import CMSCDataset
from Baselines.Models.SCLModel import CLModel
import numpy as np
import random
import argparse
import torch.nn.functional as F

def args_parse():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--cuda', type=int, default=0, help="device")
    parser.add_argument('--formal_epochs', type=int, default=100)
    parser.add_argument('--formal_lr', type=float, default=1e-2)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='CPSC2018')
    parser.add_argument('--model', type=str, default='CMSC')
    return parser.parse_args()

args = args_parse()

def set_rand_seed(seed=args.random_seed):
    # print("Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_rand_seed(args.random_seed)
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")


class CMSCLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(CMSCLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        sim_12 = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2) / self.temperature
        sim_21 = F.cosine_similarity(z2.unsqueeze(1), z1.unsqueeze(0), dim=2) / self.temperature

        batch_size = z1.size(0)
        positive_mask = torch.eye(batch_size, dtype=torch.float32, device=z1.device)
        negative_mask = 1.0 - positive_mask

        exp_sim_12 = torch.exp(sim_12) * negative_mask
        exp_sim_21 = torch.exp(sim_21) * negative_mask

        log_prob_12 = sim_12 - torch.log(exp_sim_12.sum(dim=1, keepdim=True))
        log_prob_21 = sim_21 - torch.log(exp_sim_21.sum(dim=1, keepdim=True))

        loss_12 = - (log_prob_12 * positive_mask).sum(dim=1)
        loss_21 = - (log_prob_21 * positive_mask).sum(dim=1)

        loss = (loss_12 + loss_21).mean()

        return loss


class LinearCritic(nn.Module):
    def __init__(self, latent_dim, temperature=1.):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.projection_dim = 128
        self.w1 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(latent_dim, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
        self.cossim = nn.CosineSimilarity(dim=-1)

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))

    def forward(self, h1, h2):
        z1, z2 = self.project(h1), self.project(h2)
        sim11 = self.cossim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(z2.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(z1.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        return raw_scores, targets

def nt_xent_loss(raw_scores, targets):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(raw_scores, targets)
    return loss

def train_cmsc(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for i, (x_i, x_j) in enumerate(train_loader):
        x_i, x_j = x_i.to(device), x_j.to(device)
        optimizer.zero_grad()
        z_i, z_j = model(x_i), model(x_j)
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 100 == 0:
            print('train_step:{},loss:{}'.format(i, loss.item()))
    return total_loss / len(train_loader)

# 设置超参数
dataset = args.dataset
batch_size = 64
epochs = args.formal_epochs
lr = args.formal_lr
temperature = args.temperature

model = CLModel().to(device)
for param in model.projector.parameters():
    param.requires_grad = False
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min=0, last_epoch=-1)

train_dataset = CMSCDataset('../../Dataset/data_' + dataset + '_segments/person_all_idx.txt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
criterion = CMSCLoss(temperature=temperature)
# 训练模型
for epoch in range(epochs):
    loss = train_cmsc(model, train_loader, optimizer, criterion, 'cuda')
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss}')
torch.save(model, './savedModels/CMSC_{}.pth'.format(args.dataset))