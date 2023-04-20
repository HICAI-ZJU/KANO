
from .loss_computer import NCESoftmaxLoss
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
logger = logging.getLogger()


class ContrastiveLoss(nn.Module):
    def __init__(self, loss_computer: str, temperature: float, args) -> None:
        super().__init__()
        self.device = args.device

        if loss_computer == 'nce_softmax':
            self.loss_computer = NCESoftmaxLoss(self.device)
        else:
            raise NotImplementedError(f"Loss Computer {loss_computer} not Support!")
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # SimCSE
        batch_size = z_i.size(0)

        emb = F.normalize(torch.cat([z_i, z_j]))

        similarity = torch.matmul(emb, emb.t()) - torch.eye(batch_size*2).to(self.device) * 1e12
        similarity = similarity * 20
        loss = self.loss_computer(similarity)
        
        return loss

class FlatNCE(nn.Module):
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__()
    
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        # logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(positives.shape[0], dtype=torch.long)
        logits = (negatives - positives)/self.temperature
        clogits = torch.logsumexp(logits, dim=1, keepdim=True)
        loss = torch.exp(clogits - clogits.detach())




# _, features = self.model(images)
# logits, labels = self.flat_loss(features)
# v = torch.logsumexp(logits, dim=1, keepdim=True) #(512,1)
# loss_vec = torch.exp(v-v.detach())

# assert loss_vec.shape == (len(logits),1)
# dummy_logits = torch.cat([torch.zeros(logits.size(0),1).to(self.args.device), logits],1)
# loss = loss_vec.mean()-1 + self.criterion(logits, labels).detach() 
