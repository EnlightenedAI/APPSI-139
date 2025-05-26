import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(torch.sum(F_loss,dim=1),dim=0)
        elif self.reduction == 'sum':
            return torch.sum(F_loss,dim=0)
        else:
            return F_loss
class FocalLoss_1(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss_1, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = target
        m=nn.Sigmoid()
        logit =m(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma
        return loss.sum()/y.sum()

