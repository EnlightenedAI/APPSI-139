import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        prob = torch.softmax(inputs, dim=1)
        prob_true_class = torch.gather(prob, dim=1, index=targets.unsqueeze(1))
        focal_weight = (1 - prob_true_class) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        alpha_t = [self.alpha if ta == 1 else 1-self.alpha for ta in target]
        alpha_t = torch.tensor(alpha_t, dtype=torch.float32, device=target.device)
        focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class Focal_Loss_multi(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(Focal_Loss_multi, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss