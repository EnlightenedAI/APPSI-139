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
        # print('bce:',BCE_loss)
        # print('pt:',pt)
        # m = nn.Sigmoid()
        # ot=m(inputs)
        # ot[(targets == 1) & (ot >= 0.5) ] = 0
        # ot[(targets == 0) & (ot<0.5)] = 0
        # print(ot)


        # print('ot',ot)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(torch.sum(F_loss,dim=1),dim=0)
            # print(torch.mean(F_loss[ot!=0]))
            # return torch.mean(F_loss[ot!=0])
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
        # print(logit)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        # print(loss)
        # print(1 - logit)
        loss = loss * (1 - logit) ** self.gamma # focal loss
        # print(loss)
        return loss.sum()/y.sum()

#
# # # 示例
# inputs = torch.randn(3, 5).requires_grad_()
# targets = torch.empty(3,5).random_(2)
#
# # print(inputs)
# # print(targets)
# # #
# focal_loss = FocalLoss()
# loss = focal_loss(inputs, targets)
# # print(loss)
# #
# focal_loss = FocalLoss(reduction='mean')
# loss = focal_loss(inputs, targets)
# print(loss)
#
# focal_loss = FocalLoss(reduction='none')
# loss = focal_loss(inputs, targets)
# print(loss)

