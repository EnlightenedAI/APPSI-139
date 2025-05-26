import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 获取预测概率
        prob = torch.softmax(inputs, dim=1)
        prob_true_class = torch.gather(prob, dim=1, index=targets.unsqueeze(1))

        # 计算 focal loss
        focal_weight = (1 - prob_true_class) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss

        # 对损失进行归约处理
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# 示例
# inputs = torch.randn(16, 4)  # 假设有16个样本，4个类别
#
# print(inputs)
# targets = torch.randint(0, 4, (16, ))  # 随机生成目标类别
#
# print(targets)
#
# focal_loss = FocalLoss()
# loss = focal_loss(inputs, targets)
# print('Focal Loss:', loss.item())
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# import torchvision
# import torchvision.transforms as F

# # from IPython.display import display
# class Focal_Loss(nn.Module):

#     def __init__(self, alpha=0.25, gamma=2,reduction='none'):
#         super(Focal_Loss, self).__init__()
#         self.gamma = gamma
#         self.alpha=alpha
#         weight = torch.tensor([1-alpha, alpha]).to('cuda')
#         # self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)
#         # self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         alphas = [self.alpha if ta == 1 else 1-self.alpha for ta in target]
        
#         alphas = torch.tensor(alphas, dtype=torch.float32, device=target.device)
#         p = torch.exp(-logp)
#         # print(p)
#         loss = (1 - p) ** self.gamma * logp *self.alpha
#         # print(loss)
#         return loss.mean()
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Convert the target to one-hot encoding
        # target_one_hot = F.one_hot(target, num_classes=input.size(-1)).to(torch.float32)
        # target_one_hot = F.one_hot(target, num_classes=num_classes).to(torch.float32)
        # target_one_hot = F.one_hot(target, num_classes=2).float()
        # print(target_one_hot)
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(input, target.long(), reduction='none')
        
        # Get the probabilities
        p_t = torch.exp(-ce_loss)
        
        # Compute the focal loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        # print(ce_loss)
        # Apply alpha weighting
        alpha_t = [self.alpha if ta == 1 else 1-self.alpha for ta in target]
        # alpha_t = [1-self.alpha,self.alpha]
        # print(alpha_t,target)
        alpha_t = torch.tensor(alpha_t, dtype=torch.float32, device=target.device)
        focal_loss = focal_loss* alpha_t 
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        # elif self.reduction == 'sum':
        #     return focal_loss.sum()
        # else:
        #     return focal_loss

# Usage example
# model = YourModel()
# criterion = FocalLoss(alpha=0.25, gamma=2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# output = model(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()
class Focal_Loss_multi(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(Focal_Loss_multi, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Convert the target to one-hot encoding
        # target_one_hot = F.one_hot(target, num_classes=input.size(-1)).to(torch.float32)
        # target_one_hot = F.one_hot(target, num_classes=num_classes).to(torch.float32)
        # Calculate cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction='none')
        
        # Get the probabilities
        p_t = torch.exp(-ce_loss)
        
        # Compute the focal loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # Apply alpha weighting
        # alpha_t = [self.alpha if ta == 1 else 1-self.alpha for ta in target]
        # alpha_t
        # print(alpha_t,target)
        # alpha_t = torch.tensor(alpha_t, dtype=torch.float32, device=target.device)
        # focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        # elif self.reduction == 'sum':
        #     return focal_loss.sum()
        # else:
        #     return focal_loss