import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')
        self.size_average = size_average

    def dice_loss(self, input, target):
        smooth = 1.
        input = torch.sigmoid(input)
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
        
        bce = self.bce_loss(input, target_onehot)
        
        dice = 0
        for i in range(input.shape[1]): # Loop over classes
            dice += self.dice_loss(input[:, i], target_onehot[:, i])
        dice /= input.shape[1]
        
        return 0.8 * bce + 0.2 * dice