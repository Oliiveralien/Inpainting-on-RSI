import torch
import numpy as np
import torch.nn.functional as F

class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1.-pos)) + torch.mean(F.relu(1.+neg)))


class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)


class PerceptualLoss(torch.nn.Module):
    """
    Use vgg16 for perceptual loss, compute the feature distance
    """
    def __init__(self, weight=1, layers=[0,9,13,17], feat_extractors=None):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        self.feat_extractors = feat_extractors
        self.layers = layers

    def forward(self, imgs, recon_imgs):
        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))
        feats = self.feat_extractors(imgs, self.layers)
        recon_feats = self.feat_extractors(recon_imgs, self.layers)
        loss = 0
        for feat, recon_feat in zip(feats, recon_feats):
            loss = loss + torch.mean(torch.abs(feat - recon_feat))
        return self.weight*loss

class StyleLoss(torch.nn.Module):
    """
    Use vgg16 for style loss, compute the feature distance
    """
    def __init__(self, weight=1, layers=[0,9,13,17], feat_extractors=None):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.feat_extractors = feat_extractors
        self.layers = layers
    def gram(self, x):
        gram_x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        return torch.bmm(gram_x, torch.transpose(gram_x, 1, 2))

    def forward(self, imgs, recon_imgs):
        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))
        feats = self.feat_extractors(imgs, self.layers)
        recon_feats = self.feat_extractors(recon_imgs, self.layers)
        loss = 0
        for feat, recon_feat in zip(feats, recon_feats):
            loss = loss + torch.mean(torch.abs(self.gram(feat) - self.gram(recon_feat))) / (feat.size(2) * feat.size(3) )
        return self.weight*loss

class ReconLoss(torch.nn.Module):
    """
    l1/l2 loss

    """
    def __init__(self, chole_alpha, cunhole_alpha, rhole_alpha, runhole_alpha):
        super(ReconLoss, self).__init__()
        self.chole_alpha = chole_alpha
        self.cunhole_alpha = cunhole_alpha
        self.rhole_alpha = rhole_alpha
        self.runhole_alpha = runhole_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        masks_viewed = masks.view(masks.size(0), -1)
        # L1 
        return self.rhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))  + \
                self.runhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))  + \
                self.chole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))   + \
                self.cunhole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))
        # L2
        # return self.rhole_alpha*torch.mean((imgs - recon_imgs) ** 2 * masks / masks_viewed.mean(1).view(-1,1,1,1))  + \
        #         self.runhole_alpha*torch.mean((imgs - recon_imgs) ** 2 * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))  + \
        #         self.chole_alpha*torch.mean((imgs - recon_imgs) ** 2 * masks / masks_viewed.mean(1).view(-1,1,1,1))   + \
        #         self.cunhole_alpha*torch.mean((imgs - recon_imgs) ** 2 * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))