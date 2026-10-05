import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

## ArcFace head: adds an angular margin so classes are pushed further apart
class ArcFaceHead(nn.Module):
    def __init__(self, in_dim, num_classes, s=ARC_SCALE, m=ARC_MARGIN):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels=None):
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).float()
        if labels is None:               # inference: no margin
            return cos * self.s
        sin = torch.sqrt((1.0 - cos ** 2).clamp(1e-7, 1.0))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = F.one_hot(labels, cos.size(1)).float()
        return (one_hot * phi + (1 - one_hot) * cos) * self.s

## Full network: DINOv2 -> embedding layer -> ArcFace
class LeafNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained)
        self.embed = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, EMBED_DIM),
            nn.BatchNorm1d(EMBED_DIM))
        self.head = ArcFaceHead(EMBED_DIM, num_classes)
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[-UNFREEZE_LAST_N:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    def forward(self, x, labels=None):
        feats = self.backbone(x)
        return self.head(self.embed(feats), labels)