import torch.nn.functional as F
from driver.MST import *
import torch.optim.lr_scheduler
from driver.Layer import *
import numpy as np

class DomainClassifier(object):
    def __init__(self, model):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, lstm_hidden, masks):
        score = self.model.forward(lstm_hidden, masks)
        self.score = score

    def compute_loss(self, true_labels):
        loss = F.cross_entropy(self.score, true_labels)
        return loss

    def compute_accuray(self, true_labels):
        total = true_labels.size()[0]
        pred_labels = self.score.data.max(1)[1].cpu()
        correct = pred_labels.eq(true_labels).cpu().sum().item()
        return correct, total
