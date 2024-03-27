import torch
from torch import nn

# Define Huber loss function
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = torch.abs(y_true - y_pred)
        quadratic_part = torch.clamp(error, max=self.delta)
        linear_part = error - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.delta * linear_part
        return loss.mean()

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, y_pred, y_true):
        dot_product = torch.sum(y_pred * y_true)
        norm_A = torch.norm(y_pred)
        norm_B = torch.norm(y_true)
        similarity = dot_product / (norm_A * norm_B)
        return 1 - similarity
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        log_cosh = torch.log(torch.cosh(error))
        loss = torch.mean(log_cosh)
        return loss