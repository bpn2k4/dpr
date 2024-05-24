import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupScheduler(_LRScheduler):
  def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    step = self.last_epoch + 1

    if step < self.warmup_steps:
      warmup_factor = float(step) / float(max(1, self.warmup_steps))
      return [base_lr * warmup_factor for base_lr in self.base_lrs]
    else:
      decay_factor = max(0.0, float(self.total_steps - step) /
                         float(max(1, self.total_steps - self.warmup_steps)))
      return [base_lr * decay_factor for base_lr in self.base_lrs]
