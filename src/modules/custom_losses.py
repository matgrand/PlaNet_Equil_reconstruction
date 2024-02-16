import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import lightning as L
from torchmetrics import Metric



class MyAccuracy(Metric):
    '''
    Example of custom loss using torchmetrics
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
    

class UserCrossEntropyLoss(L.LightningModule):
    '''
    Example of custom loss using L.LightningModule
    '''
    def __init__(self):
        super(UserCrossEntropyLoss,self).__init__()
        self.loss = CrossEntropyLoss()

    def forward(
            self,
            pred : Tensor,
            y: Tensor):
        return self.loss(pred,y)













