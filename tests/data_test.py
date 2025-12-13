import torch 
from data_utils.metrics import binary_iou

def test_binary_iou():
    y_hat = torch.tensor([[[1,1,0]]])
    y = torch.tensor([[[1,1,1]]])

    acc = binary_iou(torch.unsqueeze(y_hat, 0), torch.unsqueeze(y, 0))

    assert((float(acc) - 0.666) < 0.001)