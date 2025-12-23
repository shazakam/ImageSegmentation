import torch

def binary_iou(pred, target, threshold=0.5, eps=1e-7):
    """
    pred:   (B, 1, H, W) logits or probabilities
    target: (B, 1, H, W) binary mask {0,1}
    """
    pred = torch.sigmoid(pred) if pred.dtype.is_floating_point else pred
    pred = (pred > threshold).bool()
    target = target.bool()

    intersection = (pred & target).sum(dim=(1,2,3))
    union = (pred | target).sum(dim=(1,2,3))

    iou = (intersection + eps) / (union + eps)

    # handle empty masks
    iou[union == 0] = 1.0

    return iou.mean()