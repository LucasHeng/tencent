import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_key, negative_keys=None, next_action_type=None, negative_mode='unpaired', pos_mask=None ):
        return info_nce(query, positive_key, negative_keys, next_action_type,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=negative_mode, pos_mask=pos_mask)


def info_nce(query, positive_key, negative_keys=None, next_action_type=None, temperature=0.1, reduction='mean', negative_mode='unpaired', pos_mask=None):
    # L2归一化
    target_dtype = query.dtype
    
    if next_action_type is not None:
        next_action_type = next_action_type.reshape(-1)
        hit_pos = torch.where(next_action_type==1)[0]       
        if (len(next_action_type) != len(query)):
                raise ValueError(f'Vectors of <next_action_type> {len(next_action_type)} and <query> {len(query)} should have the same number of components.')
        

    # 转换positive_key到相同类型
    if positive_key.dtype != target_dtype:
        positive_key = positive_key.to(target_dtype)
    
    # 转换negative_keys到相同类型（如果存在）
    if negative_keys is not None and negative_keys.dtype != target_dtype:
        negative_keys = negative_keys.to(target_dtype)

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    
    S = query.shape[0]
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
    negative_keys = negative_keys.reshape(-1, negative_keys.shape[-1])
    negative_logits = query @ transpose(negative_keys)

    # First index in last dimension are the positive samples
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    if next_action_type is not None:
        logits[hit_pos, 0] *= 1.5
    labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    diag_mean = positive_logit.mean()
    non_diag_mean = negative_logits.mean()

    K = 10
    _, topk_indices = torch.topk(logits, k=K, dim=1)  # [S, k]
    expanded_labels = labels.view(-1, 1).expand_as(topk_indices)  # [S, k]
    correct = (topk_indices == expanded_labels)  # [S, k]
    correct_topk = correct.any(dim=1)  # [S]
    accuracy = correct_topk.float().mean() * 100.0

    return F.cross_entropy(logits / temperature, labels, reduction=reduction), accuracy, diag_mean, non_diag_mean, 0


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]