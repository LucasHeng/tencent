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
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if next_action_type is not None:
        next_action_type = next_action_type.reshape(-1)
        hit_pos = torch.where(next_action_type==1)[0]    
        if (len(next_action_type) != len(query)):
                raise ValueError(f'Vectors of <next_action_type> {len(next_action_type)} and <query> {len(query)} should have the same number of components.')
        
    S = query.shape[0]
    # Explicit negative keys

    # Cosine between positive pairs
    # [B*S, 1]
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

    # Negative keys are implicitly off-diagonal positive keys.

    # Cosine between all combinations
    #[batch_size*max_len, batch_size*max_len]
    # [S, D]
    # [1, S, D] [S, 1, D]
    # device = query.device
    # 计算logits并应用掩码
    # logits = query @ transpose(positive_key)
    
    # # Positive keys are the entries on the diagonal
    # labels = torch.arange(len(query), device=query.device)
    # diag_mask = torch.eye(S, dtype=torch.bool, device=logits.device)
    
    # if pos_mask is not None:
    #     filter_mask = pos_mask & ~diag_mask
    #     logits[filter_mask] = -1e18
    
    # # [S, S]
    # if next_action_type is not None:
    #     logits[hit_pos, hit_pos] *= 1.5


    # # 3. 对角线有效元素：对角线且有效
    # diag_valid_mask = diag_mask
    # diag_elements = logits[diag_valid_mask]
    # diag_mean = diag_elements.mean() if diag_elements.numel() > 0 else torch.tensor(0.0, device=logits.device)
    # # 4. 非对角线有效元素：非对角线且有效
    # non_diag_valid_mask = ~diag_mask & ~pos_mask
    # non_diag_elements = logits[non_diag_valid_mask]
    # non_diag_mean = non_diag_elements.mean() if non_diag_elements.numel() > 0 else torch.tensor(0.0, device=logits.device)
    
    K = 10
    _, topk_indices = torch.topk(logits, k=K, dim=1)  # [S, k]
    expanded_labels = labels.view(-1, 1).expand_as(topk_indices)  # [S, k]
    correct = (topk_indices == expanded_labels)  # [S, k]
    correct_topk = correct.any(dim=1)  # [S]
    accuracy = correct_topk.float().mean() * 100.0

    # # 10个命中率
    # # 生成列索引偏移：0, 1, ..., k-1（形状为[1, k]）
    # col_indices = torch.arange(K, device=query.device).view(1, -1)  # 形状 [1, k]
    
    # # 生成行起始索引：0, 1, ..., s-1（形状为[s, 1]）
    # row_starts = torch.arange(S, device=query.device).view(-1, 1)  # 形状 [s, 1]
    
    # # 计算原始矩阵：每行 = 行起始索引 + 列偏移（形状 [s,S k]）
    # raw_matrix = row_starts + col_indices
    
    # # 创建掩码：标记所有大于等于s的元素（超出有效范围）
    # mask = raw_matrix >= S
    
    # # 超出范围的元素置为-1
    # matrix = raw_matrix.masked_fill(mask, -1)
    # # 扩展matrix为[S, K, 1]，便于与B的每行进行广播比较
    # matrix_expanded = matrix.unsqueeze(2)  # 形状 [S, K, 1]
    
    # # 扩展B为[S, 1, K]，便于与matrix的每个元素比较
    # topk_indices = topk_indices.unsqueeze(1)  # 形状 [S, 1, K]
    # # 逐元素比较：matrix[i][j]与B[i]的所有元素是否相等 → 形状 [S, K, K]
    # # 然后按最后一维取any，判断是否存在至少一个相等 → 形状 [S, K]
    # duplicate_mask = (matrix_expanded == topk_indices).any(dim=2)
    # total_duplicates = duplicate_mask.sum().item()
    # # 计算矩阵的总元素数量
    # total_elements = duplicate_mask.numel()  # numel() 返回张量中元素的总数
    
    # # 计算平均值（重复率）
    # duplicate_mean = total_duplicates / total_elements if total_elements != 0 else 0.0

    return F.cross_entropy(logits / temperature, labels, reduction=reduction), accuracy, diag_mean, non_diag_mean, 0


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]