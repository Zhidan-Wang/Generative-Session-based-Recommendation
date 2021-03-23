import torch
import numpy as np

def get_recall(indices, targets):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)        # 将targets进行扩展至k的幅度,方便进行比较
    hits = (targets == indices).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero(as_tuple=False)[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall

def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero(as_tuple=False)
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr.item()

def get_precision(indices, targets):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)        # 将targets进行扩展至k的幅度,方便进行比较
    hits = (targets == indices).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero(as_tuple=False)[:, :-1].size(0)
    precision = float(n_hits) / targets.size(0)/targets.size(1)
    return precision

def get_ndcg(indices, targets, k):
    targets = targets.view(-1, 1).expand_as(indices)        # 将targets进行扩展至k的幅度,方便进行比较
    hits = (targets == indices).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    result = 0
    for single_hit in hits:
        result += np.log2(2)/np.log2(single_hit[1].item()+2)
    return result


def evaluate(indices, targets, k):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    precision = get_precision(indices, targets)
    ndcg = get_ndcg(indices, targets, k)
    if recall==0 and precision==0:
      f1=0
    else:
      f1 = 2*recall*precision/(recall+precision)
    
    return f1, ndcg, recall, mrr
