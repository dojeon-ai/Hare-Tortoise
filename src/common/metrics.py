import torch


def get_grad_norm_stats(model) -> dict:
    grad_norm = []
    stats = {}
    for p in model.parameters():
        if p.grad is not None:
            grad_norm.append(p.grad.detach().data.norm(2))
    grad_norm = torch.stack(grad_norm)
    stats['min_grad_norm'] = torch.min(grad_norm).item()
    stats['mean_grad_norm'] = torch.mean(grad_norm).item()
    stats['max_grad_norm'] = torch.max(grad_norm).item()

    return stats


def accuracy(
        output: torch.Tensor, #(n,) 
        target: torch.Tensor, #(n,) 
        topk=(1,)
    ):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    n= target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    acc_list = []
    for k in topk:
        acc_k = correct[:k].float().sum(0).mean(0)
        acc_list.append(acc_k)

    return acc_list


# smooth-rank: https://openreview.net/forum?id=uGEBxC8dnEh
# stable-rank: https://openreview.net/pdf?id=H1enKkrFDB
def get_rank(S):
    """
    (params) S: torch.Tensor (n), array of singular values
    """    
    EPS = 1e-6
    
    # smooth rank
    P = S / S.sum() + EPS
    smooth_rank = (-P@(P.log())).exp()

    # stable rank
    cum_prob = torch.cumsum(S, dim=0) / S.sum()
    stable_rank = torch.where(cum_prob > 0.99)[0][0] + 1

    return smooth_rank, stable_rank