import torch.nn.functional as F
import torch


def cross_entropy(pred, trg_seq):
    ''' Calculate cross entropy loss'''

    log_prb = F.log_softmax(pred, dim=1)
    loss = -(trg_seq * log_prb).sum(dim=1)
    loss = loss.mean()

    return loss


def kl_loss(q_logit, p_logit):
    ''' Calculate K-L div loss'''

    # p 是真实分布， q 是拟合分布
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), -1)
    return torch.sum(_kl)


def mse_loss(q_logit, p_logit):
    ''' Calculate MSE loss'''

    # p 是真实分布， q 是拟合分布
    loss = (p_logit - q_logit) * (p_logit - q_logit)
    loss = torch.sum(loss, -1)

    return torch.mean(loss)
