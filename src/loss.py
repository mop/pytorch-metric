import torch
import sklearn
import torch.nn.functional as F

def pairwise_distance(a: torch.Tensor, squared=False) -> torch.Tensor:
    pw_dist_sq = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    pw_dist_sq = torch.clamp(pw_dist_sq, min=0.0)

    invalid = torch.le(pw_dist_sq, 0.0)

    if squared:
        pw_dists = pw_dist_sq
    else:
        pw_dists = torch.sqrt(pw_dist_sq + invalid.float() * 1e-16)

    off_diags = 1 - torch.eye(*pw_dists.size(), device=pw_dists.device)
    pw_dists = torch.mul(pw_dists, off_diags)

    return pw_dists


def binarize_and_smooth_labels(T, num_classes, smooth_const=0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, num_classes))
    T = T * (1 - smooth_const)
    T[T == 0] = smooth_const / (num_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T

    
class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, num_classes, embedding_size, scale, **kwargs):
        super().__init__(**kwargs)
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embedding_size) / 8)
        self.scale = scale

    def forward(self, X, T):
        # self.scale is sqrt(1/T)
        P = self.proxies

        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        # distance to proxies
        D = pairwise_distance(torch.cat([X, P]), squared=True)[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T=T, num_classes=len(P), smooth_const=0)

        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss
