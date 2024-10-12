import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class Loss(L.LightningModule):
    def __init__(self, alpha: float):
        super(Loss, self).__init__()
        assert 0.0 <= alpha <= 1.0, f"Alpha must be in [0, 1], got {alpha}."

        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, y_hat: Tensor, y: Tensor, latent_representations: list, nb_views: int):
        y = y.unsqueeze(1).expand(-1, nb_views)  # (batch, nb_views)

        # reshape to (batch * nb_views, num_classes)
        y_hat = y_hat.view(-1, y_hat.size(2))
        y = y.reshape(-1)

        if self.alpha == 1.0:
            return self.cross_entropy(y_hat, y)
        else:
            return (
                    self.alpha * self.cross_entropy(y_hat, y) +
                    (1 - self.alpha) * self.distinct_path_regularization(latent_representations))

        # return (self.cross_entropy(y_hat, y) +
        #         self.distinct_path_regularization(latent_representations))

    # L2 distance between all pairs of views
    # def distinct_path_regularization(self, latent_representations):
    #     """Compute the regularization term to penalize similar representations.
    #     In: List of latent representations of shape (batch, nb_views, C, H, W)
    #     Out: Scalar
    #     """
    #     reg_loss = 0.0
    #     for l in latent_representations:
    #         assert l.dim() == 5, f"Latent representation shape is {l.shape}, expected 5 dimensions."
    #         # Compute pairwise distances between views
    #         batch, nb_views, *dims = l.shape
    #         l = l.view(batch, nb_views, -1)  # Flatten spatial dimensions
    #
    #         distances = torch.cdist(l, l, p=2)  # Pairwise L2 distances
    #         reg_loss += distances.mean()
    #     return reg_loss

    def distinct_path_regularization(self, latent_representations):
        """Compute the regularization term to penalize similar representations.
        In: List of latent representations of shape (batch, nb_views, C, H, W)
        Out: Scalar
        """
        reg_loss = 0.0
        for l in latent_representations:
            assert l.dim() == 5, f"Latent representation shape is {l.shape}, expected 5 dimensions."
            # Compute pairwise dot products between views
            batch, nb_views, *dims = l.shape
            l = l.view(batch, nb_views, -1)  # Flatten spatial dimensions

            # Normalize the vectors
            l = F.normalize(l, p=2, dim=-1)

            # Compute pairwise dot products
            dot_products = torch.bmm(l, l.transpose(1, 2))  # (batch, nb_views, nb_views)

            # We want to penalize high similarity, so we take the mean of the dot products
            reg_loss += dot_products.mean()
        return reg_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    loss = Loss(0)
    b = 1
    v = 5
    y_hat = torch.randn(b, v, 10)  # (batch, nb_views, num_classes)
    y = torch.randint(0, 10, (b, v))  # (batch, nb_views)
    latent_representations = [torch.randn(b, v, 64, 7, 7), torch.randn(b, v, 128, 3, 3)]
    loss(y_hat, y, latent_representations, v)
