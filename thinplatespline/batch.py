"""Thin plate splines for batches."""

import torch

DEVICE = torch.device("cpu")


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def K_matrix(X, Y):
    """Calculates the upper-right (k, k) submatrix of the
        (k + 3, k + 3) shaped L matrix.

    Parameters
    ----------
    X : (N, k, 2) torch.tensor of k points in 2 dimensions.
    Y : (1, m, 2) torch.tensor of m points in 2 dimensions.

    Returns
    -------torch.linalg.solve`
    K : torch.tensor
    """

    eps = 1e-9

    D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
    K = D2 * torch.log(D2 + eps)
    return K


def P_matrix(X):
    """Makes the minor diagonal submatrix P
    of the (k + 3, k + 3) shaped L matrix.

    Stacks a column of 1s before the coordinate columns in X.

    Parameters
    ----------
    X : (N, k, 2) torch.tensor of k points in 2 dimensions.

    Returns
    -------
    P : (N, k, 3) tensor, which is 1 in the first column, and
        exactly X in the remaining columns.
    """
    n, k = X.shape[:2]
    device = X.device

    P = torch.ones(n, k, 3, device=device)
    P[:, :, 1:] = X
    return P


class TPS_coeffs(torch.nn.Module):
    """Finds the thin-plate spline coefficients for the tps
    function that interpolates from X to Y.  # 从x插值到y

    Parameters
    ----------
    X : torch tensor (N, K, 2), eg. projected points.
    Y : torch tensor (1, K, 2), eg. a UV map.

    Returns
    -------
    W : torch.tensor. (N, K, 2), the non-affine part of the spline
    A : torch.tensor. (N, K+1, K) the affine part of the spline.
    """

    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        """Override abstract function.

            [K     P ] =  [V]
       L =
            [P^T  O ]   [O]

        """

        n, k = X.shape[:2]  # n = 77, k =2
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device) # [1, 80, 80]
        K = K_matrix(X, X)


        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]
        Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]


class TPS(torch.nn.Module):
    """Calculate the thin-plate-spline (TPS) surface at xy locations.

    Thin plate splines (TPS) are a spline-based technique for data
    interpolation and smoothing.
    see: https://en.wikipedia.org/wiki/Thin_plate_spline

    Constructor Params:
    device: torch.device
    size: tuple Output grid size as HxW. Output image size, default (256. 256).

    Parameters
    ----------
    X : torch tensor (N, K, 2), eg. projected points.
    Y : torch tensor (1, K, 2), for example, a UV map.

    Returns
    -------
     grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
    """

    def __init__(self, size: tuple = (256, 256), device=DEVICE):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device
        self.tps = TPS_coeffs()
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, X, Y):
        """Override abstract function."""
        h, w = self.size
        W, A = self.tps(X, Y)  # 这一步是求tps系数的，给你两组控制点，求出TPS系数
        U = K_matrix(self.grid, X) #
        P = P_matrix(self.grid)
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2)   # 返回变换后的坐
