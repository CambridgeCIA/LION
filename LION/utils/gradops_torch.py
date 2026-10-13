import torch

from LION.utils import finite_diff_torch as fdt


class GradOpsTorch:
    def __init__(
        self,
        dx_forward=fdt.dx_forward,
        dy_forward=fdt.dy_forward,
        dx_backward=fdt.dx_backward,
        dy_backward=fdt.dy_backward,
    ):
        self.dx_forward = dx_forward
        self.dy_forward = dy_forward
        self.dx_backward = dx_backward
        self.dy_backward = dy_backward

    def nabla_h(self, u):
        # $\Nabla_h$ and $\mathcal{E}_h$ ???
        # See page 13 in 'Recovering piecewise smooth multichannel...'
        # https://unipub.uni-graz.at/obvugroa/content/titleinfo/125370

        # Parameters
        # ----------
        # u : torch.Tensor
        #     Assume 2D tensor of shape [n, n] (for now).

        #     scalar field?

        # Returns
        # -------
        # grad_u : torch.Tensor
        #     Assume 3D tensor of shape [n, n, 2] (for now).
        #     The point is that, the shape of the output is
        #       one more dimension added to the end of input,
        #     and that extra last dimension is of size 2.

        #     Gradient of the scalar field u?
        # assert u.dim() == 2,
        # f"u must be a 2D tensor, but got {u.dim()}D tensor"
        # Compute the gradient in both x and y directions

        """

        Example
        -------
        >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).cpu()
        >>> GradOpsTorch.nabla_h(u)
        tensor([[[3, 1],
                 [3, 1],
                 [3, 0]],
        <BLANKLINE>
                [[3, 1],
                 [3, 1],
                 [3, 0]],
        <BLANKLINE>
                [[0, 1],
                 [0, 1],
                 [0, 0]]])
        """

        # tensor([[[3, 1],
        #          [3, 1],
        #          [3, -3]],
        # <BLANKLINE>
        #         [[3, 1],
        #          [3, 1],
        #          [3, -6]],
        # <BLANKLINE>
        #         [[-7, 1],
        #          [-8, 1],
        #          [-9, -9]]])

        dx_f = self.dx_forward(u)
        dy_f = self.dy_forward(u)
        nabla_h_u = torch.stack([dx_f, dy_f], dim=-1)
        # del dx_f, dy_f
        # with torch.no_grad(): torch.cuda.empty_cache()
        # gc.collect()
        return nabla_h_u

    def e_h(self, v):
        # """

        # Parameters
        # ----------
        # v : torch.Tensor
        #     Assume 3D tensor of shape [n, n, 2] (for now).

        # Returns
        # -------
        # w : torch.Tensor
        #     Assume 4D tensor of shape [n, n, 2, 2] (for now).

        # Example
        # -------
        # >>> v = torch.tensor(
        # >>>   [[[1, 1], [2, 2], [3, 3]], [[4, 4],
        # >>>       [5, 5], [6, 6]], [[7, 7], [8, 8], [9, 9]]])
        # >>> GradOpsTorch.e_h(v)
        # tensor([[[[ 1,  2],
        #           [ 2,  5]],
        # <BLANKLINE>
        #           [[ 5,  6],
        #            [ 6,  9]]],
        # <BLANKLINE>
        #           [[[ 9, 10],
        # """
        # assert len(v.shape) == 3, f"v must be a 3D tensor,
        # but got {len(v.shape)}D tensor"
        assert v.shape[-1] == 2, (
            "v must have 2 channels in the last dimension, "
            + f"but got {v.shape[-1]} channels"
        )
        dx_b_1 = self.dx_backward(v[..., 0])
        dy_b_1 = self.dy_backward(v[..., 0])
        dx_b_2 = self.dx_backward(v[..., 1])
        dy_b_2 = self.dy_backward(v[..., 1])
        # half = torch.tensor(0.5)
        # print(f"dx_b_1: {dx_b_1}, dy_b_1: {dy_b_1},
        # dx_b_2: {dx_b_2}, dy_b_2: {dy_b_2}")
        # w_1 = torch.tensor([dx_b_1, half * (dy_b_1 + dx_b_2)**2])
        # w_2 = torch.tensor([half * (dy_b_1 + dx_b_2)**2, dy_b_2])
        # w = torch.tensor([w_1, w_2])

        w = torch.stack(
            [
                torch.stack([dx_b_1, 0.5 * (dy_b_1 + dx_b_2)], dim=-1),
                torch.stack([0.5 * (dy_b_1 + dx_b_2), dy_b_2], dim=-1),
            ],
            # Notes: Any dim is fine because symmetric (b == c)
            dim=-2,  # [a, b] and [c, d]  -->  [[a, b], [c, d]]
            # dim=-1   # [a, b] and [c, d]  -->  [[a, c], [b, d]]
        )
        # del dx_b_1, dy_b_1, dx_b_2, dy_b_2
        # with torch.no_grad(): torch.cuda.empty_cache()
        # gc.collect()
        return w

    def div_h_v(self, v):
        """
        $\text{div}_h$.
        See page 14 in 'Recovering piecewise smooth multichannel...'
        https://unipub.uni-graz.at/obvugroa/content/titleinfo/125370

        Parameters
        ----------
        v : torch.Tensor
            Assume 3D tensor of shape [n, n, 2] (for now).
            The point is that, the shape of the input is
            one more dimension added to the end of the output,
            and that extra last dimension is of size 2.

            representing the vector field?

        Returns
        -------
        div_v : torch.Tensor
            Assume 2D tensor of shape [n, n] (for now).

            Divergence of the vector field v?

        Example
        -------
        """
        # assert v.dim() == 3,
        # f"v must be a 3D tensor, but got {v.dim()}D tensor"
        # Compute the divergence from the gradient in both x and y directions
        dx_b_1 = self.dx_backward(v[..., 0])
        dy_b_2 = self.dy_backward(v[..., 1])
        div_h_v = dx_b_1 + dy_b_2
        # del dx_b_1, dy_b_2
        # with torch.no_grad(): torch.cuda.empty_cache()
        # gc.collect()
        return div_h_v

    def div_h_w(self, w):
        # assert len(w.shape) == 4, f"w must be a 4D tensor,
        # but got {len(w.shape)}D tensor"
        assert w.shape[-2] == 2, (
            "w must have 2 channels in the second last dimension, "
            + f"but got {w.shape[-2]} channels"
        )
        assert w.shape[-1] == 2, (
            "w must have 2 channels in the last dimension, "
            + f"but got {w.shape[-1]} channels"
        )
        dx_f_11 = self.dx_forward(w[..., 0, 0])
        dy_f_12 = self.dy_forward(w[..., 0, 1])

        dx_f_12 = self.dx_forward(w[..., 0, 1])
        dy_f_22 = self.dy_forward(w[..., 1, 1])

        v_1 = dx_f_11 + dy_f_12
        v_2 = dx_f_12 + dy_f_22
        v = torch.stack([v_1, v_2], dim=-1)
        # del dx_f_11, dy_f_12, dx_f_12, dy_f_22, v_1, v_2
        # with torch.no_grad(): torch.cuda.empty_cache()
        # gc.collect()
        return v
