from __future__ import annotations
import math
import tomosipo as ts
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.models.score_inverse.utils import batch_apply

class SIRTAdj:
    """
    A batch-applicable preconditioned adjoint (SIRT backprojection) operator.

    This operator performs the transformation:
        x = C * A^T * R * y
    where A is the projection operator, A^T is the backprojection operator (adjoint),
    R is the row normalization scaling, and C is the column normalization scaling.
    """
    def __init__(self, op: ts.Operator.Operator | Geometry, device: str | torch.device = "cuda"):
        """
        Initialize the SIRT preconditioned adjoint operator.

        Args:
            op: ts.Operator.Operator or Geometry. The forward projection operator or geometry.
            device: str or torch.device. The device to compute and store the scaling buffers.

        Attributes:
            device: torch.device. The target device for computation.
            raw_op: ts.Operator.Operator. The raw forward projection operator.
            C: torch.Tensor. The column scaling factor (in volume space).
            R: torch.Tensor. The row scaling factor (in projection space).
            sparse_op_adjoint: Callable. The batch-applied backprojection operator.
        """
        self.device = torch.device(device)
        
        # Convert Geometry to Operator if necessary
        if isinstance(op, Geometry):
            self.raw_op = make_operator(op)
        else:
            self.raw_op = op
            
        # 1. Compute C (Column weights / Volume space normalization)
        # Represents (A^T * 1_projection)^-1
        y_tmp = torch.ones(self.raw_op.range_shape, device=self.device)
        self.C = self.raw_op.T(y_tmp)
        self.C[self.C < ts.epsilon] = math.inf
        self.C.reciprocal_()
        
        # 2. Compute R (Row weights / Projection space normalization)
        # Represents (A * 1_volume)^-1
        x_tmp = torch.ones(self.raw_op.domain_shape, device=self.device)
        self.R = self.raw_op(x_tmp)
        self.R[self.R < ts.epsilon] = math.inf
        self.R.reciprocal_()
        
        # 3. Batch-applied Adjoint Operator
        self.sparse_op_adjoint = batch_apply(self.raw_op.T)

    def __call__(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Apply the preconditioned adjoint operator to the input projection data.

        Args:
            sino: torch.Tensor of shape (batch_size, channels, n_angles, n_detectors). The input projection data.

        Returns:
            torch.Tensor of shape (batch_size, channels, height, width): The reconstructed volume.
        """
        return self.sparse_op_adjoint(sino * self.R) * self.C