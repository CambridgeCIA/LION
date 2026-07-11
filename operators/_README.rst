Operators
=========

Originally designed for tomographic reconstruction tasks, specifically Computed Tomography (CT),
LION has been extended to work with more operators.
Specifically, LION can work with any unitary operator ``A`` (single input, single output) that has:

- a forward method ``A.forward(x)`` that maps an input image ``x`` to a measurement or projection ``y``
- an adjoint method ``A.adjoint(y)`` that maps a measurement or projection ``y`` back to an image ``x``
- shape of the domain (image ``x``) ``A.domain_shape``
- shape of the range (projection or measurement ``y``) ``A.range_shape``

The input and output should be torch tensors.
Users can define their own operators by creating a class that subclasses
`LION.utils.operators.Operator <Operator.py>`_ and implements the necessary methods.
We also include some operators such as:

- `CTProjectionOp <CTProjectionOp.py>`_:
  Discrete tomographic projection operator for CT reconstruction tasks.
  It wraps a `tomosipo <https://github.com/ahendriksen/tomosipo>`_ Operator and
  uses its PyTorch backend for efficient GPU-accelerated projection and backprojection.

- `PhotocurrentMapOp <PhotocurrentMapOp.py>`_:
  Forward operator for photocurrent mapping (PCM), modelling single-pixel measurements
  obtained from spatially modulated illumination patterns.
  It uses subsampled Walsh-Hadamard patterns and dyadic permutations to map a
  2D current map to a compressed set of photocurrent measurements.

- `WalshHadamard2D <WalshHadamard2D.py>`_:
  Two-dimensional (unnormalized) Walsh-Hadamard transform operator acting on images.
  This operator is useful in single-pixel imaging.

- `Wavelet2D <Wavelet2D.py>`_:
  Two-dimensional orthogonal wavelet transform operator on images.
  The forward method maps an image to its wavelet coefficient representation, and the
  adjoint reconstructs the image from these coefficients.

Please refer to the documentation of each operator for more details.
