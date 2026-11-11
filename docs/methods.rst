Methods in LION
===============

A map of what LION has: the kinds of learned reconstruction, the models of
each kind, how each is trained and how it reconstructs, and where each comes
from. Every method from a paper cites it: ``Class.cite()`` gives its BibTeX,
``Class.cite("MLA")`` the MLA form. The tutorials show them in use
(:doc:`tutorials/index`); the API reference has every argument (:doc:`api`).

The kinds of learned reconstruction
-----------------------------------

A learned reconstruction is a **model** (a network), **trained** by a solver,
and used to **reconstruct**, by itself or inside an algorithm. They differ in
where the network sits:

.. list-table::
   :header-rows: 1
   :widths: 18 42 20 20

   * - Kind
     - What the network does
     - Trained by
     - Reconstructs with
   * - Post-processing
     - Removes the artefacts of a classical reconstruction (e.g. FDK):
       image to image.
     - ``SupervisedSolver``, or a self-supervised solver
     - ``ForwardReconstructor``
   * - Unrolled iterative
     - An iterative algorithm unrolled into a fixed number of steps, each with
       learned parts, the operator between them: takes the measurements.
     - ``SupervisedSolver``
     - ``ForwardReconstructor``
   * - Learned FBP
     - Filtered backprojection with learned parts (filter, interpolation,
       fusion): takes the measurements.
     - ``SupervisedSolver``
     - ``ForwardReconstructor``
   * - Plug-and-Play (PnP)
     - A denoiser, trained on its own, used as the prior of an iterative
       algorithm (ADMM, HQS, FBS).
     - ``GaussianDenoiserSolver``
     - ``PnP``
   * - Learned regulariser
     - A function of the image (low for good images) used as the regulariser
       of a variational problem.
     - ``ARSolver``
     - ``LearnedRegularizerReconstructor``

Self-supervision is a way of *training*, not a kind of model: the
self-supervised solvers train image models (post-processing, usually) from
noisy measurements alone (see `Training`_).

Models
------

All in ``LION.models``, each a ``LIONmodel``. "Takes" is what ``forward``
takes: measurements (a model that uses the operator, made with its
parameters: ``Model(geometry)``) or images (a solver or reconstructor gives
it the operator's default reconstruction, FDK for CT).

Post-processing
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Model
     - Takes
     - Paper
   * - ``FBPConvNet``
     - measurements (FDK, then a U-Net)
     - Jin et al., "Deep convolutional neural network for inverse problems in
       imaging", IEEE Trans. Image Processing, 2017
   * - ``FBPConvNetImage``
     - images
     - the same, with the reconstruction made outside the model
   * - ``cFBPConvNet``
     - images
     - Runkel, Biguri and Schönlieb, "Continuous Learned Primal Dual", MLSP
       2024 (its continuous blocks in FBPConvNet)

Any image-to-image network below (``CNNs``) is a post-processing model too.

Unrolled iterative
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Model
     - Takes
     - Paper
   * - ``LG`` (Learned Gradient)
     - measurements
     - Adler and Öktem, "Solving ill-posed inverse problems using iterative
       deep neural networks", Inverse Problems, 2017; with a TV gradient
       (``LG.lgtv_parameters()``), LGTV, as in Kiss et al., 2025
   * - ``LPD`` (Learned Primal-Dual)
     - measurements
     - Adler and Öktem, "Learned primal-dual reconstruction", IEEE Trans.
       Medical Imaging, 2018
   * - ``cLPD``
     - measurements
     - Runkel, Biguri and Schönlieb, "Continuous Learned Primal Dual", MLSP
       2024
   * - ``ItNet``
     - measurements
     - Genzel et al., "Near-exact recovery for tomographic inverse problems
       via deep learning", ICML 2022

Learned FBP
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Model
     - Takes
     - Paper
   * - ``DeepFBPNetwork``
     - measurements
     - Tan et al., "Deep Filtered Back Projection for CT Reconstruction",
       IEEE Access, 2024
   * - ``FusionFBPNetwork``, ``DeepFusionBPNetwork``
     - measurements
     - Sainz Bear, "Deep Filtered Back Projection for CT Reconstruction",
       MPhil thesis, 2025

Image networks (``CNNs``), denoisers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image to image: post-processing networks, denoisers for PnP, or the models
of self-supervised training.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Model
     - Notes
     - Paper
   * - ``UNet``
     -
     - Ronneberger, Fischer and Brox, "U-Net: Convolutional Networks for
       Biomedical Image Segmentation", MICCAI 2015
   * - ``MSDNet``
     - few parameters, no downsampling
     - Pelt and Sethian, "A mixed-scale dense convolutional neural network
       for image analysis", PNAS, 2018
   * - ``DnCNN``
     - denoiser
     - Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep
       CNN for Image Denoising", IEEE Trans. Image Processing, 2017
   * - ``DRUNet``
     - denoiser, takes the noise level; ``DRUNet.pretrained()``: the
       authors' weights
     - Zhang et al., "Plug-and-Play Image Restoration With Deep Denoiser
       Prior", IEEE TPAMI, 2022
   * - ``GSD`` (gradient-step denoiser)
     - wraps any denoiser N as x - ∇g(x), g(x) = ½‖x - N(x)‖²
     - Hurault, Leclaire and Papadakis, "Gradient Step Denoiser for
       Convergent Plug-and-Play", ICLR 2022
   * - ``ResUNet3D``
     - 3D, for segmentation
     - Yu et al., "Design of lung nodules segmentation and recognition
       algorithm based on deep learning", BMC Bioinformatics, 2021

Learned regularisers
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Model
     - Notes
     - Paper
   * - ``AR`` (adversarial regulariser)
     -
     - Lunz, Öktem and Schönlieb, "Adversarial regularizers in inverse
       problems", NeurIPS 2018
   * - ``ACR`` (adversarial convex regulariser)
     - convex in the image
     - Mukherjee et al., "Learned convex regularizers for inverse problems",
       arXiv:2008.02839, 2020

Your own
~~~~~~~~

Any torch module is a LION model with ``TorchModel(YourClass, arguments)``;
a model that uses the operator derives from ``LIONmodel``
(:doc:`tutorials/d13_extending_lion`).

Training
--------

All in ``LION.optimizers``, each a ``LIONsolver``.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Solver
     - Trains
     - Paper
   * - ``SupervisedSolver``
     - any model, on (measurements, image) pairs
     -
   * - ``GaussianDenoiserSolver``
     - a denoiser, on clean images with Gaussian noise added (noise ranges,
       patches, augmentation, as in the literature)
     - as DnCNN and DRUNet's training (Zhang et al.)
   * - ``ARSolver``
     - an adversarial regulariser (WGAN loss, gradient penalty)
     - Lunz et al., NeurIPS 2018
   * - ``Noise2InverseSolver``
     - self-supervised, from splits of the measurements
     - Hendriksen, Pelt and Batenburg, "Noise2Inverse: Self-supervised deep
       convolutional denoising for tomography", IEEE TCI, 2020
   * - ``Sparse2InverseSolver``
     - self-supervised, sparse view
     - Gruber et al., "Sparse2Inverse: Self-supervised inversion of
       sparse-view CT data", arXiv:2402.16921, 2024
   * - ``Noisier2InverseSolver``
     - self-supervised, correlated noise
     - Gruber et al., "Noisier2Inverse: Self-Supervised Learning for Image
       Reconstruction with Correlated Noise", arXiv:2503.19468, 2025
   * - ``Proj2ProjSolver``
     - self-supervised, masked projections
     - Unal, Ertas and Yildirim, "Proj2Proj: self-supervised low-dose CT
       reconstruction", PeerJ Computer Science, 2024
   * - ``EquivariantSolver``
     - self-supervised, equivariance to rotations
     - Chen, Tachella and Davies, "Equivariant imaging: Learning beyond the
       range space", ICCV 2021
   * - ``Equivariance2InverseSolver``
     - self-supervised, equivariance and projection-wise cross-validation
     - Schut et al., "Equivariance2Inverse: A Practical Self-Supervised CT
       Reconstruction Method Benchmarked on Real, Limited-Angle, and Blurred
       Data", arXiv:2510.23317, 2025
   * - ``SelfSupervisedSolver``
     - any loss of the model and its input (e.g. SURE)
     -

Penalties on the model (a solver's ``model_regularization``,
``LION.losses.jacobian_penalty``), for PnP denoisers that converge:

- ``FirmlyNonexpansivePenalty``: Pesquet et al., "Learning maximally
  monotone operators for image recovery", SIAM J. Imaging Sciences, 2021;
- ``ProximalDenoiserPenalty``: Hurault, Leclaire and Papadakis, "Proximal
  Denoiser for Convergent Plug-and-Play Optimization with Nonconvex
  Regularization", ICML 2022.

Losses (``LION.losses``): ``SURE`` (Metzler et al., "Unsupervised learning
with Stein's unbiased risk estimator", arXiv:1805.10531, 2018; not usable yet
for CT, issue #144) and ``SUREpgLoss`` (Poisson-Gaussian SURE, Le Montagner,
Angelini and Olivo-Marin, IEEE TIP, 2014).

Reconstruction
--------------

Reconstructors (``LION.reconstructors``, each a ``LIONReconstructor``) use a
trained model; :doc:`tutorials/d10_reconstructors` puts them side by side.

.. list-table::
   :header-rows: 1
   :widths: 32 40 28

   * - Reconstructor
     - Does
     - Paper
   * - ``ForwardReconstructor``
     - the model alone, as its solver ran it
     -
   * - ``PnP``
     - ADMM, HQS or FBS with a denoiser as the prior
     - Venkatakrishnan, Bouman and Wohlberg, "Plug-and-Play priors for model
       based reconstruction", GlobalSIP 2013; HQS as in Zhang et al., TPAMI
       2022
   * - ``LearnedRegularizerReconstructor``
     - gradient descent on ½‖Ax - y‖² + α R(x), for any regulariser R
     - Lunz et al., NeurIPS 2018

Classical reconstructions, for any operator (``LION.classical_algorithms``):
conjugate gradient, FISTA (Beck and Teboulle, 2009), SPGL1 (van den Berg and
Friedlander, 2008); for CT (``LION.operators.ct``): FDK/FBP, SIRT and TV
minimisation (tomosipo's algorithms).

Data and experiments
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Dataset
     - What it is
     - Paper
   * - ``LIDC_IDRI``
     - lung CT of 1010 patients with annotated nodules; LION simulates the
       sinograms
     - Armato et al., Medical Physics, 2011
   * - ``deteCT``
     - 2DeteCT: measured sinograms of 109 objects in three modes (5,000
       slices)
     - Kiss et al., "2DeteCT", Scientific Data, 2023

Experiments (``LION.experiments``) fix a problem on them: LION's simulated
ones on LIDC-IDRI (``ct_experiments``: clinical, low and extremely low dose;
full, sparse-view and limited-angle sampling), which also run on 2DeteCT's
measured data, and the 2DeteCT benchmark paper's (``deteCT_benchmark``: Kiss
et al., "Benchmarking learned algorithms for computed tomography image
reconstruction tasks", Applied Mathematics for Modern Challenges, 2025).

Operators
---------

``LION.operators``: CT (``ct``: the geometry, the projector, noise, FDK),
photocurrent mapping (``pcm``: Koutsourakis, Thompson and Blakesley, Solar
RRL, 2021), wavelets, and composite and debiasing operators for sparse
recovery. Everything above that uses an operator works with any of them that
defines what it needs (e.g. ``autograd()`` to train through it).
:doc:`tutorials/d14_beyond_ct` uses photocurrent mapping's, with the
sparse-recovery tools and PnP.
