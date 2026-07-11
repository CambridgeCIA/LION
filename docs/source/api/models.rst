LION.models
===========

Package guidance
----------------

.. include:: ../../../LION/models/README.md
   :parser: myst_parser.sphinx_

Model base classes
------------------

.. automodule:: LION.models.LIONmodel
   :members:
   :show-inheritance:

NCSN++ diffusion model
----------------------

.. automodule:: LION.models.diffusion.NCSNpp
   :members:
   :show-inheritance:

🚧 Undocumented model families
------------------------------

- ``LION.models.CNNs``
- ``LION.models.PnP``
- ``LION.models.fully_learned``
- ``LION.models.iterative_unrolled``
- ``LION.models.learned_fbp``
- ``LION.models.learned_regularizer``
- ``LION.models.post_processing``

These families are indexed by their present package locations. Dedicated
architecture guides and complete public-member docstrings remain future work;
their currently discoverable classes and functions are still included below.

.. sourceautosummary:: LION.models.CNNs.MSDNet

.. sourceautosummary:: LION.models.CNNs.MSD_pytorch

.. sourceautosummary:: LION.models.CNNs.dncnn

.. sourceautosummary:: LION.models.CNNs.drunet

.. sourceautosummary:: LION.models.PnP.gradient_step_denoiser

.. sourceautosummary:: LION.models.PnP.gs_drunet

.. sourceautosummary:: LION.models.iterative_unrolled.ItNet

.. sourceautosummary:: LION.models.iterative_unrolled.LG

.. sourceautosummary:: LION.models.iterative_unrolled.LPD

.. sourceautosummary:: LION.models.iterative_unrolled.cLPD

.. sourceautosummary:: LION.models.learned_fbp.DeepFBP

.. sourceautosummary:: LION.models.learned_fbp.DeepFusionBP

.. sourceautosummary:: LION.models.learned_fbp.FusionFBP

.. sourceautosummary:: LION.models.learned_regularizer.ACR

.. sourceautosummary:: LION.models.learned_regularizer.AR

.. sourceautosummary:: LION.models.post_processing.FBPConvNet

.. sourceautosummary:: LION.models.post_processing.FBPConvNetImage

.. sourceautosummary:: LION.models.post_processing.FBPMSDNet

.. sourceautosummary:: LION.models.post_processing.cFBPConvNet
