# Papers in LION

This is a list of papers implemented in the LION toolbox and where to find the code in the library.

Components that expose LION's citation API accept `cite("MLA")` for a readable
reference and `cite("bib")` for importable BibTeX. The PaDIS reconstructor
prints the complete citation set for its patch prior, samplers, EDM conventions,
and FDK initialisation.

## Machine Learning models

#### Mixed-Scale dense network (MS-D Net)

Pelt, Daniël M., and James A. Sethian. "A mixed-scale dense convolutional neural network for image analysis." Proceedings of the National Academy of Sciences 115.2 (2018): 254-259.
[https://doi.org/10.1073/pnas.1715832114](https://doi.org/10.1073/pnas.1715832114)

`LION/models/CNNs/MS-D/`               Submodule with the original repo

`LION/models/CNNs/MSD_pytorch.py`      the LIONmodel to load the original code

`LION/models/CNNs/MSDNet.py`           Our version of the MSD_pytorch model. Uses more memory

#### Learned Primal Dual (LPD)

Adler, Jonas, and Ozan Öktem. "Learned primal-dual reconstruction." IEEE transactions on medical imaging 37.6 (2018): 1322-1332.
[https://doi.org/10.1109/TMI.2018.2799231](https://doi.org/10.1109/TMI.2018.2799231)

`LION/models/iterative_unrolled/LPD.py` 

#### Continous Learned Primal Dual (cLPD)

C. Runkel, A. Biguri and C. -B. Schönlieb, "Continuous Learned Primal Dual," 2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP), London, United Kingdom, 2024, pp. 1-6,
[https://doi.org/10.1109/MLSP58920.2024.10734760](https://doi.org/10.1109/MLSP58920.2024.10734760)

`LION/models/iterative_unrolled/cLPD.py` 

#### Learned Gradient (LG)

Adler, Jonas, and Ozan Öktem. "Solving ill-posed inverse problems using iterative deep neural networks." Inverse Problems 33.12 (2017): 124007.
[https://doi.org/10.1088/1361-6420/aa9581](https://doi.org/10.1088/1361-6420/aa9581)

`LION/models/iterative_unrolled/LG.py`

#### NCSN++ and score-based SDE sampling

Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano
Ermon, and Ben Poole. "Score-Based Generative Modeling through Stochastic
Differential Equations." International Conference on Learning Representations
(2021).
[Official conference listing](https://openreview.net/forum?id=PxTIG12RRHS)

`LION/models/diffusion/NCSNpp.py`

## Diffusion priors and inverse-problem samplers

#### Patch Diffusion Inverse Solver (PaDIS)

Hu, Jason, Bowen Song, Xiaojian Xu, Liyue Shen, and Jeffrey Fessler. "Learning
Image Priors Through Patch-Based Diffusion Models for Solving Inverse
Problems." Advances in Neural Information Processing Systems 37 (2024):
1625-1660.
[https://doi.org/10.52202/079017-0052](https://doi.org/10.52202/079017-0052)

`LION/reconstructors/PaDIS.py` assembles the whole-image patch score and provides
the reconstruction samplers. `LION/losses/PaDIS.py` and
`LION/optimizers/PaDISSolver.py` implement patch training.

#### Diffusion posterior sampling (DPS)

Chung, Hyungjin, Jeongsol Kim, Michael Thompson McCann, Marc Louis Klasky, and
Jong Chul Ye. "Diffusion Posterior Sampling for General Noisy Inverse
Problems." The Eleventh International Conference on Learning Representations
(2023).
[Official conference listing](https://openreview.net/forum?id=OnD9zGAGT0k)

`LION/reconstructors/PaDIS.py` implements DPS measurement conditioning for
patch and whole-image diffusion priors.

#### Annealed Langevin dynamics

Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of
the Data Distribution." Advances in Neural Information Processing Systems 32
(2019).
[Official NeurIPS listing](https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html)

`LION/reconstructors/PaDIS.py` implements Langevin prior updates and the
PaDIS-DPS/Langevin sampler.

#### Predictor-corrector sampling

The predictor-corrector framework is attributed to Song et al.'s score-SDE
paper above. The PaDIS-specific patch-layout and CT sampling conventions are
additionally attributed to Hu et al.'s PaDIS publication.

`LION/reconstructors/PaDIS.py`

#### Denoising Diffusion Null-Space Model (DDNM)

Wang, Yinhuai, Jiwen Yu, and Jian Zhang. "Zero-Shot Image Restoration Using
Denoising Diffusion Null-Space Model." The Eleventh International Conference
on Learning Representations (2023).
[Official conference listing](https://openreview.net/forum?id=mRieQgMtNTQ)

`LION/reconstructors/PaDIS.py` implements the VE-DDNM correction used with the
PaDIS and whole-image scores.

#### EDM preconditioning and noise schedules

Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine. "Elucidating the
Design Space of Diffusion-Based Generative Models." Advances in Neural
Information Processing Systems 35 (2022): 26565-26577.
[https://doi.org/10.52202/068431-1926](https://doi.org/10.52202/068431-1926)

`LION/models/diffusion/NCSNpp.py`, `LION/losses/PaDIS.py`,
`LION/optimizers/PaDISSolver.py`, and `LION/reconstructors/PaDIS.py`

## Classical and plug-and-play reconstruction

#### Feldkamp-Davis-Kress (FDK)

Feldkamp, L. A., L. C. Davis, and J. W. Kress. "Practical Cone-Beam
Algorithm." JOSA A 1.6 (1984): 612-619.
[https://doi.org/10.1364/JOSAA.1.000612](https://doi.org/10.1364/JOSAA.1.000612)

`LION/classical_algorithms/fdk.py`

#### Chambolle-Pock total-variation reconstruction

Chambolle, Antonin, and Thomas Pock. "A First-Order Primal-Dual Algorithm for
Convex Problems with Applications to Imaging." Journal of Mathematical Imaging
and Vision 40.1 (2011): 120-145.
[https://doi.org/10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

`LION/classical_algorithms/tv_min.py`

#### Plug-and-Play ADMM

Chan, Stanley H., Xiran Wang, and Omar A. Elgendy. "Plug-and-Play ADMM for
Image Restoration: Fixed-Point Convergence and Applications." IEEE
Transactions on Computational Imaging 3.1 (2017): 84-98.
[https://doi.org/10.1109/TCI.2016.2629286](https://doi.org/10.1109/TCI.2016.2629286)

`LION/reconstructors/PnP.py`

## Tomography software dependencies

#### tomosipo

Hendriksen, Allard A., et al. "Tomosipo: Fast, Flexible, and Convenient 3D
Tomography for Complex Scanning Geometries in Python." Optics Express 29.24
(2021): 40494.
[https://doi.org/10.1364/OE.439909](https://doi.org/10.1364/OE.439909)

#### ASTRA Toolbox

Van Aarle, Wim, et al. "Fast and Flexible X-Ray Tomography Using the ASTRA
Toolbox." Optics Express 24.22 (2016): 25129.
[https://doi.org/10.1364/OE.24.025129](https://doi.org/10.1364/OE.24.025129)

Both backends are linked through `LION/operators/CTProjectionOp.py` and
`LION/CTtools/ct_utils.py`; tomosipo constructs LION's CT operators and dispatches
projection operations to ASTRA.

## Training Strategies

#### Supervised learning

This is not a paper (or perhasp many), but LION has a class to do a standard supervised training loop. 


`LION/optimizers/SupervisedSolver.py`

#### Noise2Inverse (self-supervised)

Hendriksen, Allard Adriaan, Daniël Maria Pelt, and K. Joost Batenburg. "Noise2inverse: Self-supervised deep convolutional denoising for tomography." IEEE Transactions on Computational Imaging 6 (2020): 1320-1335.
[https://doi.org/10.1109/TCI.2020.3019647](https://doi.org/10.1109/TCI.2020.3019647)

`LION/optimizers/Noise2InverseSolver.py`

#### Equivariant training  (self-supervised)

Chen, Dongdong, Julián Tachella, and Mike E. Davies. "Equivariant imaging: Learning beyond the range space." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
[https://doi.org/10.1109/ICCV48922.2021.00434](https://doi.org/10.1109/ICCV48922.2021.00434)

`LION/optimizers/EquivariantSolver.py`


## Datasets

#### 2DeteCT

Kiss, Maximilian B., et al. "2DeteCT-A large 2D expandable, trainable, experimental Computed Tomography dataset for machine learning." Scientific data 10.1 (2023): 576.
[https://doi.org/10.1038/s41597-023-02484-6](https://doi.org/10.1038/s41597-023-02484-6)

`LION/data_loaders/2deteCT/`       Code to download and pre-process a LION version of the 2deteCT, made with the authors. 

`LION/data_loaders/deteCT.py`      Pytorch DataSet

#### LIDC-IDRI

Armato III, Samuel G., et al. "The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans." Medical physics 38.2 (2011): 915-931.
[https://doi.org/10.1118/1.3528204](https://doi.org/10.1118/1.3528204)

`LION/data_loaders/LIDC_IDRI/`     Code to pre-process a LION version of the dataset

`LION/data_loaders/LIDC_IDRI.py`   Pytorch DataSet

## Loss functions

#### Steins Unbased risk estimator (SURE)

Metzler, Christopher A., et al. "Unsupervised learning with Stein's unbiased risk estimator." arXiv preprint arXiv:1805.10531 (2018).
[https://doi.org/10.48550/arXiv.1805.10531](https://doi.org/10.48550/arXiv.1805.10531)

`LION/losses/SURE.py`     The loss function itself. Use with `SelfSupervisedSolver`

## Misc
