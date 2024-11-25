# Papers in LION

This is a list of papers implemented in the LION toolbox adn where to find the code in the library. 

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