# Models for data-driven CT reconstruction

There are 5 existing ways of doing CT reconstruction using data-driven methods:

- Fully-learned: From sinogram to image only using NNs. 
- Post-Processing methods: a "denoising" network. Takes a noisy recon and cleans it.
- Iterative Unrolled methods: Uses the operator to imitate iterative recon algorithms, but has learned parts.
- Learned regularizer: Explicitly learned regularization functions.
- Plung and Play (PnP): Implicit learned regularization, a regularization optimization step is learned, rather than an explicit one. 

On top of these, there are some techniques that don't fit this classification, particularly because they reffer to _modes of training_ rather than methodology. For these, often the model is not the important part, but the way of training. We will mention them here anyway, to avoid information fragmentation. 
They tend to be either

- Self-supervised networks: Uses noisy data to self train and obtain noisseless recosntruction.
- Unsupervised: Do not use training (or train directly on the test data).



LION supports the following models for each category

## Fully-learned

`None` are supported. A good model that does this is not well known. Feel free to suggest any. 

## Post-Processing:

- `FBPConvNet`: A Unet that denoises a bad FBP recon. [_Jin, Kyong Hwan, et al._, **Deep convolutional neural network for inverse problems in imaging**, IEEE Transactions on Image Processing, 2017](https://ieeexplore.ieee.org/document/7949028)

## Iterative Unrolled:

- `Learned Primal Dual (LPD)`: Based on the PDHG algorithm. [_Adler, Jonas, and Öktem, Ozan._ **Learned primal-dual reconstruction**, IEEE transactions on medical imaging, 2018](https://ieeexplore.ieee.org/document/8271999)
- `ItNet`: Winner of AAPM DL-Sparse-View CT Challenge. [_Genzel, Martin et al._, **Near-exact recovery for tomographic inverse problems via deep learning**, ICML 2022](https://proceedings.mlr.press/v162/genzel22a.html)

## Learned Regularizer

WIP

## Plung and Play

WIP


## CNNs

- `Mixed Scale - Dense (MS-D)`: An alternative to a Unet for imaging, where no downsampling happens. Performs equally with less parameters. [_Pelt, Daniël M., and James A. Sethian_. **"**A mixed-scale dense convolutional neural network for image analysis.**"** Proceedings of the National Academy of Sciences, 2018](https://www.pnas.org/doi/abs/10.1073/pnas.1715832114)

# Other methods

As said above, some methods for recon are defined not by the model structure, but by the training type only, and can be modelled by most NNs. These are the ones currently supported:

## Self-supervised methods

- `Noise2Inverse`: Inverse-problems version of Noise2Noise. The original work uses a [MS-D network](https://www.pnas.org/doi/full/10.1073/pnas.1715832114), but any regression model is valid. This is implemented via `LION.optimizers.Noise2Inverse_solver`. [_Hendriksen, Allard Adriaan et al._ **Noise2inverse: Self-supervised deep convolutional denoising for tomography**, IEEE Transactions on Computational Imaging, 2020](https://ieeexplore.ieee.org/abstract/document/9178467)

## Unsupervised methods

WIP


