# Paper scripts

These files should EXACTLY (or almost exactly) reproduce experiments from papers, using LION
If you are using some model from a paper, but it is not EXACTLY the training from the paper, please add it to "example_scripts" folder


## Continous Learned Primal Dual

Runkel, Christina, Ander Biguri, and Carola-Bibiane Schönlieb. "Continuous Learned Primal Dual." arXiv preprint arXiv:2405.02478 (2024).

Neural ordinary differential equations (Neural ODEs) propose the idea that a sequence of layers in a neural network is just a discretisation of an ODE, and thus can instead be directly modelled by a parameterised ODE. This idea has had resounding success in the deep learning literature, with direct or indirect influence in many state of the art ideas, such as diffusion models or time dependant models. Recently, a continuous version of the U-net architecture has been proposed, showing increased performance over its discrete counterpart in many imaging applications and wrapped with theoretical guarantees around its performance and robustness. In this work, we explore the use of Neural ODEs for learned inverse problems, in particular with the well-known Learned Primal Dual algorithm, and apply it to computed tomography (CT) reconstruction.

