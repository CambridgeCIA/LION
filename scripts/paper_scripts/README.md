# Noise Paper Codes
This is a collection of Python scripts for reproducing the experiments as described in

Maximilian B. Kiss, Ander Biguri, Carola-Bibiane Schönlieb, K. Joost Batenburg, and Felix Lucka "Learned denoising with simulated and experimental low-dose CT data"


* ` noise_simulation_analysis.py `was used for the empricial selection of noise level for the simulated noisy data and to produce the FIgures 2 and 3 in the paper.
*  ` noise_paper_MSDNet_Exp.py `, ` noise_paper_MSDNet_Art.py `, ` noise_paper_UNet_Exp.py `, and ` noise_paper_UNet_Art.py `, were used to train the sinogram denoising algorithms.
* ` noise_paper_FBPMSDNet_Exp.py `, ` noise_paper_FBPMSDNet_Art.py `, ` noise_paper_FBPUNet_Exp.py `, and ` noise_paper_FBPUNet_Art.py `, were used to train the end-to-end denoising algorithms.
* ` min_val_NoiseExps_script.py ` was used to find the checkpointed models with minimal validation loss.
* ` slice_extraction_allExp_noisePaper.py ` was used to extract the slices for the qualitative analysis as ` .npy `-files.
* ` Qualitative_Analysis_Figure_NoisePaper.py ` and ` Qualitative_Analysis_Reference_Figure_NoisePaper.py ` were used to produce the Figures 4 and 5 in the paper.


## Requirements

* Most of the above scripts make use of the [ASTRA toolbox](https://www.astra-toolbox.com/). If you are using conda, this is available through the `astra-toolbox/` channel.

## Contributors

Maximilian Kiss (maximilian.kiss@cwi.nl), CWI, Amsterdam, Ander Biguri (ander.biguri@gmail.com), CIA, Cambridge

