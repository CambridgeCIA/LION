# Learned denoising with simulated and experimental low-dose CT data

**Work in progress:** we are working on full reproducibility of this paper. For now this folder holds the paper's data setup (`data.py`), not its training scripts.

Kiss, Maximilian B., Ander Biguri, Carola-Bibiane Schönlieb, K. Joost Batenburg, and Felix Lucka. "Learned denoising with simulated and experimental low-dose CT data." *Scientific Reports* 16 (2025). [doi:10.1038/s41598-025-30457-9](https://doi.org/10.1038/s41598-025-30457-9) ([arXiv:2408.08115](https://arxiv.org/abs/2408.08115))

The paper compares deep-learning denoisers trained on **simulated** low-dose data with ones trained on **experimental** low-dose data, using the 2DeteCT dataset:

- **Simulated:** the high-dose mode 2 sinograms with added noise (Poisson and 5% detector cross-talk), with one I0 = 200 for every slice.
- **Experimental:** the measured low-dose mode 1 sinograms (tube current 33.3 µA instead of 1000 µA).
- **Target:** mode 2.
- **Tasks:** sinogram denoising, and end-to-end training with an FBP layer.
- **Networks:** U-Net and MSD-Net, trained in each setting and tested on both the simulated and the experimental test sets.

`data.py` gives the 2DeteCT dataset of each setting:

```python
from data import dataset

train = dataset("simulated", "sino2recon", "train")   # or "experimental", "sino2sino"
test_experimental = dataset("experimental", "sino2recon", "test")
```

The simulated noise uses one I0 for every slice, as the paper did: `deteCT.noise_parameters(I0=200)`. LION's default for the 2DeteCT loader is now one I0 per detector, so it is set explicitly here.

Not reproduced yet:
- **Intensity domain:** the paper denoises the beam intensity loss images (before the negative logarithm), while LION's 2DeteCT loader gives log sinograms by default.
- **Training details:** e.g. the learning rate and batch size.
