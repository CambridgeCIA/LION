<s>
- Visualise real data
- Add top 32 lines of the measurement file into the slides
- use actual values in those lines to make the examples
- Run photocurrent script on real data for a few cases only
- Tweak PnP ADMM a bit to get better results for those cases
- Plot the masks like the butterfly slides
</s>

- Lead with the problem formula $\hat{x} = \argmin{x} \| Hx - y \| + R(x)$ (no need the scalar for the regularization term?)
- Compare LR and PnP: learn the regulariser explicitly vs. implicitly
    - focus more on PnP, only mention that learned regulariser doesn't involve
    an objective function shift?
- SPGL1:
    - show what the transformed objective function is
    - mention sparsity, performance depends on how compressible the image is,
    which is usually the case in PCM where the images are made up of smooth regions
    - will discuss, formalise, validate, and improve performance with GPU acceleration
