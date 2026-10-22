# Noise study: relative vs physical noise model

LION's `ct_utils.sinogram_add_noise` can scale line integrals in two ways
before it simulates photon counts:

- **relative** (`mass_attenuation=None`, the default): each sinogram is divided
  by its own maximum. The most attenuating ray of every slice keeps `I0/e`
  photons, whatever the size of the object.
- **physical** (`mass_attenuation=ct.WATER_MASS_ATTENUATION[60]`): the real
  transmission of a density image, in mm. `I0` is the number of photons per
  detector pixel in vacuum, as in XCIST. Larger objects get noisier data, and
  the thickest rays can run out of photons.

The scripts in this folder give the evidence needed to choose between the two,
and the physical `I0` values that would replace the current ones.

| script | needs | what it answers |
|---|---|---|
| `compare_models_water.py` | CPU, ~10 s | How do the two models differ on water cylinders of 100–450 mm? Which physical I0 gives the same image noise? |
| `calibrate_lidc.py` | GPU, tomosipo, LIDC-IDRI, ~10 min | Under the relative model, how much does the effective dose change from slice to slice? Which physical I0 keeps each experiment as hard on average? |
| `detect_measured_noise.py` | 2DeteCT data, CPU, ~15 min | On real measured data, does the noise follow one I0 per scanner (physical model) or one per slice (relative model)? Which simulator I0 matches mode1? |

`noise_tools.py` holds the shared functions. They are tested on synthetic
data in `pytests/test_noise_study.py`: the variance formula against LION's
simulation, the noise estimator recovering a known I0 and sigma, and the
model-consistency test.

Run each script from anywhere, with LION installed:

```
python scripts/noise_study/compare_models_water.py
python scripts/noise_study/calibrate_lidc.py --out results/
python scripts/noise_study/detect_measured_noise.py --out results/
```

## How to read the results

These are the criteria we proposed for the decision. They were set down
before the results were known.

1. **Measured data (`detect_measured_noise.py`).** For each detector, the
   physical fit of mode1 should give the same I0 for every slice. Concretely,
   its spread (CV) should be within ~10 %, it should not correlate with the
   slice's maximum, and its misfit should be a few %. The relative fit should
   spread widely, or fit clearly worse, or both. The `mode1 - mode2` check
   should be close to 1, which validates the estimator. The "simulator I0"
   for mode1 should be close to the 200 that Kiss et al. (2024) found. If all
   of this holds, real data follow the physical model.
2. **LIDC-IDRI (`calibrate_lidc.py`).** The 5–95 % range of the equivalent
   physical I0 shows how much the relative model changes the dose from one
   slice to the next. A range wider than about 2× means the current
   experiments mix very different doses. The medians are the physical I0
   values that keep each experiment as hard as it is now, on average.
3. **Water cylinders (`compare_models_water.py`).** This shows the mechanism.
   It shows where each model puts the noise, and when the physical model
   reaches photon starvation, which the relative model never reaches.

If 1 and 2 hold, we can switch the experiments to the physical model with the
I0 values from 2. Old parameter files then keep the relative model, because a
missing `mass_attenuation` means the old behaviour.

## Known limits of the physical model

These limits are shared with most published simulators of this kind:

- It is monochromatic, so there is no beam hardening.
- It is water-equivalent: density times the mass attenuation of water.
- It has no bowtie filter.
- It has no tube-current modulation. Real scanners adapt the dose to the
  patient (AEC), which the relative model imitates crudely and at its
  extreme.
- Photon starvation needs a choice before the log. Photon counts are never
  negative, but they can be 0. The detector signal that LION simulates (an
  energy-integrating detector: photons plus Gaussian electronic noise, after
  the dark field) can be 0 or negative, as real raw data can at ultra-low dose
  (Ding et al. 2018; Ye et al. 2019). LION replaces every value <= 0 with
  1e-6. That gives line integrals of log(I0) + 13.8, far beyond any real
  attenuation, and these become the streaks. LoDoPaB-CT has no electronic
  noise and replaces 0 photons with 0.1. Scanners apply their own pre-log
  corrections. The value used sets how strong starvation streaks are.
  Changing it is a separate decision. It never matters for the relative
  model, where no ray drops below I0/e.
