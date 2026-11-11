# Datasets

How to download and prepare the datasets LION uses (LIDC-IDRI, 2DeteCT,
Walnuts), and where LION looks for them (`LION_DATA_PATH`), is on the
documentation site: <https://cambridgecia.github.io/LION/datasets.html>, with
a page per dataset (source: `docs/datasets.rst` and `docs/datasets/`).

Each dataset's scripts are in its folder here: `LIDC_IDRI/` (download and
preprocessing; its README has the download script's options), `2deteCT/`,
`walnuts/`. The loaders are `LIDC_IDRI.py` and `deteCT.py`; their docstrings
describe their parameters.
