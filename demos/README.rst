Tutorials
=========

The demos of LION, as tutorials: each builds on the last, from the CT
operator to publishing a trained model, and together they cover most of
LION's API. Each is a Python file with ``# %%`` cells, to run whole, a cell at
a time in an editor that knows cells (VS Code, Spyder), or as a notebook
(download it below each page).

They need a GPU (LION's CT operators use tomosipo and ASTRA) and, from the
data tutorial on, the processed LIDC-IDRI data in ``LION_DATA_PATH``
(:doc:`/datasets` says how to get it).

The models trained in the tutorials are toys, trained for a few epochs on a
few slices to show how LION is used: their results are poor. Trained as their
papers train them, they do what the papers report.
