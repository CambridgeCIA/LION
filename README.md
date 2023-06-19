# AI tools for learned tomographic reconstruction

University of Cambridge Computational Image Analysis (CIA) groups AI tools for tomographic reconstruction.


**WARNING** Building in progress.

Install: 

```
conda env create --name aitomotools --file=env.yml
```

Optional, if you want pre-commits. 
Install pre commits for auto-formating your commits.
Highly suggested if you want reproducibility, this will auto-save changes in your conda enviroments and will update your conda enviroment when pulling

```
conda activate aitomotools
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type post-merge
```


# Developers:
Read `developers.md`
