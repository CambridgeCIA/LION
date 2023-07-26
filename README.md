# AI tools for learned tomographic reconstruction
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

University of Cambridge Computational Image Analysis (CIA) groups AI tools for tomographic reconstruction.


**WARNING** Building in progress.

Install: 

```
git clone https://github.com/CambridgeCIA/AItomotools.git
cd AItomotools
conda env create --file=env.yml
git submodule update --init --recursive
conda activate aitools
python setup.py install
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


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AnderBiguri"><img src="https://avatars.githubusercontent.com/u/11854388?v=4?s=100" width="100px;" alt="Biguri"/><br /><sub><b>Biguri</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=AnderBiguri" title="Code">💻</a> <a href="#design-AnderBiguri" title="Design">🎨</a> <a href="#ideas-AnderBiguri" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tutorial-AnderBiguri" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://emilienvalat.net/"><img src="https://avatars.githubusercontent.com/u/46785587?v=4?s=100" width="100px;" alt="Emilien Valat"/><br /><sub><b>Emilien Valat</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=Emvlt" title="Code">💻</a> <a href="#design-Emvlt" title="Design">🎨</a> <a href="#ideas-Emvlt" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-Emvlt" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ferdiasherry.com"><img src="https://avatars.githubusercontent.com/u/12610714?v=4?s=100" width="100px;" alt="Ferdia"/><br /><sub><b>Ferdia</b></sub></a><br /><a href="#design-fsherry" title="Design">🎨</a> <a href="#tool-fsherry" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/michi217"><img src="https://avatars.githubusercontent.com/u/62284237?v=4?s=100" width="100px;" alt="michi217"/><br /><sub><b>michi217</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=michi217" title="Code">💻</a> <a href="#data-michi217" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/OliverCoughlan"><img src="https://avatars.githubusercontent.com/u/39098447?v=4?s=100" width="100px;" alt="Oliver Coughlan"/><br /><sub><b>Oliver Coughlan</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=OliverCoughlan" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->