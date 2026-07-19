"""Citation support for the PaDIS reconstructor."""

from __future__ import annotations


class PaDISCitations:
    """Provide citations for PaDIS and its sampling algorithms."""

    @staticmethod
    def cite(cite_format: str = "MLA") -> None:
        """Print citations for PaDIS and its implemented sampling components.

        The citation set covers the PaDIS patch prior, DPS, annealed Langevin
        dynamics, predictor-corrector sampling, DDNM, EDM conventions, and FDK
        initialisation. Use ``cite_format="bib"`` for importable BibTeX.
        """
        citations = {
            "MLA": (
                "Hu, Jason, Bowen Song, Xiaojian Xu, Liyue Shen, and Jeffrey "
                'Fessler. "Learning Image Priors Through Patch-Based Diffusion '
                'Models for Solving Inverse Problems." Advances in Neural '
                "Information Processing Systems, vol. 37, pp. 1625-1660, 2024. "
                "doi:10.52202/079017-0052.",
                "Chung, Hyungjin, Jeongsol Kim, Michael Thompson McCann, Marc "
                'Louis Klasky, and Jong Chul Ye. "Diffusion Posterior Sampling '
                'for General Noisy Inverse Problems." The Eleventh International '
                "Conference on Learning Representations, 2023.",
                'Song, Yang, and Stefano Ermon. "Generative Modeling by '
                'Estimating Gradients of the Data Distribution." Advances in '
                "Neural Information Processing Systems, vol. 32, 2019.",
                "Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek "
                'Kumar, Stefano Ermon, and Ben Poole. "Score-Based Generative '
                'Modeling through Stochastic Differential Equations." '
                "International Conference on Learning Representations, 2021.",
                'Wang, Yinhuai, Jiwen Yu, and Jian Zhang. "Zero-Shot Image '
                'Restoration Using Denoising Diffusion Null-Space Model." The '
                "Eleventh International Conference on Learning Representations, "
                "2023.",
                "Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine. "
                '"Elucidating the Design Space of Diffusion-Based Generative '
                'Models." Advances in Neural Information Processing Systems, '
                "vol. 35, pp. 26565-26577, 2022. doi:10.52202/068431-1926.",
                'Feldkamp, L. A., L. C. Davis, and J. W. Kress. "Practical '
                'Cone-Beam Algorithm." JOSA A, vol. 1, no. 6, pp. 612-619, 1984. '
                "doi:10.1364/JOSAA.1.000612.",
            ),
            "bib": (
                """@inproceedings{hu_learning_2024,
  title = {Learning Image Priors Through Patch-Based Diffusion Models for Solving Inverse Problems},
  booktitle = {Advances in Neural Information Processing Systems},
  author = {Hu, Jason and Song, Bowen and Xu, Xiaojian and Shen, Liyue and Fessler, Jeffrey},
  year = {2024},
  volume = {37},
  pages = {1625--1660},
  publisher = {Curran Associates, Inc.},
  doi = {10.52202/079017-0052}
}""",
                """@inproceedings{chung_diffusion_2022,
  title = {Diffusion Posterior Sampling for General Noisy Inverse Problems},
  booktitle = {The Eleventh International Conference on Learning Representations},
  author = {Chung, Hyungjin and Kim, Jeongsol and McCann, Michael Thompson and
    Klasky, Marc Louis and Ye, Jong Chul},
  year = {2023},
  url = {https://openreview.net/forum?id=OnD9zGAGT0k}
}""",
                """@inproceedings{song_generative_2019,
  title = {Generative Modeling by Estimating Gradients of the Data Distribution},
  booktitle = {Advances in Neural Information Processing Systems},
  author = {Song, Yang and Ermon, Stefano},
  year = {2019},
  volume = {32},
  publisher = {Curran Associates, Inc.},
  url = {https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html}
}""",
                """@inproceedings{song_score-based_2020-1,
  title = {Score-Based Generative Modeling through Stochastic Differential Equations},
  booktitle = {International Conference on Learning Representations},
  author = {Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P. and
    Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  year = {2021},
  url = {https://openreview.net/forum?id=PxTIG12RRHS}
}""",
                """@inproceedings{wang_zero-shot_2022-1,
  title = {Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model},
  booktitle = {The Eleventh International Conference on Learning Representations},
  author = {Wang, Yinhuai and Yu, Jiwen and Zhang, Jian},
  year = {2023},
  url = {https://openreview.net/forum?id=mRieQgMtNTQ}
}""",
                """@inproceedings{karras_elucidating_2022-1,
  title = {Elucidating the Design Space of Diffusion-Based Generative Models},
  booktitle = {Advances in Neural Information Processing Systems 35},
  author = {Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  year = {2022},
  pages = {26565--26577},
  publisher = {Neural Information Processing Systems Foundation, Inc.},
  doi = {10.52202/068431-1926}
}""",
                """@article{feldkamp_practical_1984,
  title = {Practical Cone-Beam Algorithm},
  author = {Feldkamp, L. A. and Davis, L. C. and Kress, J. W.},
  year = {1984},
  journal = {JOSA A},
  volume = {1},
  number = {6},
  pages = {612--619},
  doi = {10.1364/JOSAA.1.000612}
}""",
            ),
        }
        if cite_format not in citations:
            raise ValueError(
                f'`cite_format` "{cite_format}" is not understood, only "MLA" '
                'and "bib" are supported'
            )
        print("\n\n".join(citations[cite_format]))
