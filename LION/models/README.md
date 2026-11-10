# Models

LION's models, by kind (post-processing, unrolled iterative, learned FBP,
CNNs and denoisers, PnP, learned regularisers), what each takes, how each is
trained and reconstructs, and their papers:
[Methods in LION](https://cambridgecia.github.io/LION/methods.html)
(source: [`docs/methods.rst`](../../docs/methods.rst)).

To write a model: derive from `LIONmodel` (`LIONmodel.py`), as in the
tutorial on extending LION (`demos/d13_extending_lion.py`); the rules are in
the contributor guide's Models section.
