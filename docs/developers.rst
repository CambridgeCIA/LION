Developer guide
===============

How LION is built, for changing it or adding to it. The
:doc:`tutorials <tutorials/index>` show how to *use* each piece, and
:doc:`tutorials/d13_extending_lion` how to write your own in a script. This
page follows one training through the code, says where everything lives, and
lists what a new piece of LION needs. The rules the code follows (names,
parameters, errors, tests) are in the `contributor guide
<https://github.com/CambridgeCIA/LION/blob/main/contributor-guide.md>`_; what
LION records and how, in `metadata.md
<https://github.com/CambridgeCIA/LION/blob/main/metadata.md>`_.

The idea
--------

LION is plain torch with a few small base classes. A model is a
``torch.nn.Module``, an operator a function of tensors, a solver a training
loop, a dataset a torch ``Dataset``. The base classes add only what research
needs to be repeated:

- **every piece is made from parameters.** A ``LIONParameter`` (a namespace
  that saves to JSON) holds everything that decides what the piece does, and
  ``default_parameters()`` lists them all, with their defaults;
- **every piece can be made again from its file.** A piece keeps its
  constructor's arguments under the same names, so LION writes it as the call
  that makes it (a *spec*: ``{"_target_": "LION.models...", ...}``) and
  ``instantiate`` makes it again;
- **every saved model carries its records**: the data it was trained on, how
  it was trained, and the code and machine it ran on.

Anything else (a loss, a metric, a scheduler, a data augmentation) is plain
torch or a plain function. Before adding a class, check that a function or a
parameter doesn't do.

How the pieces connect
----------------------

::

    OperatorParameters ──make_operator()──▶ Operator
      (CT: Geometry)                          (CT: CTProjectionOp)
         │                                       forward, adjoint, autograd(),
         │                                       default_reconstruction (FDK)
         ├──▶ Experiment ──get_dataset(split)──▶ LIONDataset ──▶ DataLoader
         │      geometry, noise, data split        (y, x) pairs       │
         │                                                            ▼
         ├──▶ LIONmodel (the network)  ◀──trains──  LIONsolver (the loop)
         │                                │            mini_batch_step(y, x)
         │                                ▼
         │                          model.save: .pt (weights) + .json (records)
         │                                │
         └──▶ LIONReconstructor (model inside an algorithm) ──▶ Evaluation

The operator's parameters are what everything shares: experiments,
models, solvers and reconstructors take them (``operator_params``; for CT, the
geometry) and make the operator themselves, so they work with any kind of
operator. Each kind of operator (``LION/operators/ct``, ``pcm``) has its
parameters, its operator, and the reconstructions only it has.

Following one training
----------------------

The README's first training, step by step, with where each step happens.

.. code-block:: python

    experiment = ct_experiments.LowDoseCTRecon()
    training = experiment.get_dataset("train")
    model = FBPConvNet(experiment.operator_params)
    solver = SupervisedSolver(model, torch.optim.Adam(model.parameters()),
                              torch.nn.MSELoss())
    solver.set_training(DataLoader(training, batch_size=4, shuffle=True))
    solver.set_saving(folder, "FBPConvNet")
    solver.train(10)
    solver.save_final_results()

**The experiment** (``LION/experiments/ct_experiments.py``). ``Experiment``
holds a ``LIONParameter`` with the geometry (``operator_params``), the noise
(``noise_params``) and the data loader's settings; each experiment
(``LowDoseCTRecon``, ...) is only its ``default_parameters()``.
``get_dataset("train")`` makes the data loader (``LIDC_IDRI``) for that split
and gives it the experiment's noise as a sinogram transform, then stores the
experiment and the split on it, for its record.

**The data** (``LION/data_loaders/LIDC_IDRI.py``). ``__getitem__`` reads a
slice (a density image), projects it with the experiment's operator, and adds
noise drawn inside ``rng.seeded(seed, sample)``: each sample is the same every
time it is read. It returns ``(sinogram, image)``: the measurements and the
target. ``sample_ids()`` and ``sample_groups()`` (patients) let the record say
what was used, and evaluations check that no patient is in two splits.

**The model** (``LION/models/LIONmodel.py``, then
``LION/models/post_processing/FBPConvNet.py``). ``LIONmodel.__init__``
checks the arguments, fills ``model_parameters`` from
``default_parameters()`` (warning on names it doesn't know), stores torch's
random state (so the initial weights can be made again) and the
normalisation. The model then makes its operator with
``self._make_operator()`` (``autograd=True`` gives ``self.A``, ``self.AT``
when gradients go through it) and builds its layers.
``model_parameters.model_input_type`` says what ``forward`` takes:
measurements (``SINOGRAM``: FBPConvNet, which runs FDK itself; LPD) or an
image (``IMAGE``: the CNNs, given a reconstruction).

**The solver** (``LION/optimizers/LIONsolver.py``, then
``SupervisedSolver.py``). ``LIONsolver.__init__`` takes the operator's
parameters from the model (or its argument), makes ``self.op``, moves the model
to ``default_device()``, and replaces a ``"default"`` reconstruction in
``solver_params`` by the real function, so the record names it.
``set_training``, ``set_validation``, ``set_saving`` and
``set_checkpointing`` only store what they're given.

``train(n_epochs)``:

1. ``check_training_ready()``, then the loss arrays, the checkpoint defaults,
   and the latest checkpoint if there is one (``load_checkpoint``: it then
   continues from that epoch; ``n_epochs`` is the total);
2. a new run seeds torch, numpy and ``random`` (``seed``), and records how the
   initial weights can be made again (``_initial_weights``);
3. each epoch, ``epoch_step``: ``train_step`` (for each mini-batch:
   ``mini_batch_step(y, target)``, backward, optimizer step), then
   ``validate()`` when due, saving the best model, then the scheduler;
4. a checkpoint every ``checkpoint_freq`` epochs (``save_checkpoint``).

``SupervisedSolver.mini_batch_step`` is three lines: the output
(``self._forward(y)``), the loss against the target, and
``self._regularized(...)``, which adds ``model_regularization`` if there is
one. ``_forward`` is ``model_forward(model, y, op)`` (in ``LIONsolver.py``):
an image model gets the initial reconstruction (FDK for CT) first, and the
model's normalisation is applied around it. That function is how a model is
run everywhere: training, validation, testing, ``ForwardReconstructor``.

**Saving** (``LIONmodel.save``). ``save_final_results()`` calls
``model.save(file, training=..., dataset=..., loss=..., epoch=...)``, which
writes two files: ``file.pt``, the weights and plain numbers only (loaded with
``weights_only=True``); and ``file.json``, the model's spec
(``LION/utils/spec.py``), the environment record
(``LION/utils/environment.py``), the data record
(``LION/data_loaders/data_record.py``) and the training record
(``LION/optimizers/training_record.py``). ``LIONmodel.load(file)`` reads them
back; ``LION.package`` builds on them to publish, rebuild and retrain.

**Using it** (``LION/reconstructors``, ``LION/evaluation``).
``ForwardReconstructor(geometry, model).reconstruct(y)`` runs
``model_forward`` without gradients. ``LIONReconstructor.reconstruct`` takes
one sample or a batch, and calls ``reconstruct_sample`` for each; PnP and the
learned regulariser are algorithms there, with the model inside.
``Evaluation`` runs a method on every test sample, and ``Comparison`` compares
methods against the operator's default reconstruction.

Where things live
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Folder
     - What it has
   * - ``LION/operators``
     - ``Operator``, generic operators; ``ct/`` (geometry, projector, FDK,
       SIRT, TV, noise, HU), ``pcm/``
   * - ``LION/classical_algorithms``
     - algorithms for any operator: CG, FISTA, SPGL1
   * - ``LION/data_loaders``
     - ``LIONDataset``, the datasets, ``data_record``; each dataset's
       download and preprocessing scripts
   * - ``LION/experiments``
     - ``Experiment`` and LION's CT experiments; the 2DeteCT benchmark's
   * - ``LION/models``
     - ``LIONmodel``, ``TorchModel``; the models by kind (``CNNs``,
       ``post_processing``, ``iterative_unrolled``, ``learned_fbp``,
       ``learned_regularizer``, ``PnP``)
   * - ``LION/optimizers``
     - ``LIONsolver``, the solvers, ``training_record``
   * - ``LION/losses``, ``LION/metrics``
     - losses (SURE, Jacobian penalties); PSNR, SSIM, HaarPSI
   * - ``LION/reconstructors``
     - ``LIONReconstructor``, ``ForwardReconstructor``, ``PnP``,
       ``LearnedRegularizerReconstructor``
   * - ``LION/evaluation``
     - ``Evaluation``, ``Comparison``, tables and figures
   * - ``LION/utils``
     - ``parameter`` (``LIONParameter``, ``OperatorParameters``), ``spec``,
       ``paths``, ``device``, ``rng``, ``normaliser``, ``citation``,
       ``environment``, the backend imports
   * - ``LION/package.py``
     - ``publish``, ``load``, ``build``, ``train_again``
   * - ``pytests/``
     - the tests, flat; ``helpers.py`` (small operators and models),
       ``conftest.py`` (markers, fixtures)
   * - ``demos/``, ``scripts/``
     - the tutorials (the site's gallery); example and paper scripts
   * - ``docs/``
     - this site

Adding to LION
--------------

What each kind of piece defines, and what to do so it is part of LION. In
every case: a docstring (its arguments, its paper, where it deviates), a
``cite()`` if it comes from a paper, one test of the feature, and a line in
:doc:`methods`. The API reference finds new modules by itself.

A model
~~~~~~~

- In ``LION/models/<kind>/``, deriving from ``LIONmodel``.
- ``__init__(self, operator_params, model_parameters=None)`` (or only
  ``model_parameters`` if it uses no operator); call
  ``super().__init__(model_parameters, operator_params)`` first, then
  ``self._make_operator(...)`` if it uses one, then the layers from
  ``self.model_parameters``.
- ``default_parameters()``: every setting, the paper's values, and
  ``model_input_type``.
- ``forward``: a batch. No normalisation inside (callers apply it).
- Follow the paper and the authors' code; say where it differs.

A solver
~~~~~~~~

- In ``LION/optimizers/``, deriving from ``LIONsolver``, with the standard
  arguments in their order (``model, optimizer, loss_fn, operator_params,
  solver_params, model_regularization, scheduler, scheduler_step, verbose,
  device, seed, deterministic``), passed to ``super().__init__``.
- ``mini_batch_step(y, target)``: the loss of one mini-batch, through
  ``self._forward`` and ``self._regularized``.
- ``default_parameters()``: its settings. Values computed during training (an
  estimated noise level) start as ``None`` there, so they are recorded and
  resumed.
- Override ``_forward`` only when the method reconstructs differently (e.g.
  Noise2Inverse's splits); it is then used in validation and testing too.
- Randomness of training (noise it adds) comes from torch's generator, which
  the solver seeds; noise added in validation goes inside
  ``rng.seeded(...)``.

A reconstructor
~~~~~~~~~~~~~~~

- In ``LION/reconstructors/``, deriving from ``LIONReconstructor``.
- ``default_parameters()``: every setting of the algorithm;
  ``reconstruct_sample(y, x0, verbose)``: one sample, reading
  ``self.parameters``. ``@torch.no_grad()`` unless it needs gradients.
- The base class gives ``reconstruct`` (batches), ``save``/``load`` and
  ``instantiate``.

A kind of operator
~~~~~~~~~~~~~~~~~~

- A folder ``LION/operators/<kind>/``: its parameters (deriving from
  ``OperatorParameters``, with ``make_operator()``), its operator (deriving
  from ``Operator``: ``forward``, ``adjoint``, ``domain_shape``,
  ``range_shape``), and its reconstructions.
- ``autograd()`` if gradients can go through it; ``default_reconstruction_fn()``
  if it has a usual reconstruction (the baseline, and what image models get);
  ``split_count``/``split``/``split_measurements`` on its parameters if
  the self-supervised splitting methods apply.
- Its backend, if it has one, is imported only in its folder, through a
  ``LION/utils/<backend>_import.py``.
- Test ``adjoint`` with ``helpers.dotproduct_adjointness_test``.

A dataset or an experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~

- A dataset derives from ``LIONDataset``, keeps its arguments as attributes
  of the same names, lists its samples (``sample_ids()``,
  ``sample_files()``, ``sample_groups()``), and draws its random numbers
  inside ``rng.seeded(seed, sample)``. Download and preprocessing scripts go
  in ``LION/data_loaders/<dataset>/``; data under ``LION_DATA_PATH``. Check it
  against the real data, not only fake data.
- An experiment derives from ``Experiment`` and is its
  ``default_parameters()``. Changing an existing experiment changes published
  results: add a parameter for the change, whose absence means the old
  behaviour (contributor guide).

A metric or a loss
~~~~~~~~~~~~~~~~~~

- A ``torch.nn.Module``; ``(output, target)``. A metric takes
  ``(x, target, reduce=None, batched=True, data_range=None)``, returns one
  score per sample, and sets ``higher_is_better``.

Checking your change
--------------------

- **Tests:** ``pytest pytests/ -n auto`` (CPU; CI runs these). Tests that need
  tomosipo or a GPU are marked and skipped elsewhere: run them on a GPU
  machine when you touch CT code. For CT logic without tomosipo, use the
  matrix operators in ``pytests/helpers.py``.
- **Format:** ``pre-commit run --files <files>`` (Black). New Python files
  start with the licence header (``pytests/test_licence_headers.py``).
- **The site:** ``sphinx-build -W -b html docs docs/_build/html`` (CI fails
  on any warning: docstrings are reStructuredText). Tutorials only run with
  ``LION_DOCS_RUN_DEMOS=1``, on a GPU machine with the data;
  ``utils/publish_docs.sh`` builds and publishes the site from there.
- **Scripts and tutorials** you changed: run them. A new feature that users
  will look for gets a section in a tutorial, or an example script.
