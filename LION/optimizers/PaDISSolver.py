class PaDISSolver(LIONSolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: Callable,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        solver_params: Optional[SolverParams] = None,
        save_folder: Optional[pathlib.Path] = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry,
            verbose,
            device,
            solver_params=solver_params,
            save_folder=save_folder,
        )

    def mini_batch_step(self, sino_batch, target_batch) -> torch.Tensor:
        """
        This performs a single step of the optimization and returns the loss.
        The inputted tensors are already on the correct device.
        """

        batch_size = sino_batch.shape[0]


def add_noise_to_data(data, noising_step):
    pass
