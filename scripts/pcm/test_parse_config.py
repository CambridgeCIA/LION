import tyro

from LION.pcm.config import ExperimentConfig


class CustomClassNumpyDocstrings:
    def __init__(self, x: int, y: str = "default") -> None:
        """A custom class for testing tyro CLI parsing.

        Parameters
        ----------
        x : int
            A required integer parameter.
        y : str, optional
            An optional string parameter with a default value, by default "default".
        """
        self.x = x
        self.y = y


class CustomClassGoogleDocstrings:
    def __init__(self, a: float, b: bool = False) -> None:
        """A custom class for testing tyro CLI parsing.

        Args:
            a (float): A required float parameter.
            b (bool, optional): An optional boolean parameter with a default value, by default False.
        """
        self.a = a
        self.b = b


class NestedClassMixedDocstrings:
    def __init__(
        self, c: CustomClassNumpyDocstrings, d: CustomClassGoogleDocstrings
    ) -> None:
        """A nested class for testing tyro CLI parsing.

        Parameters
        ----------
        c : CustomClassNumpyDocstrings
            An instance of CustomClassNumpyDocstrings.
        d : CustomClassGoogleDocstrings
            An instance of CustomClassGoogleDocstrings.
        """
        self.c = c
        self.d = d


if __name__ == "__main__":
    cfg = tyro.cli(ExperimentConfig)
    # cfg = tyro.cli(CustomClassNumpyDocstrings)
    # cfg = tyro.cli(CustomClassGoogleDocstrings)
    # cfg = tyro.cli(NestedClassMixedDocstrings)
    # preset_args, remaining_args = tyro.cli(
    #     PresetArgs,
    #     return_unknown_args=True,
    # )
    # if preset_args.preset is None:
    #     cfg = tyro.cli(ExperimentConfig, args=remaining_args)
    # else:
    #     preset_cfg = get_preset_config(preset_args.preset)
    #     cfg = tyro.cli(
    #         ExperimentConfig,
    #         args=remaining_args,
    #         default=preset_cfg,
    #     )

    print(cfg)
